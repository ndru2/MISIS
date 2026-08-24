"""Отчёт по стадии очистки.

Отчёт нужен не для отчётности. Пороги отбора нельзя выбрать умозрительно: надо
видеть, что именно выбрасывается на каждом значении. Поэтому главный раздел
здесь — распределение оценки шума по полосам с примерами текста из каждой: по
нему порог сдвигается осознанно. Второй по важности — документы с наибольшей
долей потерь: если из книги выброшено три четверти блоков, дело не в порогах, а
в том, что документ разобран плохо.
"""

import json

import duckdb

from . import config


def _cell(value, limit=None) -> str:
    """Готовит значение к вставке в таблицу Markdown."""
    text = '' if value is None else str(value)
    if limit and len(text) > limit:
        text = text[:limit] + '…'
    return text.replace('|', '\\|').replace('\n', ' ⏎ ').strip()


def _table(headers, rows) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |',
             '|' + '|'.join('---' for _ in headers) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(_cell(value) for value in row) + ' |')
    return '\n'.join(lines)


def _share(part, whole) -> str:
    return f'{part / whole:.1%}' if whole else '—'


def build(clean_path=None, out_md=None, out_json=None, cfg=config.DEFAULT) -> dict:
    """Собирает отчёт по таблице очищенных блоков."""
    clean_path = str(clean_path or config.CLEAN_BLOCKS)
    out_md = out_md or config.AUDIT_MD
    out_json = out_json or config.AUDIT_JSON

    extras = json.loads(config.EXTRAS.read_text(encoding='utf-8'))

    connection = duckdb.connect()
    connection.execute(
        f"CREATE VIEW blocks AS SELECT * FROM read_parquet('{clean_path}')")

    def query(sql, params=()):
        return connection.execute(sql, params).fetchall()

    total, kept, docs, chars_in, chars_out = query("""
        SELECT count(*), sum(keep::int), count(DISTINCT doc_id),
               sum(length(text)), sum(CASE WHEN keep THEN length(text_out) ELSE 0 END)
        FROM blocks
    """)[0]

    sections = [
        '# Аудит очистки корпуса',
        '',
        f'- документов: **{docs}**',
        f'- блоков на входе: **{total}**, на выходе: **{kept}** '
        f'({_share(kept, total)})',
        f'- символов на входе: **{chars_in:,}**, на выходе: **{chars_out:,}** '
        f'({_share(chars_out, chars_in)})'.replace(',', ' '),
        '',
        'Отброшенное не удалено из таблицы: у каждого блока стоит `keep=false` и '
        'причина в `drop_reason`, поэтому порог можно менять без повторного прогона.',
        '',
        '## Пороги этого прогона',
        '',
        _table(['порог', 'значение'],
               sorted(extras['thresholds'].items())),
        '',
        '## Отброшено по причинам',
        '',
    ]

    reasons = query("""
        SELECT drop_reason, count(*), sum(length(text))
        FROM blocks WHERE NOT keep
        GROUP BY drop_reason ORDER BY 2 DESC
    """)
    sections.append(_table(
        ['причина', 'блоков', 'доля блоков', 'символов'],
        [(reason, number, _share(number, total), f'{volume:,}'.replace(',', ' '))
         for reason, number, volume in reasons]))
    sections.append('')
    sections.append('`merged` — не потеря: текст этих блоков перенесён в соседний '
                    'блок, ссылка в `merged_into`.')
    sections.append('')

    sections += ['## Оценка шума по полосам', '',
                 'Столбец «отброшено» показывает, что случится при текущем пороге '
                 f'{cfg.garbage_threshold}. Примеры взяты только из блоков, '
                 'собранных распознаванием: текстовый слой под подозрение не '
                 'попадает.', '']

    bands = []
    for step in range(10):
        low, high = step / 10, (step + 1) / 10 + (0.001 if step == 9 else 0)
        number, dropped = query("""
            SELECT count(*), sum((NOT keep AND drop_reason = 'garbage')::int)
            FROM blocks
            WHERE garbage_score >= ? AND garbage_score < ? AND NOT reliable
        """, (low, high))[0]
        bands.append((f'{low:.1f}–{high:.1f}', number, dropped or 0))
    sections.append(_table(['полоса', 'блоков (распознанных)', 'отброшено'], bands))
    sections.append('')

    for step in range(4, 10):
        low, high = step / 10, (step + 1) / 10 + (0.001 if step == 9 else 0)
        samples = query("""
            SELECT round(garbage_score, 2), garbage_reason, type, text_source,
                   substr(text_out, 1, ?)
            FROM blocks
            WHERE garbage_score >= ? AND garbage_score < ? AND NOT reliable
            ORDER BY hash(block_id) LIMIT 6
        """, (cfg.audit_sample_chars, low, high))
        if not samples:
            continue
        sections.append(f'### Полоса {low:.1f}–{high:.1f}')
        sections.append('')
        sections.append(_table(
            ['оценка', 'главный признак', 'тип', 'источник', 'текст'], samples))
        sections.append('')

    sections += ['## Что осталось, по типам блоков', '']
    types = query("""
        SELECT type, count(*), sum(keep::int),
               sum(CASE WHEN keep THEN length(text_out) ELSE 0 END)
        FROM blocks GROUP BY type ORDER BY 2 DESC
    """)
    sections.append(_table(
        ['тип', 'было', 'осталось', 'доля', 'символов на выходе'],
        [(name, was, left, _share(left, was), f'{volume:,}'.replace(',', ' '))
         for name, was, left, volume in types]))
    sections.append('')

    sections += ['## Что осталось, по источнику текста', '']
    sources = query("""
        SELECT text_source, count(*), sum(keep::int)
        FROM blocks GROUP BY text_source ORDER BY 2 DESC
    """)
    sections.append(_table(
        ['источник', 'было', 'осталось', 'доля'],
        [(name, was, left, _share(left, was)) for name, was, left in sources]))
    sections.append('')

    sections += ['## Склейка', '']
    rules = {
        'hyphen': 'слово, разорванное переносом',
        'same_line': 'строка, разрезанная по горизонтали',
        'paragraph': 'продолжение абзаца строкой ниже',
        'cross_page': 'абзац через границу страницы',
    }
    sections.append(_table(
        ['правило', 'что склеено', 'случаев'],
        [(name, rules.get(name, ''), number)
         for name, number in sorted(extras['merge_rules'].items(),
                                    key=lambda item: -item[1])]))
    sections.append('')

    grown = query("""
        SELECT substr(text, 1, 60), merged_count, substr(text_out, 1, ?)
        FROM blocks WHERE keep AND merged_count > 0
        ORDER BY merged_count DESC LIMIT 10
    """, (cfg.audit_sample_chars,))
    if grown:
        sections.append('Самые собранные блоки — было, сколько прирастило, стало:')
        sections.append('')
        sections.append(_table(['исходный текст', 'приклеено', 'результат'], grown))
        sections.append('')

    sections += ['## Примеры отброшенного', '']
    for reason, _, _ in reasons:
        if reason in ('merged', 'empty'):
            continue
        samples = query("""
            SELECT type, text_source, round(garbage_score, 2), substr(text_out, 1, ?)
            FROM blocks WHERE drop_reason = ?
            ORDER BY hash(block_id) LIMIT ?
        """, (cfg.audit_sample_chars, reason, cfg.audit_samples))
        sections.append(f'### {reason}')
        sections.append('')
        sections.append(_table(['тип', 'источник', 'оценка шума', 'текст'], samples))
        sections.append('')

    sections += ['## Колонтитулы', '',
                 f'Всего найдено разных строк обрамления: '
                 f'{extras["boilerplate_total"]}.', '']
    sections.append(_table(
        ['документ', 'строка', 'страниц', 'из страниц', 'блоков'],
        [(item['doc_id'].split('/')[-1], item['key'], item['pages'],
          item['doc_pages'], item['blocks'])
         for item in extras['boilerplate_top'][:30]]))
    sections.append('')

    copies = extras.get('duplicate_documents') or []
    sections += ['## Отброшенные копии документов', '',
                 f'Похожесть выше {cfg.doc_duplicate_drop} означает, что это '
                 'один и тот же файл, разложенный по разным тематическим '
                 'папкам. Из группы копий остаётся та, что лежит ближе к корню; '
                 'её блоки не тронуты, у остальных стоит '
                 '`drop_reason=duplicate_document`.', '']
    if copies:
        sections.append(_table(
            ['отброшено', 'осталось'],
            [(item['dropped'], item['kept']) for item in copies]))
    else:
        sections.append('Точных копий не найдено.')
    sections.append('')

    sections += ['## Похожие документы', '',
                 'Здесь ничего не удалено — это список на ручной разбор. '
                 'Сборники симпозиумов содержат те же статьи, что лежат '
                 'отдельными файлами, но совпадают с ними лишь частично, и '
                 'какая из версий полнее, порогом не решить.', '',
                 f'Пар выше порога {cfg.doc_similarity_report}: '
                 f'{extras["similar_total"]}.', '']
    partial = [item for item in extras['similar_documents']
               if item['similarity'] < cfg.doc_duplicate_drop]
    if partial:
        sections.append(_table(
            ['похожесть', 'документ', 'документ'],
            [(item['similarity'], item['left'], item['right'])
             for item in partial[:30]]))
    else:
        sections.append('Все найденные пары оказались точными копиями.')
    sections.append('')

    sections += ['## Документы с наибольшими потерями', '',
                 'Высокая доля отброшенного означает не строгий порог, а плохо '
                 'разобранный документ: скан без текстового слоя, картинки вместо '
                 'текста, сбой распознавания. Это список на пересмотр парсером.', '']
    worst = query("""
        SELECT doc_id, count(*), sum(keep::int),
               sum((drop_reason = 'garbage')::int),
               sum((drop_reason = 'boilerplate')::int)
        FROM blocks GROUP BY doc_id
        HAVING count(*) >= 50
        ORDER BY sum(keep::int)::double / count(*) LIMIT 20
    """)
    sections.append(_table(
        ['документ', 'блоков', 'осталось', 'доля', 'мусор', 'колонтитулы'],
        [(doc_id, number, left, _share(left, number), noise, boiler)
         for doc_id, number, left, noise, boiler in worst]))
    sections.append('')

    # Считается по очищенному тексту, а не по исходному: в исходном `(cid:NN)`
    # встречается втрое чаще, но подавляющее большинство — глиф маркера списка,
    # который нормализация снимает вместе с самим маркером. Счёт по сырому
    # тексту показывал бы проблему там, где её уже нет.
    cid_blocks, cid_total = query(r"""
        SELECT count(*), sum(len(regexp_extract_all(text_out, '\(cid:\d+\)')))
        FROM blocks WHERE keep AND regexp_matches(text_out, '\(cid:\d+\)')
    """)[0]
    sections += ['## Осталось починить в парсере', '',
                 f'- блоков, где `(cid:NN)` дожил до очищенного текста: '
                 f'**{cid_blocks}**, самих символов: **{cid_total or 0}**. '
                 'Шрифт вставлен без таблицы соответствия; читается из '
                 'встроенной в шрифт таблицы имён глифов.',
                 '- строки вне рамок макета в выгрузку не попадают вовсе, и '
                 'здесь их не видно: потерю содержимого этот отчёт не измеряет.',
                 '']

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text('\n'.join(sections), encoding='utf-8')

    summary = {
        'documents': docs,
        'blocks_in': total,
        'blocks_out': kept,
        'chars_in': chars_in,
        'chars_out': chars_out,
        'dropped': {reason: number for reason, number, _ in reasons},
        'by_type': [{'type': name, 'was': was, 'left': left}
                    for name, was, left, _ in types],
        'merge_rules': extras['merge_rules'],
        'similar_documents': extras['similar_total'],
        'boilerplate_strings': extras['boilerplate_total'],
        'cid_blocks_left': cid_blocks,
        'thresholds': extras['thresholds'],
    }
    out_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    connection.close()
    return summary
