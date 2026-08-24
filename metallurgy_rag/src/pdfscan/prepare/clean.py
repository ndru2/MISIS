"""Стадия очистки: от сырых блоков к тексту, пригодному для чанкинга.

Порядок шагов не произволен. Мусор убирается первым, чтобы он не мешал считать
повторяемость строк. Колонтитулы — вторыми, иначе они стоят между последним
абзацем страницы и первым абзацем следующей и не дают их сшить. Склейка идёт
третьей, и только после неё имеет смысл искать точные повторы: до склейки
одинаковыми выглядят обрывки, а не блоки.

Ни один шаг ничего не удаляет физически. Блок помечается ``keep=False`` с
причиной, а текст остаётся в таблице рядом с исходным. Порог отбора после этого
меняется без повторного прогона, а по отчёту видно не только сколько выброшено,
но и что именно.

Проходов по корпусу два. Первый учит символьную модель — ей нужна вся связная
проза, собранная из текстового слоя. Второй чистит документы по одному и сразу
пишет результат. Держать корпус в памяти не нужно ни на одном из них.
"""

import json
from collections import Counter

import pyarrow as pa

from pdfscan.rag.normalize import clean_text

from . import boilerplate, charlm, config, dedup, garbage, merge, store, textstats

CLEAN_FIELDS = store.RAW_FIELDS + [
    ('text_out', pa.string()),
    ('script', pa.string()),
    ('n_chars', pa.int32()),
    ('n_cid', pa.int32()),
    ('doubling_ratio', pa.float32()),
    ('vowel_ratio', pa.float32()),
    ('consonant_run_max', pa.int32()),
    ('short_token_ratio', pa.float32()),
    ('nonalpha_ratio', pa.float32()),
    ('repeat_run_max', pa.int32()),
    ('deviation', pa.float32()),
    ('garbage_score', pa.float32()),
    ('garbage_reason', pa.string()),
    ('boilerplate_key', pa.string()),
    ('merged_into', pa.string()),
    ('merged_count', pa.int32()),
    ('merged_ids', pa.string()),
    ('page_min', pa.int32()),
    ('page_max', pa.int32()),
    ('keep', pa.bool_()),
    ('drop_reason', pa.string()),
]

CLEAN_SCHEMA = pa.schema(CLEAN_FIELDS)

# Колонки, нужные для обучения символьной модели. Читать вместе с ними рамки и
# разметку незачем: Parquet отдаёт столбцы по отдельности.
_LM_COLUMNS = ['text', 'type', 'reliable']


def _prepare(rows) -> int:
    """Заводит рабочие поля и отбрасывает блоки, от которых после чистки ничего не осталось."""
    empty = 0
    for row in rows:
        raw = row.get('text') or ''
        row['n_cid'] = raw.count('(cid:')
        row['text_out'] = clean_text(raw)
        row['keep'] = bool(row['text_out'])
        row['drop_reason'] = '' if row['keep'] else 'empty'
        row['merged_into'] = None
        row['merged_count'] = 0
        row['merged_ids'] = []
        row['boilerplate_key'] = None
        row['page_min'] = row['page']
        row['page_max'] = row['page']
        row['deviation'] = None
        row['garbage_score'] = 0.0
        row['garbage_reason'] = ''
        empty += 0 if row['keep'] else 1
    return empty


def _score_garbage(rows, model, cfg) -> int:
    """Считает признаки и оценку шума для блоков документа."""
    dropped = 0
    for row in rows:
        row.update(textstats.signals(row['text_out']))
        if not row['keep']:
            continue

        if row['script'] != 'none':
            row['deviation'] = model.deviation(row['text_out'], row['script'])
        row['garbage_score'], row['garbage_reason'] = garbage.score(row, cfg)

        if garbage.is_garbage(row, cfg):
            row['keep'] = False
            row['drop_reason'] = 'garbage'
            dropped += 1
    return dropped


def _drop_boilerplate(rows, cfg) -> tuple:
    """Помечает повторяющееся обрамление страниц."""
    alive = [row for row in rows if row['keep']]
    flags, summary = boilerplate.detect(alive, cfg)
    for row in alive:
        key = flags.get(row['block_id'])
        if key is not None:
            row['boilerplate_key'] = key
            row['keep'] = False
            row['drop_reason'] = 'boilerplate'
    return len(flags), summary


def _train_model(raw_path, progress=True):
    """Учит символьную модель или берёт готовую с диска."""
    if config.CHARLM_MODEL.exists():
        if progress:
            print('символьная модель загружена')
        return charlm.CharLM.load()

    if progress:
        print('учу символьную модель на текстовом слое корпуса')

    def samples():
        for rows in store.iter_batches(raw_path, columns=_LM_COLUMNS):
            yield from charlm.training_samples(rows)

    model = charlm.CharLM.train(samples(), progress=progress)
    model.save()
    return model


def clean_document(rows, model, cfg=config.DEFAULT) -> dict:
    """Прогоняет все шаги очистки по блокам одного документа.

    Меняет словари на месте и возвращает счётчики отброшенного.
    """
    rows.sort(key=lambda row: row['order'])

    counts = Counter()
    counts['empty'] = _prepare(rows)
    counts['garbage'] = _score_garbage(rows, model, cfg)
    dropped, summary = _drop_boilerplate(rows, cfg)
    counts['boilerplate'] = dropped

    alive = [row for row in rows if row['keep']]
    rules = merge.merge_document(alive, cfg)
    counts['merged'] = sum(rules.values())
    counts['orphan'] = merge.drop_orphans(
        [row for row in rows if row['keep']], cfg)
    counts['duplicate'] = dedup.exact_duplicates(rows, cfg)

    return {'counts': counts, 'rules': Counter(rules), 'boilerplate': summary}


def _drop_copies(out_path, copies: dict, progress=True) -> int:
    """Снимает отбор у документов, оказавшихся копиями других.

    Отдельным проходом по уже записанной таблице, потому что решение принимается
    только когда посчитаны подписи всех документов, а к этому времени первые из
    них давно на диске. Проход идёт по документу за раз и пишет во временный
    файл, который затем встаёт на место исходного одним переименованием: обрыв
    на середине оставит прежнюю таблицу, а не половину новой.
    """
    temporary = out_path.with_suffix(out_path.suffix + '.tmp')
    removed = 0

    with store.DocumentWriter(temporary, CLEAN_SCHEMA) as writer:
        for rows in store.iter_documents(out_path):
            winner = copies.get(rows[0]['doc_id'])
            if winner is not None:
                for row in rows:
                    if row['keep']:
                        row['keep'] = False
                        row['drop_reason'] = 'duplicate_document'
                        removed += 1
            writer.write(rows)

    temporary.replace(out_path)
    if progress:
        print(f'копий документов отброшено: {len(copies)}, '
              f'в них блоков: {removed}')
    return removed


def run(raw_path=None, out_path=None, cfg=config.DEFAULT, progress=True) -> dict:
    """Выполняет стадию очистки целиком и возвращает сводку."""
    raw_path = raw_path or config.RAW_BLOCKS
    out_path = out_path or config.CLEAN_BLOCKS

    total_documents = store.count_documents(raw_path)
    if progress:
        print(f'{total_documents} документов в {raw_path.name}')

    model = _train_model(raw_path, progress)

    coefficients = dedup.make_coefficients(cfg)
    dropped = Counter()
    rules = Counter()
    boilerplate_summary = []
    doc_ids, signatures, sizes = [], [], {}
    kept_per_doc = {}
    blocks_in = blocks_out = chars_in = chars_out = 0

    with store.DocumentWriter(out_path, CLEAN_SCHEMA) as writer:
        for position, rows in enumerate(store.iter_documents(raw_path), 1):
            report = clean_document(rows, model, cfg)
            dropped.update(report['counts'])
            rules.update(report['rules'])
            boilerplate_summary.extend(report['boilerplate'])

            kept = [row for row in rows if row['keep']]
            blocks_in += len(rows)
            blocks_out += len(kept)
            chars_in += sum(len(row.get('text') or '') for row in rows)
            chars_out += sum(len(row['text_out']) for row in kept)

            doc_id = rows[0]['doc_id']
            signature, size = dedup.document_signature(
                [row['text_out'] for row in kept], coefficients, cfg)
            doc_ids.append(doc_id)
            signatures.append(signature)
            sizes[doc_id] = size
            kept_per_doc[doc_id] = len(kept)

            for row in rows:
                row['merged_ids'] = ' '.join(row['merged_ids']) or None
            writer.write(rows)

            if progress and position % 25 == 0:
                print(f'  очищено {position}/{total_documents} документов, '
                      f'осталось блоков {blocks_out} из {blocks_in}')

    if progress:
        print('сравниваю документы между собой')
    similar = dedup.similar_pairs(doc_ids, signatures, sizes, cfg)
    copies = dedup.duplicate_documents(similar, kept_per_doc, cfg)
    if copies:
        removed = _drop_copies(out_path, copies, progress)
        dropped['duplicate_document'] = removed
        blocks_out -= removed

    boilerplate_summary.sort(key=lambda item: -item['pages'])
    extras = {
        'thresholds': {
            'garbage_threshold': cfg.garbage_threshold,
            'garbage_min_chars': cfg.garbage_min_chars,
            'boilerplate_min_pages': cfg.boilerplate_min_pages,
            'boilerplate_min_share': cfg.boilerplate_min_share,
            'fragment_max_chars': cfg.fragment_max_chars,
            'orphan_max_chars': cfg.orphan_max_chars,
            'duplicate_min_chars': cfg.duplicate_min_chars,
        },
        'dropped': dict(dropped),
        'merge_rules': dict(rules),
        'boilerplate_top': boilerplate_summary[:60],
        'boilerplate_total': len(boilerplate_summary),
        'similar_documents': similar[:60],
        'similar_total': len(similar),
        'duplicate_documents': [{'dropped': loser, 'kept': winner}
                                for loser, winner in sorted(copies.items())],
        'documents': len(doc_ids),
        'blocks_in': blocks_in,
        'blocks_out': blocks_out,
        'chars_in': chars_in,
        'chars_out': chars_out,
    }

    config.EXTRAS.parent.mkdir(parents=True, exist_ok=True)
    config.EXTRAS.write_text(
        json.dumps(extras, ensure_ascii=False, indent=2), encoding='utf-8')

    if progress:
        print(f'осталось {blocks_out} блоков из {blocks_in} → {out_path}')
    return extras
