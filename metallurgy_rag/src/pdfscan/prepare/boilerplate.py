"""Колонтитулы и прочий повторяющийся текст обрамления.

Признак колонтитула — не место на странице и не класс блока, а повторяемость:
«Proceedings of Copper 2010» стоит на каждой странице сборника, и в ответе на
запрос эта строка не значит ничего. Считается число разных страниц документа, на
которых встретилась одна и та же строка, а цифры в ней предварительно сводятся к
одному символу, иначе меняющийся номер страницы делал бы каждый колонтитул
уникальным.

Порог задан и в страницах, и в их доле: три страницы отсекают случайное
совпадение, доля — документы, где страниц всего десяток. Повторяемость считается
внутри документа, а не по корпусу: «References» встречается в корпусе шестьсот
раз, но это заголовок раздела в разных статьях, а не обрамление.
"""

from collections import defaultdict

from . import config, textstats


def _is_candidate(row: dict, cfg) -> bool:
    """Колонтитулом может быть только короткая строка или явное обрамление."""
    if row.get('type') in config.FURNITURE_TYPES:
        return True
    return len(row.get('text_out') or '') <= cfg.boilerplate_max_chars


def detect(rows, cfg=config.DEFAULT) -> tuple:
    """Ищет повторяющиеся строки в блоках одного документа.

    Возвращает отображение ``block_id -> ключ`` и сводку для отчёта: какие именно
    строки признаны обрамлением и на скольких страницах они стоят.
    """
    if not rows:
        return {}, []

    doc_id = rows[0]['doc_id']
    n_pages = len({row['page'] for row in rows}) or 1

    key_pages = defaultdict(set)
    for row in rows:
        if not _is_candidate(row, cfg):
            continue
        key = textstats.norm_key(row.get('text_out') or '')
        if key:
            key_pages[key].add(row['page'])

    repeated = {
        key: pages for key, pages in key_pages.items()
        if len(pages) >= cfg.boilerplate_min_pages
        and len(pages) >= cfg.boilerplate_min_share * n_pages
    }
    if not repeated:
        return {}, []

    flags, hits = {}, defaultdict(int)
    for row in rows:
        if not _is_candidate(row, cfg):
            continue
        key = textstats.norm_key(row.get('text_out') or '')
        if key in repeated:
            flags[row['block_id']] = key
            hits[key] += 1

    summary = [{
        'doc_id': doc_id,
        'key': key,
        'pages': len(pages),
        'doc_pages': n_pages,
        'blocks': hits[key],
    } for key, pages in sorted(repeated.items(), key=lambda item: -len(item[1]))]

    return flags, summary
