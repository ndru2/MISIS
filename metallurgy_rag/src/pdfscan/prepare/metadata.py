"""Метаданные документа: первая страница, путь папки, колонтитул.

У металлургии год обязателен: технология 1975 и 2020 — разные ответы. MinerU
сам год не пишет, ``doc_id`` — обрезанный путь. Собираем то, что можно снять
детерминированно, без LLM: категорию сырья из папки, заголовок с первой
страницы, авторов из строки под ним, год из имени, колонтитула и заголовка.
"""

from __future__ import annotations

import re
from collections import Counter

from pdfscan.prepare.textstats import script as text_script

_YEAR_RE = re.compile(r'\b((?:19|20)\d{2})\b')
_ITS_YEAR_RE = re.compile(r'ИТС\s+\d+[-\u2013](\d{4})', re.IGNORECASE)
_AUTHOR_LINE_RE = re.compile(
    r'(?:'
    r'(?:[A-ZА-ЯЁ]\.\s*){1,3}[A-ZА-ЯЁ][A-Za-zА-Яа-яё\-]+'
    r'|'
    r'[A-ZА-ЯЁ][A-Za-zА-Яа-яё\-]+(?:\s+[A-ZА-ЯЁ]\.){1,3}'
    r')'
)
_NOT_AUTHOR_RE = re.compile(
    r'university|institute|mines|plc|ltd|inc|комбинат|институт|университет'
    r'|журнал|издател|elsevier|springer',
    re.IGNORECASE,
)


def category_from_doc_id(doc_id: str) -> str:
    return (doc_id or '').split('__', 1)[0].strip()


def doc_type_from_doc_id(doc_id: str) -> str:
    name = doc_id or ''
    if re.search(r'\bИТС\b|\bГОСТ\b|\bISO\b', name, re.IGNORECASE):
        return 'standard'
    if 'Книги' in name or re.search(r'\bbook\b', name, re.IGNORECASE):
        return 'book'
    return 'article'


def _years_in(text: str) -> list[int]:
    found = [int(y) for y in _YEAR_RE.findall(text or '')]
    found += [int(y) for y in _ITS_YEAR_RE.findall(text or '')]
    return [y for y in found if 1900 <= y <= 2026]


def _pick_year(doc_id: str, rows: list[dict]) -> int | None:
    from_name = _years_in(doc_id)
    if from_name:
        return from_name[-1]

    votes = Counter()
    for row in rows:
        if row.get('page', 1) > 3:
            continue
        if row.get('type') not in {'Header', 'Footer', 'Title', 'NarrativeText'}:
            continue
        for year in _years_in(row.get('text') or ''):
            votes[year] += 1
    if not votes:
        return None
    # Чаще всего это год издания в колонтитуле, а не страница «185».
    year, count = votes.most_common(1)[0]
    return year if count >= 1 else None


def _first_title(rows: list[dict]) -> str:
    titles = [row for row in rows
              if row.get('type') == 'Title' and (row.get('text') or '').strip()]
    if not titles:
        return ''
    first_page = min(row.get('page', 1) for row in titles)
    on_page = [row for row in titles if row.get('page', 1) == first_page]
    on_page.sort(key=lambda row: (row.get('text_level') or 1, row.get('order', 0)))
    return (on_page[0].get('text') or '').strip()


def _authors(rows: list[dict], title: str) -> list[str]:
    """Строка сразу под заголовком: «H.M. Chikashi» или «Ванюков А.В.»."""
    title_row = next(
        (row for row in rows
         if row.get('type') == 'Title' and (row.get('text') or '').strip() == title),
        None)
    if title_row is None:
        return []

    following = [
        row for row in rows
        if row.get('order', 0) > title_row.get('order', 0)
        and row.get('page') == title_row.get('page')
        and row.get('type') in {'NarrativeText', 'ListItem'}
        and (row.get('text') or '').strip()
    ]
    if not following:
        return []

    text = following[0]['text'].strip()
    if len(text) > 180 or _NOT_AUTHOR_RE.search(text):
        # «H.M. Chikashi Konkola Copper Mines plc» — имя есть, организация тоже.
        text = text.split(',')[0]
    names = [m.group(0).strip() for m in _AUTHOR_LINE_RE.finditer(text)]
    # Уникальные, в порядке появления.
    seen, ordered = set(), []
    for name in names:
        key = re.sub(r'\s+', ' ', name)
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered[:8]


def _language(rows: list[dict]) -> str:
    sample = []
    for row in rows:
        if row.get('type') not in {'NarrativeText', 'Title'}:
            continue
        sample.append(row.get('text') or '')
        if sum(len(s) for s in sample) >= 1500:
            break
    blob = ' '.join(sample)
    kind = text_script(blob)
    if kind == 'cyr':
        return 'ru'
    if kind == 'lat':
        return 'en'
    return 'und'


def extract_document(rows: list[dict]) -> dict:
    """Сводка по одному документу. Блоки не меняет."""
    if not rows:
        return {}
    doc_id = rows[0].get('doc_id') or ''
    title = _first_title(rows)
    authors = _authors(rows, title)
    return {
        'doc_id': doc_id,
        'category': rows[0].get('category') or category_from_doc_id(doc_id),
        'doc_type': doc_type_from_doc_id(doc_id),
        'title': title,
        'authors': authors,
        'year': _pick_year(doc_id, rows),
        'lang': _language(rows),
        'n_blocks': len(rows),
        'n_pages': len({row.get('page') for row in rows}),
    }


def stamp(rows: list[dict], meta: dict) -> None:
    """Пишет поля документа на каждый блок — так они доезжают до чанка и Qdrant."""
    authors = '; '.join(meta.get('authors') or [])
    for row in rows:
        row['year'] = meta.get('year')
        row['authors'] = authors or None
        row['title'] = meta.get('title') or None
        row['lang'] = meta.get('lang')
        row['doc_type'] = meta.get('doc_type')
        if not row.get('category'):
            row['category'] = meta.get('category')
