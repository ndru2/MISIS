"""Атомарный чанкинг: режем дерево блоков, не окна с overlap.

Скользящее перекрытие только у прозы. У таблицы и формулы overlap структурный:
повторяем шапку и подпись, а не «отрезали 15% с конца». Иначе BM25 находит одну
и ту же шапку дважды, а генератор видит половинки строк.

Атом — кусок, который нельзя резать: формула вместе с вводом и легендой,
таблица целиком, абзац, пункт списка. Сборка атомов в чанк уже смотрит на бюджет.
"""

from __future__ import annotations

import re

from pdfscan.rag.chunk import (
    _Tokenizer, _assemble, _lead_in, _retext, _sentences, _table_chunks,
)
from pdfscan.rag.normalize import normalize_record
from pdfscan.rag.tables import parse_table_html, table_to_markdown

FURNITURE_TYPES = {'Header', 'Footer', 'PageBreak', 'PageNumber'}
HEADING_TYPES = {'Title'}
FORMULA_PREFIX = 'Formula'
TABLE_TYPES = {'Table'}
FIGURE_TYPES = {'Figure', 'FigureCaption', 'Caption'}

_TABLE_REF_RE = re.compile(
    r'\b(?:table|табл(?:ица)?)\s*[IVXЛC\d]+'
    r'|\bshown in table\b'
    r'|\bприведен[аоы]\s+в\s+табл',
    re.IGNORECASE,
)
_LEGEND_START_RE = re.compile(
    r'^(?:where|here|with|где|здесь|причём|причем)\b', re.IGNORECASE)


def is_formula(record: dict) -> bool:
    return str(record.get('type', '')).startswith(FORMULA_PREFIX)


def is_legend(record: dict) -> bool:
    if record.get('is_legend'):
        return True
    text = (record.get('text_clean') or record.get('text') or '').strip()
    if not text:
        return False
    if _LEGEND_START_RE.match(text):
        return True
    latex = record.get('formula_latex') or ''
    if r'\text{' in latex:
        return True
    # Короткая расшифровка символа: «ρ_Cu, ρ_slag = density of copper...»
    if (len(text) <= 240 and '=' in text and not is_formula(record)
            and not looks_like_long_prose(text)):
        return True
    return False


def looks_like_long_prose(text: str) -> bool:
    return len(text) > 240 or text.count('.') + text.count('!') + text.count('?') >= 2


def is_table_lead_in(record: dict) -> bool:
    text = record.get('text_clean') or record.get('text') or ''
    return bool(_TABLE_REF_RE.search(text)) and not is_formula(record)


def _skip(record: dict) -> bool:
    if record.get('type') in FURNITURE_TYPES:
        text = (record.get('text_clean') or record.get('text') or '').strip()
        return not re.search(r'[A-Za-zА-Яа-яЁё]{4,}', text)
    return False


def build_atoms(records: list[dict]) -> list[dict]:
    """Группирует блоки в атомы, которые чанкер уже не рвёт по токенам."""
    records = [normalize_record(dict(r)) for r in records]
    atoms, index, n = [], 0, len(records)

    while index < n:
        record = records[index]
        if _skip(record):
            index += 1
            continue

        if record.get('type') in HEADING_TYPES:
            atoms.append(_atom('heading', [record]))
            index += 1
            continue

        if record.get('type') in TABLE_TYPES:
            group = []
            if atoms and atoms[-1]['kind'] == 'prose' and is_table_lead_in(atoms[-1]['records'][-1]):
                group.extend(atoms.pop()['records'])
            group.append(record)
            atoms.append(_atom('table', group))
            index += 1
            continue

        # Подпись рисунка раньше формулы: короткий «Cu = 19 %» на оси графика
        # проходит is_legend, внутренний цикл сразу делает break по FIGURE_TYPES
        # и индекс не двигается.
        if record.get('type') in FIGURE_TYPES:
            atoms.append(_atom('figure', [record]))
            index += 1
            continue

        if is_formula(record) or is_legend(record):
            group = []
            if atoms and atoms[-1]['kind'] == 'prose':
                group.extend(atoms.pop()['records'])
            start = index
            while index < n:
                current = records[index]
                if _skip(current):
                    index += 1
                    continue
                if current.get('type') in HEADING_TYPES | TABLE_TYPES | FIGURE_TYPES:
                    break
                if is_formula(current) or is_legend(current):
                    group.append(current)
                    index += 1
                    continue
                break
            if index == start:
                index += 1
            atoms.append(_atom('formula', group or [record]))
            continue

        kind = 'list' if record.get('type') == 'ListItem' else 'prose'
        atoms.append(_atom(kind, [record]))
        index += 1

    return atoms


def _atom(kind: str, records: list[dict]) -> dict:
    return {
        'kind': kind,
        'records': records,
        'pages': sorted({r.get('page', 0) for r in records}),
        'split_allowed': kind in {'prose', 'list'},
    }


def _table_parent_text(records: list[dict], section: str, counter) -> str:
    """Полная таблица для small-to-big: нашли строку — отвечаем по всей таблице."""
    table = next(r for r in records if r.get('type') in TABLE_TYPES)
    parsed = parse_table_html(table.get('table_html'))
    markdown = table_to_markdown(parsed) if parsed else (table.get('text_clean') or table.get('text') or '')
    caption = table.get('caption') or ''
    lead = [r for r in records if r is not table]
    pieces = [(r.get('text_clean') or r.get('text') or '', False) for r in lead]
    if caption:
        pieces.append((caption, False))
    pieces.append((markdown, False))
    display = '\n'.join(part for part, _ in pieces if part)
    assembled = _assemble(
        lead + [_retext(table, display)] if lead else [_retext(table, display)],
        section, counter)
    return assembled['text']


def _section_push(path: list[str], heading: str, level: int) -> list[str]:
    heading = heading.strip()
    if not heading:
        return path
    level = max(1, int(level or 1))
    return path[: level - 1] + [heading]


def _prose_tail(record: dict, sentences: int, budget: int, counter) -> dict | None:
    if sentences <= 0 or budget <= 0:
        return None
    pieces = list(_sentences(record.get('text_clean') or '', budget, counter))
    if not pieces:
        return None
    kept = pieces[-sentences:]
    text = ' '.join(kept)
    if counter.count(text) > budget:
        lead = _lead_in(record, budget, counter)
        return lead[0] if lead else None
    return _retext(record, text)


def build_atomic_chunks(records, model_name=None, max_tokens=650,
                        min_tokens=180, target_tokens=400,
                        prose_overlap_sentences=2):
    """Собирает атомы в чанки с overlap только у прозы.

    ``min_tokens`` / ``target_tokens`` — ориентиры: плотная технология даст
    крупные куски, обзор — мелкие. Жёсткий потолок — ``max_tokens``, кроме
    формульного атома: его нельзя резать, даже если он длиннее бюджета.
    """
    counter = _Tokenizer(model_name)
    atoms = build_atoms(records)
    chunks = []
    section_path: list[str] = []
    current, current_tokens, current_kinds = [], 0, []
    pending_overlap = None

    def section_name():
        return ' / '.join(section_path)

    def flush(*, overlap=False):
        nonlocal current, current_tokens, current_kinds, pending_overlap
        if current:
            chunk = _assemble(current, section_name(), counter)
            chunk['section_path'] = list(section_path)
            chunk['atom_kinds'] = list(current_kinds)
            chunk['parent_text'] = (
                chunk['text'] if any(kind in {'table', 'formula'} for kind in current_kinds)
                else None
            )
            chunks.append(chunk)
            if overlap and current_kinds and current_kinds[-1] in {'prose', 'list'}:
                tail = _prose_tail(current[-1], prose_overlap_sentences,
                                   max(1, int(max_tokens * 0.15)), counter)
                pending_overlap = tail
            else:
                pending_overlap = None
        else:
            pending_overlap = None
        current, current_tokens, current_kinds = [], 0, []
        if pending_overlap is not None:
            current = [pending_overlap]
            current_tokens = counter.count(pending_overlap.get('text_clean') or '')
            current_kinds = ['prose']
            pending_overlap = None

    for atom in atoms:
        kind = atom['kind']
        group = atom['records']

        if kind == 'heading':
            flush(overlap=False)
            heading = group[0]
            section_path = _section_push(
                section_path,
                heading.get('text_clean') or heading.get('text') or '',
                heading.get('text_level') or 1,
            )
            continue

        if kind == 'table':
            if current:
                flush(overlap=False)
            table_record = next(r for r in group if r.get('type') in TABLE_TYPES)
            lead = [r for r in group if r is not table_record]
            parts = _table_chunks(table_record, max_tokens, counter)
            n_parts = len(parts)
            parent = _table_parent_text(group, section_name(), counter)
            for part_index, part in enumerate(parts):
                prefix = []
                caption = table_record.get('caption') or ''
                if caption and caption not in part:
                    prefix.append(caption)
                if n_parts > 1:
                    prefix.append(f'строки {part_index + 1} из {n_parts}')
                body = '\n'.join(prefix + [part]) if prefix else part
                piece_records = list(lead) + [_retext(table_record, body)]
                chunk = _assemble(piece_records, section_name(), counter)
                chunk['section_path'] = list(section_path)
                chunk['atom_kinds'] = ['table']
                chunk['parent_text'] = parent
                chunk['table_part'] = part_index
                chunk['table_parts'] = n_parts
                chunks.append(chunk)
            continue

        if kind == 'formula':
            tokens = sum(counter.count(r.get('text_clean') or r.get('text') or '')
                         for r in group)
            if current and current_tokens + tokens > max_tokens:
                flush(overlap=False)
            current.extend(group)
            current_tokens += tokens
            current_kinds.append('formula')
            # Формульный атом закрывает кусок: следующий абзац — уже другая мысль.
            flush(overlap=False)
            continue

        # Проза и список: пакуем до бюджета, overlap — хвост последнего абзаца.
        text = group[0].get('text_clean') or group[0].get('text') or ''
        tokens = counter.count(text)
        if current and current_tokens + tokens > max_tokens:
            flush(overlap=True)
        if tokens > max_tokens and atom['split_allowed']:
            from pdfscan.rag.chunk import split_long
            parts = split_long(text, max_tokens, counter)
            for part in parts[:-1]:
                if current:
                    flush(overlap=True)
                current = [_retext(group[0], part)]
                current_tokens = counter.count(part)
                current_kinds = [kind]
                flush(overlap=True)
            current = [_retext(group[0], parts[-1])]
            current_tokens = counter.count(parts[-1])
            current_kinds = [kind]
            continue
        current.extend(group)
        current_tokens += tokens
        current_kinds.append(kind)

    flush(overlap=False)

    for order, chunk in enumerate(chunks):
        chunk['chunk_id'] = f"{chunk['doc_id']}#c{order}"
        chunk['order'] = order
    return chunks
