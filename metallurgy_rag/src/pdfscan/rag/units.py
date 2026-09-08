"""Плоские search units для multi-vector RAG.

Chunk остаётся родительским контекстом. Формулы, строки/ячейки таблиц и
величины — самостоятельные дочерние объекты с ``parent_chunk_id``. Это не
граф: связь нужна только для small-to-big retrieval и provenance.
"""

from __future__ import annotations

import re
from collections import defaultdict

from pdfscan.rag.atoms import build_atomic_chunks
from pdfscan.rag.normalize import extract_units, flatten_formula, normalize_record
from pdfscan.rag.tables import parse_table_html

_EQ_NUMBER_RE = re.compile(r"(?:\\tag\{([^}]+)\}|\[([\d.]+)\]|\(([\d.]+)\))")


def _meta(chunk: dict) -> dict:
    return {
        key: chunk.get(key)
        for key in ('doc_id', 'title', 'authors', 'year', 'lang', 'doc_type', 'category')
    }


def _parent_for(block_id: str, by_block: dict[str, list[dict]]) -> dict | None:
    choices = by_block.get(block_id) or []
    return choices[0] if choices else None


def _unit(*, unit_id: str, unit_type: str, parent: dict, text: str,
          search_text: str, vector_kind: str, **extra) -> dict:
    return {
        'unit_id': unit_id,
        'chunk_id': unit_id,
        'unit_type': unit_type,
        'vector_kind': vector_kind,
        'parent_chunk_id': parent.get('chunk_id'),
        'parent_text': parent.get('parent_text') or parent.get('text'),
        'text': text,
        'text_search': search_text,
        'section': parent.get('section') or '',
        'section_path': parent.get('section_path') or [],
        'pages': parent.get('pages') or [],
        'types': parent.get('types') or [],
        'has_formula': bool(parent.get('has_formula')),
        'has_table': bool(parent.get('has_table')),
        'has_chemistry': bool(parent.get('has_chemistry')),
        'elements': parent.get('elements') or [],
        'element_names': parent.get('element_names') or [],
        'units': parent.get('units') or [],
        'reliable': bool(parent.get('reliable', True)),
        **_meta(parent),
        **extra,
    }


def _chunk_units(chunks: list[dict]) -> list[dict]:
    produced = []
    for chunk in chunks:
        produced.append({
            'unit_id': chunk['chunk_id'],
            'chunk_id': chunk['chunk_id'],
            'unit_type': 'chunk',
            'vector_kind': 'text',
            'parent_chunk_id': None,
            'parent_text': chunk.get('parent_text'),
            'text': chunk['text'],
            'text_search': chunk['text_search'],
            'section': chunk.get('section') or '',
            'section_path': chunk.get('section_path') or [],
            'pages': chunk.get('pages') or [],
            'types': chunk.get('types') or [],
            'has_formula': bool(chunk.get('has_formula')),
            'has_table': bool(chunk.get('has_table')),
            'has_chemistry': bool(chunk.get('has_chemistry')),
            'elements': chunk.get('elements') or [],
            'element_names': chunk.get('element_names') or [],
            'units': chunk.get('units') or [],
            'reliable': bool(chunk.get('reliable', True)),
            **_meta(chunk),
        })
    return produced


def _formula_units(records: list[dict], by_block: dict[str, list[dict]]) -> list[dict]:
    produced = []
    for record in records:
        if not str(record.get('type') or '').startswith('Formula'):
            continue
        parent = _parent_for(record.get('block_id') or '', by_block)
        if parent is None:
            continue
        latex = record.get('formula_latex') or record.get('text') or ''
        flat = flatten_formula(latex)
        numbers = [''.join(group for group in match.groups() if group)
                   for match in _EQ_NUMBER_RE.finditer(latex)]
        number = numbers[0] if numbers else (record.get('number') or '')
        kind = 'chemistry' if record.get('formula_class') == 'chemistry' else 'math'
        context = parent.get('section') or ''
        search = ' '.join(filter(None, [
            context, f'equation {number}' if number else '', flat,
            ' '.join(record.get('element_names') or []),
        ]))
        produced.append(_unit(
            unit_id=f"{record['block_id']}#formula", unit_type='formula', parent=parent,
            text=latex, search_text=search, vector_kind=kind,
            formula_latex=latex, formula_flat=flat, equation_number=number,
            formula_class=record.get('formula_class') or 'math',
            has_formula=True, has_chemistry=kind == 'chemistry',
            elements=record.get('elements') or [],
            element_names=record.get('element_names') or [],
            units=record.get('units') or [],
        ))
    return produced


def _table_units(records: list[dict], by_block: dict[str, list[dict]]) -> list[dict]:
    produced = []
    for record in records:
        if record.get('type') != 'Table':
            continue
        parent = _parent_for(record.get('block_id') or '', by_block)
        table = parse_table_html(record.get('table_html') or '')
        if parent is None or not table:
            continue
        caption = (record.get('caption') or parent.get('section') or '').strip()
        header = [str(value).strip() for value in table.get('header') or []]
        table_id = f"{record['block_id']}#table"
        for row_index, row in enumerate(table.get('rows') or []):
            cells = [str(value).strip() for value in row]
            label = cells[0] if cells else ''
            facts = []
            for column_index, value in enumerate(cells):
                column = header[column_index] if column_index < len(header) else f'column {column_index + 1}'
                if value:
                    facts.append(f'{column}: {value}')
            search = ' | '.join(filter(None, [caption, label, *facts]))
            row_id = f'{table_id}#r{row_index}'
            produced.append(_unit(
                unit_id=row_id, unit_type='table_row', parent=parent,
                text=search, search_text=search, vector_kind='table', table_id=table_id,
                row_id=row_id, row_index=row_index, row_label=label, header=header,
                cells=cells, caption=caption, has_table=True,
            ))
            for column_index, value in enumerate(cells):
                if not value:
                    continue
                column = header[column_index] if column_index < len(header) else f'column {column_index + 1}'
                cell_search = ' | '.join(filter(None, [caption, label, column, value]))
                produced.append(_unit(
                    unit_id=f'{row_id}#c{column_index}', unit_type='table_cell', parent=parent,
                    text=cell_search, search_text=cell_search, vector_kind='table', table_id=table_id,
                    row_id=row_id, row_index=row_index, column_id=f'{table_id}#c{column_index}',
                    column_index=column_index, row_label=label, column_header=column,
                    cell_value=value, caption=caption, has_table=True,
                    units=extract_units(cell_search),
                ))
    return produced


def _quantity_units(chunks: list[dict]) -> list[dict]:
    produced = []
    for chunk in chunks:
        for index, measurement in enumerate(chunk.get('units') or []):
            raw = measurement.get('text') or ''
            if not raw:
                continue
            search = ' '.join(filter(None, [chunk.get('section') or '', raw, chunk.get('text_search') or '']))
            produced.append(_unit(
                unit_id=f"{chunk['chunk_id']}#quantity{index}", unit_type='quantity', parent=chunk,
                text=raw, search_text=search, vector_kind='unit', quantity=measurement,
                value=measurement.get('value'), unit_raw=measurement.get('unit'),
                unit_canonical=measurement.get('canonical'), si_value=measurement.get('si_value'),
                dimension=measurement.get('canonical'),
            ))
    return produced


def build_search_units(records: list[dict], *, model_name=None, max_tokens=650) -> list[dict]:
    """Создаёт parent chunks и их дочерние плоские search units."""
    normalized = [normalize_record(dict(record)) for record in records]
    chunks = build_atomic_chunks(normalized, model_name=model_name, max_tokens=max_tokens)
    # Чанкер намеренно не знает о каталожных полях. Все дочерние единицы
    # наследуют их от родителя, поэтому проставляем один раз здесь.
    document_meta = {
        key: normalized[0].get(key)
        for key in ('title', 'authors', 'year', 'lang', 'doc_type', 'category')
    }
    for chunk in chunks:
        for key, value in document_meta.items():
            if chunk.get(key) is None:
                chunk[key] = value
    by_block: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        for block_id in chunk.get('block_ids') or []:
            by_block[block_id].append(chunk)
    return (
        _chunk_units(chunks)
        + _formula_units(normalized, by_block)
        + _table_units(normalized, by_block)
        + _quantity_units(chunks)
    )
