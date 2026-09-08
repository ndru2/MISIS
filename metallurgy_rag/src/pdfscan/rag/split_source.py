"""Чтение ``corpus_split`` в ту же схему блоков, что у чанкера и графа.

Раскладка лежит по категориям, а атомарный чанк и тройки ждут один документ
в порядке чтения, с типами ``Title`` / ``Formula`` / ``Table``. Здесь категории
собираются обратно: заголовок остаётся заголовком, химия — ``Formula
(chemistry)``, чтобы фильтр ``has_chemistry`` и разбор веществ сработали так же,
как на ``mineru_clean``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pdfscan import paths
from pdfscan.prepare.split_corpus import CATEGORIES

# Подтип текстового блока → тип, который понимает ``build_atoms``.
_TEXT_TYPES = {
    'heading': 'Title',
    'text': 'NarrativeText',
    'ref_text': 'NarrativeText',
    'page_footnote': 'NarrativeText',
    'aside_text': 'NarrativeText',
    'caption': 'FigureCaption',
    # Расшифровка обозначений: чанкер приклеит её к соседней формуле.
    'nomenclature': 'NarrativeText',
}


def _block_id(record: dict) -> str:
    return f"{record.get('doc_id') or ''}#{record.get('order', 0)}"


def _common(record: dict) -> dict:
    return {
        'doc_id': record.get('doc_id'),
        'block_id': _block_id(record),
        'page': record.get('page') if record.get('page') is not None else 0,
        'reliable': True,
        'category': record.get('topic'),
        'section': record.get('section') or '',
        'parser': 'corpus_split',
    }


def as_index_record(record: dict, category: str) -> dict | None:
    """Одна запись раскладки → блок для ``build_atomic_chunks`` / троек."""
    base = _common(record)
    if category == 'text':
        subtype = record.get('subtype') or 'text'
        text = record.get('text') or ''
        if not text.strip():
            return None
        mapped = _TEXT_TYPES.get(subtype, 'NarrativeText')
        return {**base, **{
            'type': mapped,
            'text': text,
            'table_html': None,
            'caption': '',
            'formula_latex': None,
            'text_level': 1 if mapped == 'Title' else None,
            'is_legend': subtype == 'nomenclature',
        }}

    if category in ('formulas_chem', 'formulas_math'):
        latex = record.get('latex') or ''
        if not latex.strip():
            return None
        chemistry = record.get('formula_class') == 'chemistry'
        return {**base, **{
            'type': 'Formula (chemistry)' if chemistry else 'Formula',
            'text': latex,
            'table_html': None,
            'caption': '',
            'formula_latex': latex,
            'text_level': None,
            'is_legend': False,
            'formula_class': record.get('formula_class'),
            'number': record.get('number') or '',
        }}

    if category == 'tables':
        html = record.get('table_html') or ''
        caption = record.get('caption') or ''
        footnote = record.get('footnote') or ''
        if not html.strip() and not caption.strip():
            return None
        return {**base, **{
            'type': 'Table',
            'text': caption or footnote,
            'table_html': html or None,
            'caption': caption,
            'footnote': footnote,
            'formula_latex': None,
            'text_level': None,
            'is_legend': False,
        }}
    return None


def _read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def document_ids(split_dir: Path) -> list[str]:
    names = set()
    for category in CATEGORIES:
        folder = split_dir / category
        if not folder.exists():
            continue
        names.update(path.stem for path in folder.glob('*.jsonl'))
    return sorted(names)


def load_document(split_dir: Path, doc_id: str) -> list[dict]:
    """Блоки одного документа в порядке ``order`` из раскладки."""
    records = []
    for category in CATEGORIES:
        path = split_dir / category / f'{doc_id}.jsonl'
        for raw in _read_jsonl(path):
            mapped = as_index_record(raw, category)
            if mapped:
                mapped['_order'] = raw.get('order', 0)
                records.append(mapped)
    records.sort(key=lambda item: (item.get('_order') or 0, item.get('page') or 0))
    for record in records:
        record.pop('_order', None)
    return records


def split_documents(split_dir=None):
    """Документы раскладки по одному, готовые к чанкингу."""
    split_dir = Path(split_dir or paths.ROOT / 'corpus_split')
    if not split_dir.exists():
        raise SystemExit(f'нет {split_dir}; сначала: python -m pdfscan.prepare.split_corpus')
    ids = document_ids(split_dir)
    if not ids:
        raise SystemExit(f'в {split_dir} нет *.jsonl по категориям')
    for doc_id in ids:
        records = load_document(split_dir, doc_id)
        if records:
            yield records
