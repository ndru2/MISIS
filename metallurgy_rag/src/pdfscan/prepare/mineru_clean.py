"""Очистка блоков MinerU: пустые, обрамление, подписи без описания, склейка, дедуп.

Символьную модель шума сюда не тащим: она ловила кашу unstructured OCR, а у
MinerU ломается другое — колонтитулы, пустые абзацы и дубли книг в двух папках.
Формулы и таблицы по счётчику гласных всё равно выглядят мусором, и гонять их
через charlm значит выкидывать как раз то, ради чего корпус собирали.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow as pa

from pdfscan.prepare import (boilerplate, clean as clean_mod, config, dedup,
                             furniture, metadata, merge, store)
from pdfscan.rag.normalize import clean_text

MINERU_EXTRA_FIELDS = [
    ('parser', pa.string()),
    ('category', pa.string()),
    ('mineru_type', pa.string()),
    ('text_level', pa.int32()),
    ('caption', pa.string()),
    ('footnote', pa.string()),
    ('formula_latex', pa.string()),
    ('is_legend', pa.bool_()),
    ('list_type', pa.string()),
    ('image_path', pa.string()),
    ('year', pa.int32()),
    ('authors', pa.string()),
    ('title', pa.string()),
    ('lang', pa.string()),
    ('doc_type', pa.string()),
]

MINERU_CLEAN_FIELDS = list(clean_mod.CLEAN_FIELDS) + MINERU_EXTRA_FIELDS
MINERU_CLEAN_SCHEMA = pa.schema(MINERU_CLEAN_FIELDS)
_SCHEMA_NAMES = {name for name, _ in MINERU_CLEAN_FIELDS}


def _flatten_bbox(row: dict) -> None:
    box = row.get('bbox') or {}
    if isinstance(box, dict):
        row.setdefault('bbox_x0', box.get('x0'))
        row.setdefault('bbox_top', box.get('top'))
        row.setdefault('bbox_x1', box.get('x1'))
        row.setdefault('bbox_bottom', box.get('bottom'))
    row.setdefault('bbox_x0', None)
    row.setdefault('bbox_top', None)
    row.setdefault('bbox_x1', None)
    row.setdefault('bbox_bottom', None)
    row.setdefault('page_width', None)
    row.setdefault('page_height', None)


def _prepare(rows) -> int:
    empty = 0
    for row in rows:
        _flatten_bbox(row)
        raw = row.get('text') or ''
        row['n_cid'] = raw.count('(cid:')
        row['text_out'] = clean_text(raw)
        row['keep'] = bool(row['text_out']) or bool(row.get('table_html'))
        row['drop_reason'] = '' if row['keep'] else 'empty'
        row['merged_into'] = None
        row['merged_count'] = 0
        row['merged_ids'] = []
        row['boilerplate_key'] = None
        row['page_min'] = row.get('page') or 0
        row['page_max'] = row.get('page') or 0
        row['deviation'] = None
        row['garbage_score'] = 0.0
        row['garbage_reason'] = ''
        row['script'] = None
        row['n_chars'] = len(row['text_out'] or '')
        row['doubling_ratio'] = 0.0
        row['vowel_ratio'] = 0.0
        row['consonant_run_max'] = 0
        row['short_token_ratio'] = 0.0
        row['nonalpha_ratio'] = 0.0
        row['repeat_run_max'] = 0
        row.setdefault('parser', 'mineru')
        row.setdefault('languages', None)
        row.setdefault('math_font_ratio', None)
        row.setdefault('italic_ratio', None)
        row.setdefault('n_subscript', None)
        row.setdefault('n_superscript', None)
        row.setdefault('n_fonts', None)
        empty += 0 if row['keep'] else 1
    return empty


def clean_document(rows, cfg=config.DEFAULT) -> dict:
    """Те же шаги, что у unstructured, минус оценка OCR-шума."""
    rows.sort(key=lambda row: row.get('order', 0))

    counts = Counter()
    counts['empty'] = _prepare(rows)
    counts['furniture'] = furniture.drop_furniture(rows)
    counts['caption_only'] = furniture.drop_caption_only(rows)
    dropped, summary = clean_mod._drop_boilerplate(rows, cfg)
    counts['boilerplate'] = dropped

    alive = [row for row in rows if row['keep']]
    rules = merge.merge_document(alive, cfg)
    counts['merged'] = sum(rules.values())
    counts['orphan'] = merge.drop_orphans(
        [row for row in rows if row['keep']], cfg)
    counts['duplicate'] = dedup.exact_duplicates(rows, cfg)

    return {'counts': counts, 'rules': Counter(rules), 'boilerplate': summary}


def _parquet_row(row: dict) -> dict:
    packed = {}
    for name, _type in MINERU_CLEAN_FIELDS:
        value = row.get(name)
        if name == 'merged_ids' and isinstance(value, list):
            value = ' '.join(value) or None
        elif name == 'languages' and not isinstance(value, str):
            value = ','.join(value or ()) or None
        elif isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        packed[name] = value
    return packed


def iter_mineru_jsonl(root=None):
    """JSONL адаптера: ``data/mineru_blocks/*.jsonl``."""
    root = Path(root or config.MINERU_JSONL_DIR)
    files = sorted(root.glob('*.jsonl'))
    for path in files:
        yield path, store.iter_jsonl(path)


def run(jsonl_dir=None, out_path=None, cfg=config.DEFAULT, progress=True) -> dict:
    """Чистит корпус MinerU, пишет parquet, метаданные и сводку для аудита."""
    jsonl_dir = Path(jsonl_dir or config.MINERU_JSONL_DIR)
    out_path = Path(out_path or config.MINERU_CLEAN_BLOCKS)
    if not jsonl_dir.exists():
        raise SystemExit(
            f'нет {jsonl_dir}; сначала: python -m pdfscan.parse.mineru '
            'parsed_literature_1 --out data/mineru_blocks')

    files = sorted(jsonl_dir.glob('*.jsonl'))
    if not files:
        raise SystemExit(f'нет jsonl в {jsonl_dir}')
    if progress:
        print(f'{len(files)} документов в {jsonl_dir}')

    coefficients = dedup.make_coefficients(cfg)
    dropped = Counter()
    rules = Counter()
    boilerplate_summary = []
    documents_meta = []
    doc_ids, signatures, sizes = [], [], {}
    kept_per_doc = {}
    blocks_in = blocks_out = chars_in = chars_out = 0

    with store.DocumentWriter(out_path, MINERU_CLEAN_SCHEMA) as writer:
        for position, path in enumerate(files, 1):
            rows = store.iter_jsonl(path)
            if not rows:
                continue
            report = clean_document(rows, cfg)
            dropped.update(report['counts'])
            rules.update(report['rules'])
            boilerplate_summary.extend(report['boilerplate'])

            meta = metadata.extract_document(rows)
            metadata.stamp(rows, meta)
            documents_meta.append(meta)

            kept = [row for row in rows if row['keep']]
            blocks_in += len(rows)
            blocks_out += len(kept)
            chars_in += sum(len(row.get('text') or '') for row in rows)
            chars_out += sum(len(row.get('text_out') or '') for row in kept)

            doc_id = rows[0]['doc_id']
            signature, size = dedup.document_signature(
                [row['text_out'] for row in kept if row.get('text_out')],
                coefficients, cfg)
            doc_ids.append(doc_id)
            signatures.append(signature)
            sizes[doc_id] = size
            kept_per_doc[doc_id] = len(kept)

            writer.write([_parquet_row(row) for row in rows])

            if progress and position % 25 == 0:
                print(f'  очищено {position}/{len(files)} документов, '
                      f'осталось блоков {blocks_out} из {blocks_in}')

    if progress:
        print('сравниваю документы между собой')
    similar = dedup.similar_pairs(doc_ids, signatures, sizes, cfg)
    copies = dedup.duplicate_documents(similar, kept_per_doc, cfg)
    if copies:
        removed = clean_mod._drop_copies(
            out_path, copies, progress, schema=MINERU_CLEAN_SCHEMA)
        dropped['duplicate_document'] = removed
        blocks_out -= removed
        for meta in documents_meta:
            winner = copies.get(meta['doc_id'])
            meta['kept'] = winner is None
            if winner:
                meta['duplicate_of'] = winner
    else:
        for meta in documents_meta:
            meta['kept'] = True

    boilerplate_summary.sort(key=lambda item: -item['pages'])
    extras = {
        'corpus': 'mineru',
        'skip_garbage': True,
        'thresholds': {
            'boilerplate_min_pages': cfg.boilerplate_min_pages,
            'boilerplate_min_share': cfg.boilerplate_min_share,
            'fragment_max_chars': cfg.fragment_max_chars,
            'orphan_max_chars': cfg.orphan_max_chars,
            'duplicate_min_chars': cfg.duplicate_min_chars,
            'doc_duplicate_drop': cfg.doc_duplicate_drop,
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
        'metadata': {
            'with_year': sum(1 for m in documents_meta if m.get('year')),
            'with_authors': sum(1 for m in documents_meta if m.get('authors')),
            'with_title': sum(1 for m in documents_meta if m.get('title')),
            'by_type': dict(Counter(m.get('doc_type') for m in documents_meta)),
            'by_lang': dict(Counter(m.get('lang') for m in documents_meta)),
        },
    }

    config.EXTRAS.parent.mkdir(parents=True, exist_ok=True)
    config.EXTRAS.write_text(
        json.dumps(extras, ensure_ascii=False, indent=2), encoding='utf-8')
    config.DOCUMENTS_JSON.write_text(
        json.dumps(documents_meta, ensure_ascii=False, indent=2), encoding='utf-8')

    if progress:
        print(f'осталось {blocks_out} блоков из {blocks_in} → {out_path}')
        print(f'метаданные: {config.DOCUMENTS_JSON}')
    extras['documents_meta'] = documents_meta
    return extras
