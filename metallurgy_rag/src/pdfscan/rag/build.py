"""Построение и поиск плоского multi-vector RAG в Qdrant.

    python -m pdfscan.rag.build qdrant --source split --recreate
    python -m pdfscan.rag.build qsearch "Fe3O4 + C" --kind chemistry
"""

from __future__ import annotations

import argparse
from collections import Counter

from pdfscan.prepare import config as prepare_config
from pdfscan.prepare import store
from pdfscan.rag import qdrant_index
from pdfscan.rag import retrieval
from pdfscan.rag.split_source import split_documents
from pdfscan.rag.units import build_search_units

_MINERU_COLUMNS = [
    'doc_id', 'block_id', 'page', 'type', 'text_out', 'reliable', 'table_html', 'keep', 'order',
    'caption', 'formula_latex', 'text_level', 'is_legend',
    'year', 'authors', 'title', 'lang', 'doc_type', 'category', 'formula_class', 'number',
]
_META_KEYS = ('year', 'authors', 'title', 'lang', 'doc_type', 'category')


def mineru_documents(path=None):
    path = path or prepare_config.MINERU_CLEAN_BLOCKS
    if not path.exists():
        raise SystemExit(f'нет {path}; сначала: python -m pdfscan.prepare.cli mineru')
    for rows in store.iter_documents(path, _MINERU_COLUMNS):
        records = []
        for row in sorted(rows, key=lambda item: item['order']):
            if not row['keep'] or (not row.get('text_out') and not row.get('table_html')):
                continue
            record = {
                'doc_id': row['doc_id'], 'block_id': row['block_id'], 'page': row['page'],
                'type': row['type'], 'text': row['text_out'] or row.get('caption') or '',
                'reliable': row['reliable'], 'table_html': row['table_html'],
                'caption': row.get('caption') or '', 'formula_latex': row.get('formula_latex'),
                'text_level': row.get('text_level'), 'is_legend': bool(row.get('is_legend')),
                'formula_class': row.get('formula_class'), 'number': row.get('number') or '',
            }
            record.update({key: row.get(key) for key in _META_KEYS})
            records.append(record)
        if records:
            yield records


def iter_index_documents(source: str):
    if source == 'split':
        yield from split_documents()
    else:
        yield from mineru_documents()


def collect_search_units(max_tokens: int, source='split') -> list[dict]:
    """Материализует units для малых экспериментов; индексатор идёт потоково."""
    units, documents, blocks = [], 0, 0
    for records in iter_index_documents(source):
        documents += 1
        found = build_search_units(records, model_name=None, max_tokens=max_tokens)
        units.extend(found)
        blocks += len(records)
        if documents % 10 == 0:
            print(f'   {documents} документов, {blocks} блоков → {len(units)} search units', flush=True)
    if not units:
        raise SystemExit('не собрано ни одного search unit')
    print(f'📦 {documents} документов, {blocks} блоков → {len(units)} search units')
    print('   ' + ', '.join(f'{kind}: {count}' for kind, count in sorted(
        Counter(unit['unit_type'] for unit in units).items())))
    return units


def iter_document_units(max_tokens: int, source='split'):
    """Единицы одного документа: не удерживаем весь корпус в памяти."""
    for records in iter_index_documents(source):
        yield records, build_search_units(records, model_name=None, max_tokens=max_tokens)


def _model_names(args) -> dict[str, str]:
    return {
        'text': args.text_model, 'chemistry': args.chemistry_model,
        'math': args.math_model, 'table': args.table_model, 'unit': args.unit_model,
    }


def command_qdrant(args):
    models = _model_names(args)
    if not args.dry_run:
        if args.recreate and qdrant_index.VOCAB_PATH.exists():
            qdrant_index.VOCAB_PATH.unlink()
        dimensions = qdrant_index.initialize_collection(
            url=args.url, collection=args.collection, model_names=models, recreate=args.recreate)
        print(f'🔢 named vectors: {dimensions}')
        if args.recreate:
            removed = qdrant_index.drop_legacy_collections(url=args.url)
            if removed:
                print('🗑️ Удалены legacy-коллекции: ' + ', '.join(removed))

    totals, kinds, documents, blocks = 0, Counter(), 0, 0
    vocab = qdrant_index.SparseVocab.load()
    for records, units in iter_document_units(args.max_tokens, source=args.source):
        documents += 1
        blocks += len(records)
        kinds.update(unit['unit_type'] for unit in units)
        totals += len(units)
        if not args.dry_run:
            # Крупная таблица может дать десятки тысяч ячеек. Небольшие батчи
            # не дают векторизатору и клиенту Qdrant удерживать весь корпус.
            for start in range(0, len(units), args.batch_size):
                qdrant_index.upsert_units(
                    units[start:start + args.batch_size], url=args.url,
                    collection=args.collection, model_names=models, vocab=vocab,
                    progress=False, save_vocab=False)
        if documents % 10 == 0:
            print(f'   {documents} документов, {blocks} блоков → {totals} search units', flush=True)
    if not totals:
        raise SystemExit('не собрано ни одного search unit')
    print(f'📦 {documents} документов, {blocks} блоков → {totals} search units')
    print('   ' + ', '.join(f'{kind}: {count}' for kind, count in sorted(kinds.items())))
    if args.dry_run:
        return
    vocab.save()
    print(f'💾 {args.collection}: {totals} points')


def command_qsearch(args):
    filters = {}
    if args.formula:
        filters['has_formula'] = True
    if args.table:
        filters['has_table'] = True
    if args.doc:
        filters['doc_id'] = args.doc
    if args.element:
        filters['element'] = args.element
    if args.year:
        filters['year'] = args.year
    if args.lang:
        filters['lang'] = args.lang
    if args.unit_type:
        filters['unit_type'] = args.unit_type
    kinds = [args.kind] if args.kind else None
    hits = qdrant_index.search(
        args.query, k=args.k, url=args.url, collection=args.collection,
        model_names=_model_names(args), filters=filters or None, vector_kinds=kinds)
    if not hits:
        print('Ничего не найдено.')
        return
    for position, hit in enumerate(hits, 1):
        pages = ', '.join(str(page) for page in hit.get('pages') or [])
        print(f"\n{position}. [{hit.get('unit_type')}] {hit.get('doc_id')} с. {pages}; {hit['score']}")
        print(f"   id: {hit.get('unit_id')}; parent: {hit.get('parent_chunk_id') or '—'}")
        print('   ' + (hit.get('text') or '')[:500].replace('\n', '\n   '))


def command_retrieve(args):
    profile, evidence = retrieval.retrieve(
        args.query, k=args.k, url=args.url, collection=args.collection,
        model_names=_model_names(args))
    print('каналы: ' + ', '.join(profile.channels))
    if profile.filters:
        print('фильтры: ' + str(profile.filters))
    if not evidence:
        print('Ничего не найдено.')
        return
    for position, item in enumerate(evidence, 1):
        pages = ', '.join(str(page) for page in item.pages)
        print(f'\n{position}. [{item.unit_type}] {item.document_id} с. {pages}; {item.score:.5f}')
        print(f'   evidence: {item.text[:350]}')
        if item.context and item.context != item.text:
            print(f'   context: {item.context[:500]}')


def _add_models(parser):
    for kind in ('text', 'chemistry', 'math', 'table', 'unit'):
        parser.add_argument(f'--{kind}-model', default=qdrant_index.DEFAULT_MODEL,
                            help=f'Embedding model for dense_{kind}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Плоский multi-vector RAG в Qdrant')
    sub = parser.add_subparsers(dest='command', required=True)
    index = sub.add_parser('qdrant', help='собрать search units и загрузить в Qdrant')
    index.add_argument('--url', default=qdrant_index.DEFAULT_URL)
    index.add_argument('--collection', default=qdrant_index.DEFAULT_COLLECTION)
    index.add_argument('--source', choices=('split', 'mineru'), default='split')
    index.add_argument('--max-tokens', type=int, default=650)
    index.add_argument('--batch-size', type=int, default=512)
    index.add_argument('--recreate', action='store_true')
    index.add_argument('--dry-run', action='store_true')
    _add_models(index)
    index.set_defaults(func=command_qdrant)

    search = sub.add_parser('qsearch', help='гибридный поиск по search units')
    search.add_argument('query')
    search.add_argument('--url', default=qdrant_index.DEFAULT_URL)
    search.add_argument('--collection', default=qdrant_index.DEFAULT_COLLECTION)
    search.add_argument('-k', type=int, default=8)
    search.add_argument('--kind', choices=tuple(qdrant_index.VECTOR_NAMES))
    search.add_argument('--unit-type', choices=('chunk', 'formula', 'table_row', 'table_cell', 'quantity'))
    search.add_argument('--formula', action='store_true')
    search.add_argument('--table', action='store_true')
    search.add_argument('--doc')
    search.add_argument('--element')
    search.add_argument('--year', type=int)
    search.add_argument('--lang')
    _add_models(search)
    search.set_defaults(func=command_qsearch)

    retrieve = sub.add_parser('retrieve', help='router + parent-child Evidence для RAG')
    retrieve.add_argument('query')
    retrieve.add_argument('--url', default=qdrant_index.DEFAULT_URL)
    retrieve.add_argument('--collection', default=qdrant_index.DEFAULT_COLLECTION)
    retrieve.add_argument('-k', type=int, default=8)
    _add_models(retrieve)
    retrieve.set_defaults(func=command_retrieve)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
