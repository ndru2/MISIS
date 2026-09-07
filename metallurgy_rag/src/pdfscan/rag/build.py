"""Сборка поискового индекса из очищенных блоков и поиск по нему.

    python -m pdfscan.rag.build build                 # индекс из blocks_clean.parquet
    python -m pdfscan.rag.build build out/**/blocks.jsonl   # в обход очистки
    python -m pdfscan.rag.build search "запрос"       # найти куски
    python -m pdfscan.rag.build search "запрос" --formula --element Fe

По умолчанию берётся результат стадии очистки, а не исходная выгрузка. Разница
существенная: в сырых блоках остаются колонтитулы на каждой странице, обрывки
строк вместо абзацев и нечитаемое распознавание перевёрнутых таблиц — всё это
попало бы в индекс наравне с содержанием. Путь к JSONL всё же принимается: он
нужен, когда индекс собирают для проверки самого разбора.
"""

import argparse
import glob

from pdfscan.parse.export import load_blocks
from pdfscan.prepare import config as prepare_config
from pdfscan.prepare import store
from pdfscan.rag.atoms import build_atomic_chunks
from pdfscan.rag.chunk import build_chunks
from pdfscan.rag.index import EMBEDDING_MODEL, RagIndex
from pdfscan.rag import qdrant_index
from pdfscan.rag.split_source import split_documents
# Импорт по именам: ниже есть локальная переменная ``paths``.
from pdfscan.paths import RAG_INDEX_DIR

# Всё, что нужно чанкеру. Рамки и признаки шрифта он не смотрит, а Parquet
# отдаёт столбцы по отдельности, поэтому лишние читать незачем.
_CHUNK_COLUMNS = ['doc_id', 'block_id', 'page', 'type', 'text_out',
                  'reliable', 'table_html', 'keep', 'order']

_MINERU_COLUMNS = _CHUNK_COLUMNS + [
    'caption', 'formula_latex', 'text_level', 'is_legend',
    'year', 'authors', 'title', 'lang', 'doc_type', 'category',
]

_META_KEYS = ('year', 'authors', 'title', 'lang', 'doc_type', 'category')


def clean_documents(path=None):
    """Отдаёт очищенные блоки по документу: список записей для чанкинга.

    Отобранное берётся в поле ``text_out`` — это текст после нормализации и
    склейки, тогда как в ``text`` лежит исходный. Блоки с ``keep=False``
    пропускаются: их содержимое либо мусор, либо уже перенесено в соседний блок.
    """
    path = path or prepare_config.CLEAN_BLOCKS
    if not path.exists():
        raise SystemExit(f'нет {path}; сначала: python -m pdfscan.prepare.cli all')

    for rows in store.iter_documents(path, _CHUNK_COLUMNS):
        records = [{
            'doc_id': row['doc_id'],
            'block_id': row['block_id'],
            'page': row['page'],
            'type': row['type'],
            'text': row['text_out'],
            'reliable': row['reliable'],
            'table_html': row['table_html'],
        } for row in sorted(rows, key=lambda item: item['order'])
            if row['keep'] and row['text_out']]
        if records:
            yield records


def mineru_documents(path=None):
    """Очищенные блоки MinerU — с подписью таблицы, уровнем заголовка и годом."""
    path = path or prepare_config.MINERU_CLEAN_BLOCKS
    if not path.exists():
        raise SystemExit(
            f'нет {path}; сначала: python -m pdfscan.prepare.cli mineru')

    for rows in store.iter_documents(path, _MINERU_COLUMNS):
        records = []
        for row in sorted(rows, key=lambda item: item['order']):
            if not row['keep']:
                continue
            if not row.get('text_out') and not row.get('table_html'):
                continue
            record = {
                'doc_id': row['doc_id'],
                'block_id': row['block_id'],
                'page': row['page'],
                'type': row['type'],
                'text': row['text_out'] or row.get('caption') or '',
                'reliable': row['reliable'],
                'table_html': row['table_html'],
                'caption': row.get('caption') or '',
                'formula_latex': row.get('formula_latex'),
                'text_level': row.get('text_level'),
                'is_legend': bool(row.get('is_legend')),
            }
            for key in _META_KEYS:
                record[key] = row.get(key)
            records.append(record)
        if records:
            yield records


def _stamp_meta(chunk: dict, records: list[dict]) -> dict:
    source = records[0]
    for key in _META_KEYS:
        if not chunk.get(key):
            chunk[key] = source.get(key)
    return chunk


def iter_index_documents(source: str):
    """Откуда брать блоки: раскладка corpus_split или очищенный MinerU."""
    if source == 'split':
        yield from split_documents()
        return
    yield from mineru_documents()


def collect_mineru_chunks(max_tokens, source='split'):
    chunks, documents, blocks = [], 0, 0
    for records in iter_index_documents(source):
        documents += 1
        name = (records[0].get('doc_id') or '')[:72]
        print(f'   режу {documents}: {name} ({len(records)} блоков)', flush=True)
        produced = [
            _stamp_meta(chunk, records)
            for chunk in build_atomic_chunks(
                records, model_name=None, max_tokens=max_tokens)
        ]
        chunks.extend(produced)
        blocks += len(records)
        print(f'   {documents} документов, блоков {blocks} → '
              f'кусков {len(chunks)}', flush=True)
    if not chunks:
        raise SystemExit('не собрано ни одного куска')
    print(f'   {documents} документов, блоков {blocks} → кусков {len(chunks)}')
    return chunks


def collect_chunks(patterns, max_tokens):
    if patterns:
        sources = ((path, load_blocks(path)) for path in sorted(
            {p for pattern in patterns for p in glob.glob(pattern, recursive=True)}))
    else:
        sources = ((records[0]['doc_id'], records)
                   for records in clean_documents())

    chunks, documents, blocks = [], 0, 0
    for name, records in sources:
        produced = build_chunks(records, model_name=EMBEDDING_MODEL,
                                max_tokens=max_tokens)
        chunks.extend(produced)
        documents += 1
        blocks += len(records)
        if documents % 25 == 0:
            print(f'   {documents} документов, блоков {blocks} → '
                  f'кусков {len(chunks)} (последний: {str(name)[-60:]})')

    if not chunks:
        raise SystemExit('не собрано ни одного куска')
    print(f'   {documents} документов, блоков {blocks} → кусков {len(chunks)}')
    return chunks


def _command_graph(args):
    from pdfscan.rag.graph import command_build
    command_build(args)


def command_qdrant(args):
    chunks = collect_mineru_chunks(args.max_tokens, source=args.source)
    sizes = [c['n_tokens'] for c in chunks]
    print(f'📦 Всего кусков: {len(chunks)},'
          f' токенов мин/сред/макс {min(sizes)}/{sum(sizes) // len(sizes)}/{max(sizes)}')
    if args.dry_run:
        return
    if args.recreate and qdrant_index.VOCAB_PATH.exists():
        # Словарь BM25 живёт рядом с коллекцией: после полной перезаписи
        # старые id термов больше ничему не соответствуют.
        qdrant_index.VOCAB_PATH.unlink()
    qdrant_index.upsert_chunks(
        chunks, url=args.url, collection=args.collection,
        model_name=args.model, recreate=args.recreate)
    if args.graph:
        from pdfscan.rag.graph import collect_triples, upsert_triples
        triples = collect_triples(progress=True, source=args.source)
        kinds = {}
        for triple in triples:
            kinds[triple['kind']] = kinds.get(triple['kind'], 0) + 1
        print(f'📦 троек {len(triples)}: '
              + ', '.join(f'{k} {n}' for k, n in sorted(kinds.items())))
        upsert_triples(
            triples, url=args.url, collection=args.graph_collection,
            model_name=args.model, recreate=args.recreate)


def command_qdrant_search(args):
    filters = {}
    if args.formula:
        filters['has_formula'] = True
    if args.table:
        filters['has_table'] = True
    if args.doc:
        filters['doc_id'] = args.doc
    if args.element:
        filters['element'] = args.element[0]
    if args.year:
        filters['year'] = args.year
    if args.lang:
        filters['lang'] = args.lang
    results = qdrant_index.search(
        args.query, k=args.k, url=args.url, collection=args.collection,
        model_name=args.model, filters=filters or None)
    if not results:
        print('Ничего не найдено.')
        return
    for position, chunk in enumerate(results, 1):
        pages = ', '.join(str(p) for p in (chunk.get('pages') or []))
        print(f"\n{position}. [{chunk.get('doc_id')}, с. {pages}] "
              f"оценка {chunk.get('score')} год {chunk.get('year')}")
        if chunk.get('section'):
            print(f"   раздел: {chunk['section']}")
        body = (chunk.get('text') or '').strip().replace('\n', '\n   ')
        print(f'   {body[:400]}{"…" if len(body) > 400 else ""}')


def command_build(args):
    chunks = collect_chunks(args.blocks, args.max_tokens)
    sizes = [c['n_tokens'] for c in chunks]
    print(f'📦 Всего кусков: {len(chunks)},'
          f' токенов мин/сред/макс {min(sizes)}/{sum(sizes) // len(sizes)}/{max(sizes)}')

    index = RagIndex.build(chunks)
    index.save(args.index)


def command_search(args):
    index = RagIndex.load(args.index)

    filters = {}
    if args.formula:
        filters['has_formula'] = True
    if args.table:
        filters['has_table'] = True
    if args.reliable:
        filters['reliable'] = True
    if args.doc:
        filters['doc_id'] = args.doc
    if args.element:
        filters['elements'] = args.element

    results = index.search(args.query, k=args.k, filters=filters or None)
    if not results:
        print('Ничего не найдено.')
        return

    for position, chunk in enumerate(results, 1):
        pages = ', '.join(str(p) for p in chunk['pages'])
        print(f"\n{position}. [{chunk['doc_id']}, с. {pages}] оценка {chunk['score']}"
              f" (смысл {chunk['dense_score']}, слова {chunk['lexical_score']})")
        if chunk['section']:
            print(f"   раздел: {chunk['section']}")
        if chunk['elements']:
            print(f"   элементы: {', '.join(chunk['elements'])}")
        body = chunk['text'].strip().replace('\n', '\n   ')
        print(f'   {body[:400]}{"…" if len(body) > 400 else ""}')


def main():
    parser = argparse.ArgumentParser(description='Индекс и поиск по разобранным PDF')
    sub = parser.add_subparsers(dest='command', required=True)

    build = sub.add_parser('build', help='собрать индекс')
    build.add_argument('blocks', nargs='*',
                       help='пути к blocks.jsonl в обход стадии очистки')
    build.add_argument('--index', default=str(RAG_INDEX_DIR))
    build.add_argument('--max-tokens', type=int, default=400)
    build.set_defaults(func=command_build)

    search = sub.add_parser('search', help='найти куски')
    search.add_argument('query')
    search.add_argument('--index', default=str(RAG_INDEX_DIR))
    search.add_argument('-k', type=int, default=5)
    search.add_argument('--formula', action='store_true', help='только куски с формулами')
    search.add_argument('--table', action='store_true', help='только куски с таблицами')
    search.add_argument('--reliable', action='store_true', help='только надёжный текст')
    search.add_argument('--doc', help='ограничить документом')
    search.add_argument('--element', nargs='+', help='химические элементы')
    search.set_defaults(func=command_search)

    qdrant = sub.add_parser('qdrant', help='загрузить атомарные куски в Qdrant')
    qdrant.add_argument('--url', default=qdrant_index.DEFAULT_URL)
    qdrant.add_argument('--collection', default=qdrant_index.DEFAULT_COLLECTION)
    qdrant.add_argument('--model', default=qdrant_index.DEFAULT_MODEL)
    qdrant.add_argument('--max-tokens', type=int, default=650)
    qdrant.add_argument('--source', choices=('split', 'mineru'), default='split',
                        help='split — corpus_split (весь parsed_literature); '
                             'mineru — data/mineru_clean.parquet')
    qdrant.add_argument('--graph', action='store_true',
                        help='следом залить тройки таблиц/формул/величин')
    qdrant.add_argument('--graph-collection',
                        default=qdrant_index.DEFAULT_TRIPLES_COLLECTION)
    qdrant.add_argument('--recreate', action='store_true')
    qdrant.add_argument('--dry-run', action='store_true',
                        help='только нарезать куски, без эмбеддингов и Qdrant')
    qdrant.set_defaults(func=command_qdrant)

    qsearch = sub.add_parser('qsearch', help='гибридный поиск в Qdrant')
    qsearch.add_argument('query')
    qsearch.add_argument('--url', default=qdrant_index.DEFAULT_URL)
    qsearch.add_argument('--collection', default=qdrant_index.DEFAULT_COLLECTION)
    qsearch.add_argument('--model', default=qdrant_index.DEFAULT_MODEL)
    qsearch.add_argument('-k', type=int, default=8)
    qsearch.add_argument('--formula', action='store_true')
    qsearch.add_argument('--table', action='store_true')
    qsearch.add_argument('--doc')
    qsearch.add_argument('--element', nargs='+')
    qsearch.add_argument('--year', type=int)
    qsearch.add_argument('--lang')
    qsearch.set_defaults(func=command_qdrant_search)

    graph_p = sub.add_parser('graph', help='тройки Wikontic: таблицы, формулы, ru↔en')
    graph_p.add_argument('--url', default=qdrant_index.DEFAULT_URL)
    graph_p.add_argument('--collection', default=qdrant_index.DEFAULT_TRIPLES_COLLECTION)
    graph_p.add_argument('--model', default=qdrant_index.DEFAULT_MODEL)
    graph_p.add_argument('--source', choices=('split', 'mineru'), default='split')
    graph_p.add_argument('--recreate', action='store_true')
    graph_p.add_argument('--dry-run', action='store_true')
    graph_p.set_defaults(func=_command_graph)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
