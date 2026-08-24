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
from pdfscan.rag.chunk import build_chunks
from pdfscan.rag.index import EMBEDDING_MODEL, RagIndex
# Импорт по именам: ниже есть локальная переменная ``paths``.
from pdfscan.paths import RAG_INDEX_DIR

# Всё, что нужно чанкеру. Рамки и признаки шрифта он не смотрит, а Parquet
# отдаёт столбцы по отдельности, поэтому лишние читать незачем.
_CHUNK_COLUMNS = ['doc_id', 'block_id', 'page', 'type', 'text_out',
                  'reliable', 'table_html', 'keep', 'order']


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

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
