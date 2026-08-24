"""Чтение блоков из JSONL и потоковое хранение стадий в Parquet.

JSONL удобно писать по одному документу, но неудобно читать двести раз: каждый
пересчёт порогов означал бы полный обход сорока миллионов символов. Parquet
читается по столбцам, поэтому аудит по одному полю не тянет за собой текст.

Ключевое решение — одна группа строк на документ. Все шаги очистки, кроме
сравнения документов между собой, работают внутри одного документа: и
повторяемость колонтитулов, и склейка абзацев, и поиск точных дублей. Поэтому
корпус незачем держать в памяти целиком — двести тысяч блоков словарями с
четырьмя десятками полей занимают гигабайты и на ноутбуке уходят в своп. Границы
групп строк совпадают с границами документов, значит документ читается и
пишется по одному, а расход памяти не зависит от размера корпуса.
"""

import glob
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pdfscan import paths

# Порядок колонок фиксирован: так глазами в parquet-схеме видно, что стадия
# ничего не потеряла.
RAW_FIELDS = [
    ('doc_id', pa.string()),
    ('source', pa.string()),
    ('block_id', pa.string()),
    ('order', pa.int32()),
    ('page', pa.int32()),
    ('type', pa.string()),
    ('text', pa.string()),
    ('text_source', pa.string()),
    ('reliable', pa.bool_()),
    ('bbox_x0', pa.float32()),
    ('bbox_top', pa.float32()),
    ('bbox_x1', pa.float32()),
    ('bbox_bottom', pa.float32()),
    ('page_width', pa.float32()),
    ('page_height', pa.float32()),
    ('languages', pa.string()),
    ('math_font_ratio', pa.float32()),
    ('italic_ratio', pa.float32()),
    ('n_subscript', pa.int32()),
    ('n_superscript', pa.int32()),
    ('n_fonts', pa.int32()),
    ('table_html', pa.string()),
    ('prev_id', pa.string()),
    ('next_id', pa.string()),
]

RAW_SCHEMA = pa.schema(RAW_FIELDS)


def _flatten(record: dict) -> dict:
    """Разворачивает вложенные bbox, page_size и layout в плоские колонки."""
    box = record.get('bbox') or {}
    size = record.get('page_size') or {}
    layout = record.get('layout') or {}
    return {
        'doc_id': record.get('doc_id'),
        'source': record.get('source'),
        'block_id': record.get('block_id'),
        'order': record.get('order'),
        'page': record.get('page'),
        'type': str(record.get('type', '')),
        'text': record.get('text') or '',
        'text_source': record.get('text_source'),
        'reliable': bool(record.get('reliable')),
        'bbox_x0': box.get('x0'),
        'bbox_top': box.get('top'),
        'bbox_x1': box.get('x1'),
        'bbox_bottom': box.get('bottom'),
        'page_width': size.get('width'),
        'page_height': size.get('height'),
        'languages': ','.join(record.get('languages') or ()),
        'math_font_ratio': layout.get('math_font_ratio'),
        'italic_ratio': layout.get('italic_ratio'),
        'n_subscript': layout.get('n_subscript'),
        'n_superscript': layout.get('n_superscript'),
        'n_fonts': layout.get('n_fonts'),
        'table_html': record.get('table_html'),
        'prev_id': record.get('prev_id'),
        'next_id': record.get('next_id'),
    }


def iter_jsonl(path) -> list[dict]:
    """Читает один blocks.jsonl, пропуская битые строки.

    Прогон на двухсот документах прерывался, и последняя строка файла может быть
    недописанной. Ронять весь этап из-за неё нельзя.
    """
    rows = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


class DocumentWriter:
    """Пишет Parquet по документу за раз, одна группа строк на документ."""

    def __init__(self, path, schema):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self._writer = pq.ParquetWriter(self.path, schema, compression='zstd')
        self.documents = 0
        self.rows = 0

    def write(self, rows: list[dict]):
        if not rows:
            return
        self._writer.write_table(pa.Table.from_pylist(rows, schema=self.schema))
        self.documents += 1
        self.rows += len(rows)

    def close(self):
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def ingest(pattern=None, out_path=None, progress=True) -> Path:
    """Собирает все blocks.jsonl в одну таблицу, документ за документом."""
    out_path = Path(out_path or paths.DATA_DIR / 'blocks_raw.parquet')

    if pattern is None:
        files = paths.blocks_files()
    else:
        files = sorted(Path(p) for p in glob.glob(str(pattern), recursive=True))
    if not files:
        raise FileNotFoundError('не найдено ни одного blocks.jsonl; '
                                'сначала: python -m pdfscan.parse.extractor pdf/')

    with DocumentWriter(out_path, RAW_SCHEMA) as writer:
        for position, path in enumerate(files, 1):
            writer.write([_flatten(record) for record in iter_jsonl(path)])
            if progress and position % 25 == 0:
                print(f'  прочитано {position}/{len(files)} документов, '
                      f'{writer.rows} блоков')

    if progress:
        print(f'{writer.documents} документов, {writer.rows} блоков → {out_path}')
    return out_path


def iter_batches(path, columns=None):
    """Отдаёт блоки кусками по группам строк.

    Для сплошного просмотра корпуса — например, при обучении символьной модели,
    где границы документов не важны.
    """
    reader = pq.ParquetFile(path)
    for group in range(reader.num_row_groups):
        rows = reader.read_row_group(group, columns=columns).to_pylist()
        if rows:
            yield rows


def iter_documents(path, columns=None):
    """Отдаёт документы по одному: список блоков в порядке чтения.

    Границы определяются сменой ``doc_id``, а не границами групп строк. Полагаться
    на раскладку файла нельзя: таблицу мог записать другой инструмент или прежняя
    версия этого модуля, и тогда в одной группе оказалось бы несколько документов
    сразу — а все шаги очистки считают, что видят документ целиком.
    """
    projection = None
    if columns is not None:
        projection = list(columns)
        if 'doc_id' not in projection:
            projection.append('doc_id')

    current, current_id = [], None
    for rows in iter_batches(path, projection):
        for row in rows:
            if row['doc_id'] != current_id:
                if current:
                    yield current
                current, current_id = [], row['doc_id']
            current.append(row)
    if current:
        yield current


def count_documents(path) -> int:
    """Число документов в таблице."""
    doc_ids = pq.read_table(path, columns=['doc_id']).column('doc_id')
    return len(doc_ids.unique())


def load(path, columns=None) -> list[dict]:
    """Читает стадию целиком. Только для выборок и тестов, не для прогона."""
    return pq.read_table(path, columns=columns).to_pylist()
