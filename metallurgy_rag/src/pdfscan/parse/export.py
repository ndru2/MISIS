"""Выгрузка разобранных блоков в JSONL — исходное сырьё для RAG.

Таблица Excel годится для просмотра глазами, но не как источник для поиска: в
ней нет координат, нет отметки о надёжности текста и нет порядка следования
блоков. Здесь сохраняется всё, на что дальше опираются очистка, чанкинг и
индексация.

Одна строка файла — один блок документа в порядке чтения.
"""

import json
from pathlib import Path

import pdfplumber

from pdfscan.parse.text_layer import element_bbox_in_pdf

# Текст из текстового слоя PDF восстановлен посимвольно и надёжен. Всё остальное
# получено распознаванием изображения: там ломаются индексы, степени и греческие
# буквы, поэтому такие блоки помечаются и обрабатываются осторожнее.
RELIABLE_SOURCES = {'text_layer'}


def _page_sizes(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return {i: (page.width, page.height) for i, page in enumerate(pdf.pages, 1)}


def _layout_summary(element):
    """Сводка начертания: доля математических гарнитур, индексы, степени."""
    info = getattr(element, 'text_layer', None)
    if not info:
        return None
    return {
        'math_font_ratio': round(float(info.get('math_font_ratio', 0.0)), 4),
        'italic_ratio': round(float(info.get('italic_ratio', 0.0)), 4),
        'n_subscript': int(info.get('n_subscript', 0)),
        'n_superscript': int(info.get('n_superscript', 0)),
        'n_fonts': len(info.get('fonts', {}) or {}),
    }


def build_records(elements, pdf_path, languages=(), doc_id=None):
    """Собирает блоки в записи со ссылками на соседей."""
    pdf_path = Path(pdf_path)
    # В библиотеке из вложенных папок имена файлов повторяются, поэтому в
    # качестве опознавателя лучше передавать путь относительно корня.
    doc_id = doc_id or pdf_path.stem
    sizes = _page_sizes(pdf_path)

    records = []
    for element in elements:
        text = str(getattr(element, 'text', '') or '').strip()
        if not text:
            continue

        page = int(getattr(element.metadata, 'page_number', 0) or 0)
        width, height = sizes.get(page, (0.0, 0.0))
        box = element_bbox_in_pdf(element, width, height) if width else None
        source = getattr(element, 'text_source', 'ocr')

        records.append({
            'doc_id': doc_id,
            'source': str(pdf_path),
            'block_id': f'{doc_id}#{len(records)}',
            'order': len(records),
            'page': page,
            'type': str(element.category),
            'text': text,
            'text_source': source,
            'reliable': source in RELIABLE_SOURCES,
            'bbox': None if box is None else {
                'x0': round(box[0], 2), 'top': round(box[1], 2),
                'x1': round(box[2], 2), 'bottom': round(box[3], 2),
            },
            'page_size': {'width': round(width, 2), 'height': round(height, 2)},
            'languages': list(languages),
            'layout': _layout_summary(element),
            'table_html': getattr(element.metadata, 'text_as_html', None),
            'prev_id': None,
            'next_id': None,
        })

    # Соседи проставляются в пределах страницы: последний блок страницы и первый
    # блок следующей связаны формально, но по смыслу разорваны колонтитулом.
    for index, record in enumerate(records):
        if index and records[index - 1]['page'] == record['page']:
            record['prev_id'] = records[index - 1]['block_id']
        if index + 1 < len(records) and records[index + 1]['page'] == record['page']:
            record['next_id'] = records[index + 1]['block_id']

    return records


def export_blocks(elements, pdf_path, languages=(), out_path=None, doc_id=None):
    out_path = Path(out_path or f'{Path(pdf_path).stem}_blocks.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = build_records(elements, pdf_path, languages, doc_id)

    with open(out_path, 'w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    unreliable = sum(1 for r in records if not r['reliable'])
    note = f', из них ненадёжных {unreliable}' if unreliable else ''
    print(f'💾 Сохранено: {out_path} — блоков {len(records)}{note}')
    return records


def load_blocks(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]
