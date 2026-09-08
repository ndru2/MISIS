"""Адаптер MinerU → единая схема блока, как у ``blocks.jsonl``.

Источник правды — ``*_content_list_v2.json``: там иерархия заголовков, подписи
таблиц и inline-формулы внутри абзаца. Плоский ``*_content_list.json`` берётся
только если v2 нет. Markdown в индекс не идёт: температуры там уже разряжены.

Выход совместим с очисткой и чанкером: те же ``type`` / ``text`` / ``table_html``,
плюс поля MinerU (``text_level``, ``caption``, ``spans``, ``formula_latex``).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pdfscan.formulas.features import looks_like_chemical_equation
from pdfscan.paths import PARSED_DIR
from pdfscan.rag.normalize import clean_text

V2_SUFFIX = '_content_list_v2.json'
V1_SUFFIX = '_content_list.json'

# Типы MinerU, которые в индекс как текст не кладём: у рисунка и графика
# содержанием для поиска является подпись, не пиксели.
_CAPTION_ONLY = {'image', 'chart'}
_FURNITURE = {
    'page_number': 'PageNumber',
    'page_header': 'Header',
    'page_footer': 'Footer',
    'header': 'Header',
    'footer': 'Footer',
}

_TEXT_CMD_RE = re.compile(r'\\text\s*\{([^{}]*)\}')
_DEGREE_RE = re.compile(r'\\mathrm\s*\{\s*(?:\{\s*)?o(?:\s*\})?\s*\}')
_CMD_BRACE_RE = re.compile(r'(\\[A-Za-z]+)\s+\{')
_SUBSUP_BRACE_RE = re.compile(r'([_^])\s+\{')
_SUBSUP_SPACE_RE = re.compile(r'([_^])\s+')
_DIGIT_GAP_RE = re.compile(r'(?<=\d)\s+(?=\d)')
_LETTER_GAP_RE = re.compile(r'(?<=\b[A-Za-z])\s+(?=[A-Za-z]\b)')
_BRACE_GROUP_RE = re.compile(r'\{([^{}]*)\}')
_HOLD = '\x00T{0}\x00'


def repair_mineru_latex(tex: str) -> str:
    """Сжимает разрядку MinerU внутри LaTeX.

    Выносные формулы читаемые, но индексы и числа часто с пробелами:
    ``\\rho_{c u}``, ``1 2 9 0 ^{\\mathrm{o}}C``. Для BM25 это мёртвые токены:
    запрос «1290 °C» и «Fe3O4» их не найдёт. ``\\text{...}`` не трогаем — там
    легенда Stokes law, её нельзя склеивать в одно слово.
    """
    tex = (tex or '').strip()
    if tex.startswith('$$') and tex.endswith('$$'):
        tex = tex[2:-2].strip()
    tex = tex.replace('\n', ' ')

    held = []

    def hold(match):
        held.append('\\text{' + re.sub(r'\s+', ' ', match.group(1)).strip() + '}')
        return _HOLD.format(len(held) - 1)

    tex = _TEXT_CMD_RE.sub(hold, tex)
    tex = _DEGREE_RE.sub(r'\\circ', tex)
    tex = _CMD_BRACE_RE.sub(r'\1{', tex)
    tex = _SUBSUP_BRACE_RE.sub(r'\1{', tex)
    tex = _SUBSUP_SPACE_RE.sub(r'\1', tex)
    tex = re.sub(r'\s+([_^])', r'\1', tex)

    def collapse_group(match):
        body = match.group(1)
        body = _DIGIT_GAP_RE.sub('', body)
        body = _LETTER_GAP_RE.sub('', body)
        body = re.sub(r'\s+', '', body)
        return '{' + body + '}'

    for _ in range(6):
        new = _BRACE_GROUP_RE.sub(collapse_group, tex)
        if new == tex:
            break
        tex = new

    # Цифры схлопываем и снаружи групп («1 2 9 0»), буквы — только внутри
    # скобок: иначе «g d^{2}» в законе Стокса слипнется в «gd».
    tex = _DIGIT_GAP_RE.sub('', tex)
    tex = re.sub(r'\s+', ' ', tex).strip()

    for index, original in enumerate(held):
        tex = tex.replace(_HOLD.format(index), original)
    return tex


def _join_pieces(pieces) -> str:
    """Склеивает куски абзаца, не дублируя пробел и не отрывая запятую."""
    out = ''
    for piece in pieces:
        piece = piece or ''
        if not piece:
            continue
        if not out:
            out = piece
            continue
        if out[-1].isspace() or piece[0].isspace() or piece[0] in ',.;:!?)]':
            out += piece
        else:
            out += ' ' + piece
    return re.sub(r'[ \t]+', ' ', out).strip()


def _span_text(span: dict) -> str:
    kind = span.get('type')
    content = span.get('content', '')
    if kind == 'equation_inline':
        repaired = repair_mineru_latex(str(content))
        return f'${repaired}$' if repaired else ''
    if isinstance(content, str):
        return content
    return ''


def _collect_spans(nodes) -> tuple[str, list[dict]]:
    """Достаёт линейный текст и перечень спанов из v2-узлов."""
    if nodes is None:
        return '', []
    if isinstance(nodes, dict):
        nodes = [nodes]
    pieces, spans = [], []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        kind = node.get('type')
        if kind in {'text', 'equation_inline'}:
            text = _span_text(node)
            if not text:
                continue
            pieces.append(text)
            spans.append({
                'type': 'formula' if kind == 'equation_inline' else 'text',
                'text': text,
            })
            continue
        for key in ('title_content', 'paragraph_content', 'page_header_content',
                    'page_footer_content', 'page_number_content', 'item_content',
                    'table_caption', 'table_footnote', 'image_caption',
                    'image_footnote', 'chart_caption', 'chart_footnote'):
            if key in node:
                nested_text, nested_spans = _collect_spans(node[key])
                pieces.append(nested_text)
                spans.extend(nested_spans)
    return _join_pieces(pieces), spans


def _caption_of(content: dict, *keys: str) -> str:
    parts = []
    for key in keys:
        text, _ = _collect_spans(content.get(key) or [])
        if text:
            parts.append(text)
    return _join_pieces(parts)


def _bbox(raw):
    if not raw or len(raw) < 4:
        return None
    return {
        'x0': round(float(raw[0]), 2),
        'top': round(float(raw[1]), 2),
        'x1': round(float(raw[2]), 2),
        'bottom': round(float(raw[3]), 2),
    }


def _formula_type(latex: str, is_legend: bool) -> str:
    if is_legend:
        return 'Formula'
    # MinerU оставляет пробелы вокруг индексов и ``(g)``; для грамматики химии
    # они шум, для человека — нет. Проверяем и исходник, и сжатую запись.
    compact = re.sub(r'\\tag\s*\{[^{}]*\}', '', latex)
    compact = re.sub(r'\s+', '', compact)
    if looks_like_chemical_equation(latex) or looks_like_chemical_equation(compact):
        return 'Formula (chemistry)'
    return 'Formula'
    if is_legend:
        return 'Formula'
    if looks_like_chemical_equation(latex):
        return 'Formula (chemistry)'
    return 'Formula'


def _is_legend_formula(latex: str) -> bool:
    """Легенда Stokes law приходит как equation, хотя это определения."""
    if not latex:
        return False
    if r'\text{' in latex or r'\text {' in latex:
        return True
    stripped = re.sub(r'\s+', ' ', latex)
    return bool(re.match(
        r'^(?:where|here|with|где|здесь)\b', stripped, re.IGNORECASE))


def _base_record(doc_id, source, page, raw_type, bbox, extra=None):
    record = {
        'doc_id': doc_id,
        'source': str(source),
        'parser': 'mineru',
        'page': page,
        'mineru_type': raw_type,
        'text_source': 'mineru',
        'reliable': True,
        'bbox': _bbox(bbox),
        'page_size': None,
        'languages': [],
        'layout': None,
        'table_html': None,
        'text_level': None,
        'caption': '',
        'footnote': '',
        'spans': None,
        'formula_latex': None,
        'image_path': None,
        'category': doc_id.split('__', 1)[0] if doc_id else '',
        'is_legend': False,
    }
    if extra:
        record.update(extra)
    return record


def convert_v2_block(block: dict, doc_id: str, source, page: int) -> list[dict]:
    """Один блок v2 → один или несколько записей единой схемы."""
    raw_type = block.get('type') or ''
    content = block.get('content') if isinstance(block.get('content'), dict) else {}
    bbox = block.get('bbox')
    records = []

    if raw_type == 'title':
        text, spans = _collect_spans(content.get('title_content'))
        if text:
            records.append(_base_record(doc_id, source, page, raw_type, bbox, {
                'type': 'Title',
                'text': text,
                'text_level': int(content.get('level') or 1),
                'spans': spans,
            }))
        return records

    if raw_type == 'paragraph':
        text, spans = _collect_spans(content.get('paragraph_content'))
        if text:
            records.append(_base_record(doc_id, source, page, raw_type, bbox, {
                'type': 'NarrativeText',
                'text': text,
                'spans': spans,
            }))
        return records

    if raw_type == 'equation_interline':
        latex = repair_mineru_latex(content.get('math_content') or '')
        if not latex:
            return []
        legend = _is_legend_formula(latex)
        image = (content.get('image_source') or {}).get('path')
        records.append(_base_record(doc_id, source, page, raw_type, bbox, {
            'type': _formula_type(latex, legend),
            'text': latex,
            'formula_latex': latex,
            'is_legend': legend,
            'image_path': image,
        }))
        return records

    if raw_type == 'table':
        html = content.get('html') or content.get('table_body') or ''
        caption = _caption_of(content, 'table_caption')
        footnote = _caption_of(content, 'table_footnote')
        image = (content.get('image_source') or {}).get('path')
        if html or caption:
            records.append(_base_record(doc_id, source, page, raw_type, bbox, {
                'type': 'Table',
                'text': _join_pieces([caption, footnote]),
                'table_html': html or None,
                'caption': caption,
                'footnote': footnote,
                'image_path': image,
            }))
        return records

    if raw_type in _CAPTION_ONLY:
        caption_key = 'image_caption' if raw_type == 'image' else 'chart_caption'
        footnote_key = 'image_footnote' if raw_type == 'image' else 'chart_footnote'
        caption = _caption_of(content, caption_key)
        footnote = _caption_of(content, footnote_key)
        text = _join_pieces([caption, footnote])
        if text:
            records.append(_base_record(doc_id, source, page, raw_type, bbox, {
                'type': 'FigureCaption',
                'text': text,
                'caption': caption,
                'footnote': footnote,
                'image_path': (content.get('image_source') or {}).get('path'),
            }))
        return records

    if raw_type == 'list':
        items = content.get('list_items') or []
        list_type = content.get('list_type') or ''
        for item in items:
            text, spans = _collect_spans(item.get('item_content') if isinstance(item, dict) else item)
            if not text:
                continue
            records.append(_base_record(doc_id, source, page, raw_type, bbox, {
                'type': 'ListItem',
                'text': text,
                'spans': spans,
                'list_type': list_type,
            }))
        return records

    if raw_type in _FURNITURE:
        keys = {
            'page_number': 'page_number_content',
            'page_header': 'page_header_content',
            'page_footer': 'page_footer_content',
            'header': 'page_header_content',
            'footer': 'page_footer_content',
        }
        text, spans = _collect_spans(content.get(keys.get(raw_type)))
        if not text and isinstance(block.get('text'), str):
            text = block['text']
            spans = [{'type': 'text', 'text': text}] if text else []
        if text:
            records.append(_base_record(doc_id, source, page, raw_type, bbox, {
                'type': _FURNITURE[raw_type],
                'text': text,
                'spans': spans,
            }))
        return records

    # Незнакомый тип: забираем текст, если он есть, чтобы ничего не потерять.
    text, spans = _collect_spans(content) if content else ('', [])
    if not text:
        text = str(block.get('text') or '').strip()
    if text:
        records.append(_base_record(doc_id, source, page, raw_type, bbox, {
            'type': 'UncategorizedText',
            'text': text,
            'spans': spans or None,
        }))
    return records


def convert_v1_block(block: dict, doc_id: str, source) -> list[dict]:
    """Плоский content_list.json — запасной вход, если v2 нет."""
    page = int(block.get('page_idx', 0)) + 1
    raw_type = block.get('type') or ''
    bbox = block.get('bbox')
    text = block.get('text')

    if raw_type == 'text':
        level = block.get('text_level')
        body = clean_text(str(text or ''))
        if not body:
            return []
        if level:
            return [_base_record(doc_id, source, page, raw_type, bbox, {
                'type': 'Title',
                'text': body,
                'text_level': int(level),
            })]
        return [_base_record(doc_id, source, page, raw_type, bbox, {
            'type': 'NarrativeText',
            'text': body,
        })]

    if raw_type == 'equation':
        latex = repair_mineru_latex(str(text or ''))
        if not latex:
            return []
        legend = _is_legend_formula(latex)
        return [_base_record(doc_id, source, page, raw_type, bbox, {
            'type': _formula_type(latex, legend),
            'text': latex,
            'formula_latex': latex,
            'is_legend': legend,
        })]

    if raw_type == 'table':
        html = block.get('table_body') or block.get('html') or ''
        captions = block.get('table_caption') or []
        footnotes = block.get('table_footnote') or []
        caption = _join_pieces([str(x) for x in captions])
        footnote = _join_pieces([str(x) for x in footnotes])
        if not html and not caption:
            return []
        return [_base_record(doc_id, source, page, raw_type, bbox, {
            'type': 'Table',
            'text': _join_pieces([caption, footnote]),
            'table_html': html or None,
            'caption': caption,
            'footnote': footnote,
            'image_path': block.get('img_path'),
        })]

    if raw_type in {'image', 'chart'}:
        captions = block.get(f'{raw_type}_caption') or []
        footnotes = block.get(f'{raw_type}_footnote') or []
        body = _join_pieces([str(x) for x in captions + footnotes])
        if not body:
            return []
        return [_base_record(doc_id, source, page, raw_type, bbox, {
            'type': 'FigureCaption',
            'text': body,
            'caption': _join_pieces([str(x) for x in captions]),
            'footnote': _join_pieces([str(x) for x in footnotes]),
            'image_path': block.get('img_path'),
        })]

    if raw_type == 'ref_text':
        body = clean_text(str(text or ''))
        if not body:
            return []
        return [_base_record(doc_id, source, page, raw_type, bbox, {
            'type': 'ListItem',
            'text': body,
            'list_type': 'reference_list',
        })]

    mapped = _FURNITURE.get(raw_type)
    if mapped:
        body = clean_text(str(text or ''))
        if not body:
            return []
        return [_base_record(doc_id, source, page, raw_type, bbox, {
            'type': mapped,
            'text': body,
        })]

    body = clean_text(str(text or ''))
    if not body:
        return []
    return [_base_record(doc_id, source, page, raw_type, bbox, {
        'type': 'UncategorizedText',
        'text': body,
    })]


def _finalize(records: list[dict]) -> list[dict]:
    """Проставляет block_id, порядок и соседей в пределах страницы."""
    for index, record in enumerate(records):
        record['order'] = index
        record['block_id'] = f"{record['doc_id']}#{index}"
        record['prev_id'] = None
        record['next_id'] = None
    for index, record in enumerate(records):
        if index and records[index - 1]['page'] == record['page']:
            record['prev_id'] = records[index - 1]['block_id']
        if index + 1 < len(records) and records[index + 1]['page'] == record['page']:
            record['next_id'] = records[index + 1]['block_id']
    return records


def convert_document(data, doc_id: str, source='') -> list[dict]:
    """JSON MinerU (v2-страницы или плоский v1) → список блоков."""
    source = source or doc_id
    records = []
    if isinstance(data, list) and data and isinstance(data[0], list):
        for page_idx, page in enumerate(data):
            for block in page or []:
                if isinstance(block, dict):
                    records.extend(convert_v2_block(block, doc_id, source, page_idx + 1))
        return _finalize(records)

    if isinstance(data, list):
        for block in data:
            if isinstance(block, dict):
                records.extend(convert_v1_block(block, doc_id, source))
        return _finalize(records)

    raise ValueError(f'неожиданный формат MinerU: {type(data).__name__}')


def doc_id_from_path(path: Path) -> str:
    name = Path(path).name
    for suffix in (V2_SUFFIX, V1_SUFFIX):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    parent = Path(path).parent
    if parent.name == 'hybrid_auto':
        return parent.parent.name
    return parent.name


def find_content_list(doc_dir: Path) -> Path | None:
    """Ищет v2, затем v1, внутри ``hybrid_auto`` или в самой папке документа."""
    doc_dir = Path(doc_dir)
    search = [doc_dir / 'hybrid_auto', doc_dir] if (doc_dir / 'hybrid_auto').is_dir() else [doc_dir]
    for folder in search:
        found = sorted(folder.glob(f'*{V2_SUFFIX}'))
        if found:
            return found[0]
    for folder in search:
        found = sorted(folder.glob(f'*{V1_SUFFIX}'))
        if found:
            return found[0]
    return None


def load_mineru_file(path: Path, doc_id: str | None = None) -> list[dict]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    return convert_document(payload, doc_id or doc_id_from_path(path), source=path)


def iter_mineru_documents(root=None):
    """Папки документов MinerU: ``(doc_id, path_to_json, records)``."""
    root = Path(root or PARSED_DIR)
    if not root.exists():
        return
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        json_path = find_content_list(child)
        if json_path is None:
            continue
        yield doc_id_from_path(json_path), json_path, load_mineru_file(json_path)


def export_blocks(records: list[dict], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='MinerU content_list_v2.json → blocks.jsonl')
    parser.add_argument('root', nargs='?', default=str(PARSED_DIR),
                        help='папка parsed_literature_* или один JSON')
    parser.add_argument('--out', default='data/mineru_blocks',
                        help='куда писать jsonl по документам')
    args = parser.parse_args(argv)

    root = Path(args.root)
    out_dir = Path(args.out)
    written = 0

    if root.is_file():
        records = load_mineru_file(root)
        export_blocks(records, out_dir / f'{doc_id_from_path(root)}.jsonl')
        print(f'{doc_id_from_path(root)}: блоков {len(records)}')
        return

    for doc_id, json_path, records in iter_mineru_documents(root):
        export_blocks(records, out_dir / f'{doc_id}.jsonl')
        written += 1
        print(f'{doc_id[-72:]}: блоков {len(records)}')
    if not written:
        raise SystemExit(f'нет *_content_list_v2.json в {root}')
    print(f'готово: {written} документов → {out_dir}')


if __name__ == '__main__':
    main()
