"""Склейка обрывков в блоки и абзацев через границы страниц.

Разметка макета режет текст там, где ей удобно, а не там, где кончается мысль:
в корпусе восемнадцать тысяч блоков класса ``UncategorizedText`` при средней
длине шестнадцать символов — это «образу-», «ргической промышленности» и
«ковша составляет 504, —». Поодиночке такой блок бесполезен и при поиске только
шумит, а собранный обратно в абзац — обычный текст.

Отдельная беда — границы страниц. Экспорт связывает соседей только внутри
страницы, поэтому предложение, перешедшее на следующую страницу, оказывается
разорванным надвое. Слово при этом нередко разорвано переносом, и его надо
склеить без пробела.

Склейка идёт одним проходом в порядке чтения и только между блоками связного
текста: подпись, таблица и формула в неё не втягиваются. Между кандидатами не
должно стоять содержательного блока — иначе абзац приклеился бы через таблицу.
"""

import re
from statistics import median

from . import config

# Склеивать можно только связный текст. Подпись, заголовок, таблица и формула
# самостоятельны: приклеенные к абзацу, они испортят и его, и себя.
MERGEABLE_TYPES = frozenset({'NarrativeText', 'ListItem', 'UncategorizedText'})

_HYPHENS = '-‐‑–'
_TERMINAL = '.!?:;»"”'
_FIRST_LETTER_RE = re.compile(r'[^\W\d_]', re.UNICODE)
_DEFAULT_LINE_HEIGHT = 12.0


def _first_letter(text: str):
    match = _FIRST_LETTER_RE.search(text or '')
    return match.group(0) if match else None


def _starts_lowercase(text: str) -> bool:
    letter = _first_letter(text)
    return bool(letter) and letter.islower()


def _ends_open(text: str) -> bool:
    """Предложение не закончено, значит продолжение где-то рядом."""
    stripped = (text or '').rstrip()
    return bool(stripped) and stripped[-1] not in _TERMINAL


def _ends_hyphen(text: str) -> bool:
    stripped = (text or '').rstrip()
    return bool(stripped) and stripped[-1] in _HYPHENS


def _is_fragment(row: dict, cfg) -> bool:
    """Обрывок строки, а не самостоятельный блок."""
    if row['type'] == 'UncategorizedText':
        return True
    return (row['type'] in MERGEABLE_TYPES
            and len(row.get('text_out') or '') <= cfg.fragment_max_chars)


def _has_box(row: dict) -> bool:
    return row.get('bbox_x0') is not None and row.get('bbox_top') is not None


def _char_width(row: dict) -> float:
    """Оценка ширины символа по рамке блока.

    Настоящего кегля в выгрузке нет, но для блока из одной строки ширина рамки,
    делённая на число символов, приближает его достаточно.
    """
    text = row.get('text_out') or ''
    width = (row.get('bbox_x1') or 0) - (row.get('bbox_x0') or 0)
    return width / max(len(text), 1) if width > 0 else 0.0


def _line_height(rows) -> float:
    """Типичная высота строки на странице, в пунктах PDF."""
    heights = [(row['bbox_bottom'] - row['bbox_top']) for row in rows
               if _has_box(row) and row['bbox_bottom'] > row['bbox_top']]
    return median(heights) if heights else _DEFAULT_LINE_HEIGHT


def _vertical_overlap(first: dict, second: dict) -> float:
    top = max(first['bbox_top'], second['bbox_top'])
    bottom = min(first['bbox_bottom'], second['bbox_bottom'])
    heights = [first['bbox_bottom'] - first['bbox_top'],
               second['bbox_bottom'] - second['bbox_top']]
    smallest = min(heights)
    return (bottom - top) / smallest if smallest > 0 else 0.0


def _horizontal_overlap(first: dict, second: dict) -> float:
    left = max(first['bbox_x0'], second['bbox_x0'])
    right = min(first['bbox_x1'], second['bbox_x1'])
    widths = [first['bbox_x1'] - first['bbox_x0'],
              second['bbox_x1'] - second['bbox_x0']]
    smallest = min(widths)
    return (right - left) / smallest if smallest > 0 else 0.0


def _rule(target: dict, source: dict, line_heights: dict, cfg):
    """Определяет, по какому правилу блоки склеиваются, или ``None``."""
    if target['type'] not in MERGEABLE_TYPES or source['type'] not in MERGEABLE_TYPES:
        return None

    target_text = target.get('text_out') or ''
    source_text = source.get('text_out') or ''
    if not target_text or not source_text:
        return None

    # Перенос слова: дефис на конце и строчная буква в начале продолжения. Это
    # единственное правило, работающее и через границу страницы, потому что
    # разорванное слово ни при каком макете не бывает намеренным.
    if _ends_hyphen(target_text) and _starts_lowercase(source_text):
        return 'hyphen'

    same_page = target['page'] == source['page']

    if same_page and _has_box(target) and _has_box(source):
        fragment = _is_fragment(target, cfg) or _is_fragment(source, cfg)
        if fragment:
            # Одна строка, разрезанная по горизонтали: рамки стоят рядом на
            # одной высоте.
            if _vertical_overlap(target, source) >= cfg.same_line_overlap:
                gap = source['bbox_x0'] - target['bbox_x1']
                width = max(_char_width(target), _char_width(source))
                if width > 0 and 0 <= gap <= cfg.same_line_gap_chars * width:
                    return 'same_line'

            # Продолжение абзаца строкой ниже: колонка та же, предложение не
            # закончено, следующая строка начинается со строчной буквы.
            height = line_heights.get(target['page'], _DEFAULT_LINE_HEIGHT)
            gap = source['bbox_top'] - target['bbox_bottom']
            if (-height <= gap <= cfg.paragraph_gap_lines * height
                    and _horizontal_overlap(target, source) >= cfg.paragraph_x_overlap
                    and _ends_open(target_text)
                    and _starts_lowercase(source_text)):
                return 'paragraph'
        return None

    if not same_page and source['page'] == target['page'] + 1:
        if _ends_open(target_text) and _starts_lowercase(source_text):
            return 'cross_page'

    return None


def _absorb(target: dict, source: dict, rule: str):
    """Переносит текст источника в цель и помечает источник поглощённым."""
    target_text = (target.get('text_out') or '').rstrip()
    source_text = (source.get('text_out') or '').lstrip()

    if rule == 'hyphen':
        target['text_out'] = target_text.rstrip(_HYPHENS) + source_text
    else:
        target['text_out'] = f'{target_text} {source_text}'

    # Класс наследуется от содержательной стороны: абзац, приклеенный к
    # обрывку, остаётся абзацем.
    if (target['type'] == 'UncategorizedText'
            and source['type'] in ('NarrativeText', 'ListItem')):
        target['type'] = source['type']

    if _has_box(target) and _has_box(source):
        target['bbox_x0'] = min(target['bbox_x0'], source['bbox_x0'])
        target['bbox_top'] = min(target['bbox_top'], source['bbox_top'])
        target['bbox_x1'] = max(target['bbox_x1'], source['bbox_x1'])
        target['bbox_bottom'] = max(target['bbox_bottom'], source['bbox_bottom'])
    elif _has_box(source):
        for field in ('bbox_x0', 'bbox_top', 'bbox_x1', 'bbox_bottom'):
            target[field] = source[field]

    target['page_max'] = max(target['page_max'], source['page_max'])
    target['merged_count'] += 1
    target['merged_ids'].append(source['block_id'])
    target['reliable'] = bool(target['reliable']) and bool(source['reliable'])

    source['keep'] = False
    source['drop_reason'] = 'merged'
    source['merged_into'] = target['block_id']


def merge_document(blocks, cfg=config.DEFAULT) -> dict:
    """Склеивает блоки одного документа. Меняет словари на месте.

    Возвращает счётчики по правилам для отчёта.
    """
    line_heights = {}
    by_page = {}
    for block in blocks:
        by_page.setdefault(block['page'], []).append(block)
    for page, page_blocks in by_page.items():
        line_heights[page] = _line_height(page_blocks)

    counts = {}
    chain = []          # блоки, оставшиеся самостоятельными
    blocked = False     # между кандидатами встал содержательный блок

    for block in blocks:
        target = chain[-1] if chain and not blocked else None
        rule = _rule(target, block, line_heights, cfg) if target else None

        if rule:
            _absorb(target, block, rule)
            counts[rule] = counts.get(rule, 0) + 1
            continue

        if block['type'] in MERGEABLE_TYPES:
            chain.append(block)
            blocked = False
        else:
            # Таблица, формула или подпись разрывают абзац по существу, а не
            # по недоразумению разметки: склеивать через них нельзя.
            blocked = True

    return counts


def drop_orphans(blocks, cfg=config.DEFAULT) -> int:
    """Отбрасывает обрывки, которые не удалось ни к чему приклеить.

    Осторожность здесь важнее полноты: под правило попадает только
    ``UncategorizedText``, то есть текст, который разметка сама не смогла ни к
    чему отнести.
    """
    dropped = 0
    for block in blocks:
        if not block['keep'] or block['type'] != 'UncategorizedText':
            continue
        if len(block.get('text_out') or '') <= cfg.orphan_max_chars:
            block['keep'] = False
            block['drop_reason'] = 'orphan'
            dropped += 1
    return dropped
