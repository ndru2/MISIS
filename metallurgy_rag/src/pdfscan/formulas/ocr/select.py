"""Отбор формул, которые стоит распознать заново.

Пересматривать все одиннадцать с половиной тысяч формул корпуса не нужно и
вредно: там, где текст собран из символьного слоя PDF, он уже точнее любого
распознавания — индексы, степени и греческие буквы лежат готовыми. Смысл имеют
три случая.

Первый: страница оказалась сканом, и текст формулы пришёл от распознавания
страницы целиком — то есть от модели, которая училась на прозе и про дроби
ничего не знает. Второй: шрифт вставлен без таблицы соответствия, и вместо
символов стоят ``(cid:31)``. Третий, самый неприятный: от формулы остался только
её номер вида ``(3.2)``, потому что сама формула вставлена картинкой поверх
текстового слоя — здесь рамка есть, а содержимого нет вовсе.

Всем трём случаям предшествует проверка рамки. Разметка макета помечает как
формулу не только формулы: под ту же метку попадают ссылка на литературу «[1]»,
маркер списка, обрывок знака «+ –» с края строки, а с другого края — таблица и
фазовая диаграмма. Отличить их по тексту ненадёжно, зато надёжно по размеру и
форме: распознаётся картинка, и рамка, в которую формула не помещается или в
которой ей тесно не бывает, осмысленного ответа не даст — модель либо вернёт тот
же символ, либо, что хуже, уверенно выдумает формулу на его месте.
"""

import json
import re

from . import config

FORMULA_MARK = 'Formula'
RELIABLE_SOURCE = 'text_layer'
OCR_SOURCE = 'formula_ocr'

# Только номер формулы и ничего больше: «(3.2)», «2.1». Квадратные скобки сюда
# тоже попадают, но ссылки на литературу отсеиваются раньше, отдельной проверкой.
_BARE_NUMBER_RE = re.compile(r'^[\s(\[]*\d+(?:[.\-]\d+)*[\s)\]]*$')
# Номер, стоящий в конце строки, — его надо сохранить при замене: на него
# ссылается текст вокруг.
_TRAILING_NUMBER_RE = re.compile(r'(\(\s*\d+(?:[.\-]\d+)*\s*\))\s*$')

# Ссылки на литературу в квадратных скобках: «[1]», «[1] [2]», «[3, 4]»,
# «[5-7]». Разметка регулярно принимает их за формулы, потому что они стоят
# отдельным коротким блоком среди набранного текста.
_CITATION_RE = re.compile(r'^(?:\[\s*\d+(?:\s*[,\-–]\s*\d+)*\s*\]\s*)+$')


def is_formula(block: dict) -> bool:
    return FORMULA_MARK in str(block.get('type', ''))


def is_citation(text) -> bool:
    """Проверяет, что это ссылка на литературу, а не формула."""
    text = (text or '').strip()
    return bool(text) and bool(_CITATION_RE.match(text))


def region_size(block: dict):
    """Ширина и высота рамки блока в пунктах."""
    box = block.get('bbox') or {}
    return (float(box['x1']) - float(box['x0']),
            float(box['bottom']) - float(box['top']))


def region_fits(block: dict, cfg=config.DEFAULT) -> bool:
    """Похожа ли рамка блока на рамку формулы по размеру и форме."""
    width, height = region_size(block)
    if width < cfg.min_region_side_pt or height < cfg.min_region_side_pt:
        return False
    if height > cfg.max_region_height_pt:
        return False
    return width * height >= cfg.min_region_area_pt2


def is_bare_number(text) -> bool:
    """Проверяет, что от формулы остался только её номер."""
    text = (text or '').strip()
    return bool(text) and bool(_BARE_NUMBER_RE.match(text))


def trailing_number(text):
    """Возвращает номер формулы в конце строки, если он там есть."""
    match = _TRAILING_NUMBER_RE.search(text or '')
    return match.group(1) if match else None


def has_bbox(block: dict) -> bool:
    box = block.get('bbox')
    return bool(box) and all(box.get(key) is not None
                             for key in ('x0', 'top', 'x1', 'bottom'))


def reason(block: dict, cfg=config.DEFAULT):
    """Зачем пересматривать этот блок, или ``None``, если не нужно.

    Уже пересмотренные блоки пропускаются: у них стоит отметка ``formula_ocr``.
    """
    if not is_formula(block) or not has_bbox(block):
        return None
    if block.get('formula_ocr') is not None:
        return None
    if not region_fits(block, cfg):
        return None

    text = block.get('text') or ''
    if is_citation(text):
        return None

    if cfg.include_bare_numbers and (not text.strip() or is_bare_number(text)):
        return 'от формулы остался только номер'
    if cfg.include_cid and '(cid:' in text:
        return 'битые символы шрифта'
    if cfg.include_ocr and block.get('text_source') != RELIABLE_SOURCE:
        return 'текст от распознавания страницы'
    return None


def iter_candidates(blocks_path, cfg=config.DEFAULT):
    """Читает blocks.jsonl и отдаёт пары ``(блок, причина)``."""
    with open(blocks_path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                block = json.loads(line)
            except json.JSONDecodeError:
                continue
            why = reason(block, cfg)
            if why:
                yield block, why
