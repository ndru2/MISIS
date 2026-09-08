"""Явное обрамление MinerU: номера страниц, колонтитулы, пустые подписи.

У unstructured колонтитул ловили по повторяемости: номер страницы каждый раз
другой, и без сведения цифр к одному символу он ускользал. MinerU размечает
``page_number`` / ``page_header`` / ``page_footer`` сам, поэтому ждать трёх
повторов незачем — короткий доклад иначе оставил бы все номера в индексе.
"""

import re

from . import config

# Заголовок раздела MinerU отдаёт типом Title. Колонтитул с названием главы
# можно было бы оставить как подсказку, но голый номер и строка сборника в
# поиске только шумят. Подсказку раздела берёт чанкер из Title.
DROP_TYPES = config.FURNITURE_TYPES
CAPTION_TYPES = frozenset({'FigureCaption', 'Caption'})

# Панель рисунка без описания, голый номер и ось графика. «Figure 1: slag Cu»
# и «Рис. 2.5. Структура льда» сюда не попадают — после номера есть слова.
# Между буквами панели обязателен разделитель: иначе «Pechenga» разбиралась
# как восемь панелей подряд.
_PANEL_RE = re.compile(
    r'^(?:'
    r'(?:[\(\[][a-zа-яё][\)\]]|[a-zа-яё][\)\].])'
    r'(?:\s*(?:[\(\[][a-zа-яё][\)\]]|[a-zа-яё][\)\].]))*'
    r'|fig(?:ure)?\.?\s*[\dIVXLC]+(?:\.\d+)*'
    r'|рис(?:\.|унок)?\s*[\dIVXLC]+(?:\.\d+)*'
    r')\s*$',
    re.IGNORECASE,
)
_AXIS_TICK_RE = re.compile(
    r'^(?:'
    r'\d+(?:[.,]\d+)?\s*(?:°C|°С|min|мин|h|ч|wt%|мас\.?\s*%|%)?'
    r'|[A-Z][a-z]?\s*(?:wt%|мас\.?\s*%|%)'
    r')\s*$',
    re.IGNORECASE,
)
_HAS_WORD_RE = re.compile(r'[A-Za-zА-Яа-яЁё]{4,}')


def is_caption_only(text: str) -> bool:
    """Подпись без содержания: панель, номер рисунка, деление оси."""
    stripped = (text or '').strip()
    if not stripped or len(stripped) <= 2:
        return True
    if _PANEL_RE.match(stripped) or _AXIS_TICK_RE.match(stripped):
        return True
    return len(stripped) < 40 and not _HAS_WORD_RE.search(stripped)


def drop_furniture(rows) -> int:
    """Снимает отбор у номеров страниц, колонтитулов и разрывов страницы."""
    dropped = 0
    for row in rows:
        if not row.get('keep'):
            continue
        if row.get('type') not in DROP_TYPES:
            continue
        row['keep'] = False
        row['drop_reason'] = 'furniture'
        dropped += 1
    return dropped


def drop_caption_only(rows) -> int:
    """Снимает подписи, в которых нет описания рисунка."""
    dropped = 0
    for row in rows:
        if not row.get('keep'):
            continue
        if row.get('type') not in CAPTION_TYPES:
            continue
        if not is_caption_only(row.get('text_out') or row.get('text') or ''):
            continue
        row['keep'] = False
        row['drop_reason'] = 'caption_only'
        dropped += 1
    return dropped
