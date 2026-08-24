"""Очистка и нормализация блоков перед чанкингом и индексацией.

Текст, пригодный для чтения человеком, и текст, пригодный для поиска, — разные
вещи. Человеку нужна исходная запись формулы, поиску — плоская строка, где
``H_{2}O`` и ``Н₂О`` совпадают, а «оксид железа» находит ``Fe_2O_3``. Поэтому у
каждого блока получается две версии текста и разобранные метаданные: величины с
единицами измерения и состав химической записи.
"""

import re
import unicodedata

from pdfscan.formulas.features import (CHEM_HOMOGLYPHS, chemical_parse,
                                       normalize_notation)

# Названия элементов нужны, чтобы запрос словами находил запись символами.
# Список ограничен тем, что встречается в металлургии и общей химии.
ELEMENT_NAMES = {
    'H': ('водород', 'hydrogen'), 'C': ('углерод', 'carbon'), 'N': ('азот', 'nitrogen'),
    'O': ('кислород', 'oxygen'), 'F': ('фтор', 'fluorine'), 'Na': ('натрий', 'sodium'),
    'Mg': ('магний', 'magnesium'), 'Al': ('алюминий', 'aluminium'), 'Si': ('кремний', 'silicon'),
    'P': ('фосфор', 'phosphorus'), 'S': ('сера', 'sulphur'), 'Cl': ('хлор', 'chlorine'),
    'K': ('калий', 'potassium'), 'Ca': ('кальций', 'calcium'), 'Ti': ('титан', 'titanium'),
    'V': ('ванадий', 'vanadium'), 'Cr': ('хром', 'chromium'), 'Mn': ('марганец', 'manganese'),
    'Fe': ('железо', 'iron'), 'Co': ('кобальт', 'cobalt'), 'Ni': ('никель', 'nickel'),
    'Cu': ('медь', 'copper'), 'Zn': ('цинк', 'zinc'), 'Zr': ('цирконий', 'zirconium'),
    'Nb': ('ниобий', 'niobium'), 'Mo': ('молибден', 'molybdenum'), 'Ag': ('серебро', 'silver'),
    'Sn': ('олово', 'tin'), 'Ta': ('тантал', 'tantalum'), 'W': ('вольфрам', 'tungsten'),
    'Pb': ('свинец', 'lead'), 'B': ('бор', 'boron'), 'He': ('гелий', 'helium'),
}

# Единица измерения: приведение к общему написанию и, где возможно, к системе СИ.
# Составные единицы вроде «кДж/моль» разбираются по частям, поэтому в таблице
# лежат только простые.
UNITS = {
    'мг': ('kg', 1e-6), 'mg': ('kg', 1e-6),
    'г': ('kg', 1e-3), 'g': ('kg', 1e-3),
    'кг': ('kg', 1.0), 'kg': ('kg', 1.0),
    'т': ('kg', 1e3), 't': ('kg', 1e3),
    'мкм': ('m', 1e-6), 'мм': ('m', 1e-3), 'mm': ('m', 1e-3),
    'см': ('m', 1e-2), 'cm': ('m', 1e-2),
    'м': ('m', 1.0), 'm': ('m', 1.0),
    'км': ('m', 1e3), 'km': ('m', 1e3),
    'мс': ('s', 1e-3), 'ms': ('s', 1e-3),
    'с': ('s', 1.0), 's': ('s', 1.0), 'сек': ('s', 1.0),
    'мин': ('s', 60.0), 'min': ('s', 60.0),
    'ч': ('s', 3600.0), 'h': ('s', 3600.0),
    'К': ('K', 1.0), 'K': ('K', 1.0),
    'Дж': ('J', 1.0), 'J': ('J', 1.0),
    'кДж': ('J', 1e3), 'kJ': ('J', 1e3),
    'МДж': ('J', 1e6), 'MJ': ('J', 1e6),
    'ГДж': ('J', 1e9), 'эВ': ('J', 1.602e-19), 'eV': ('J', 1.602e-19),
    'Вт': ('W', 1.0), 'W': ('W', 1.0), 'кВт': ('W', 1e3), 'kW': ('W', 1e3),
    'Па': ('Pa', 1.0), 'Pa': ('Pa', 1.0), 'кПа': ('Pa', 1e3), 'kPa': ('Pa', 1e3),
    'МПа': ('Pa', 1e6), 'MPa': ('Pa', 1e6), 'ГПа': ('Pa', 1e9), 'GPa': ('Pa', 1e9),
    'моль': ('mol', 1.0), 'mol': ('mol', 1.0), 'кмоль': ('mol', 1e3),
    'Н': ('N', 1.0), 'N': ('N', 1.0), 'кН': ('N', 1e3), 'kN': ('N', 1e3),
    'В': ('V', 1.0), 'V': ('V', 1.0), 'А': ('A', 1.0), 'A': ('A', 1.0),
    '%': ('%', 1.0), '°C': ('degC', 1.0), '°С': ('degC', 1.0),
}

_UNIT_ALTERNATION = '|'.join(
    re.escape(u) for u in sorted(UNITS, key=len, reverse=True))

# Величина: число (в русской записи с запятой), при желании со степенью десяти,
# затем единица, возможно составная — «кДж/моль», «Н·м», «кг/м3».
MEASUREMENT_RE = re.compile(
    r'(?<![\w.,])'
    r'(?P<value>\d+(?:[.,]\d+)?)'
    r'(?:\s*[·⋅*x×]\s*10\^?\{?(?P<exp>[−\-+]?\d+)\}?)?'
    r'\s*'
    r'(?P<unit>(?:' + _UNIT_ALTERNATION + r')(?:\s*[/·⋅]\s*(?:' + _UNIT_ALTERNATION + r')\d?)*)'
    r'(?![\w])'
)

_ZERO_WIDTH_RE = re.compile(r'[\u200b-\u200f\u202a-\u202e\u2060\ufeff\u00ad]')
_CID_RE = re.compile(r'\(cid:\d+\)')

# Модель формул разряжает буквы внутри слов: «Weight loss» приходит из неё как
# «\mathrm{W e i g h t\;l o s s}». Глазами такая запись читается нормально, а для
# поиска она мертва — слово распадается на отдельные буквы, и по запросу «weight
# loss» блок не найдётся ни при каком ранжировании.
#
# Границу между словами держат latex-отбивки, перевод строки и разделитель
# колонок. Поэтому они сначала становятся особым знаком, потом склеиваются
# одиночные буквы, и только потом знак снова разворачивается в пробел: сделай
# наоборот, и «Weight loss» слипнется в «Weightloss».
_LATEX_GAP_RE = re.compile(r'\\\\|\\[,;:!]|\\qquad\b|\\quad\b|\\ |[~&]')
_LONE_LETTER_GAP_RE = re.compile(r'(?<=\b[A-Za-zА-Яа-я]) (?=[A-Za-zА-Яа-я]\b)')
_GAP_MARK = '\x00'
# Дефис на конце строки: слово разорвано переносом. У настоящего составного
# слова пробела после дефиса нет, поэтому пробел здесь и отличает одно от другого.
_HYPHEN_BREAK_RE = re.compile(r'(\w{2,})[-‐‑]\s+([а-яёa-z]\w*)')
_SPACE_RE = re.compile(r'[ \t\u00a0\u2000-\u200a\u202f\u205f]+')


def clean_text(text: str) -> str:
    """Приводит текст блока к виду, пригодному для чтения и для поиска."""
    text = unicodedata.normalize('NFKC', text or '')
    text = _ZERO_WIDTH_RE.sub('', text)
    text = _CID_RE.sub('', text)
    text = _HYPHEN_BREAK_RE.sub(r'\1\2', text)
    text = _SPACE_RE.sub(' ', text)
    return re.sub(r'\s*\n\s*', '\n', text).strip()


def latinize_chemistry(text: str) -> str:
    """Переводит кириллические двойники в латиницу.

    В русских учебниках «Н₂О» набирают русскими буквами, и по запросу «H2O»
    такая запись не находится. Замена делается только для формул: в обычном
    тексте она превратила бы слова в бессмыслицу.
    """
    return (text or '').translate(CHEM_HOMOGLYPHS)


def join_spaced_letters(text: str) -> str:
    """Склеивает слова, разряженные по буквам: ``W e i g h t`` → ``Weight``."""
    text = _LATEX_GAP_RE.sub(_GAP_MARK, text or '')
    text = _LONE_LETTER_GAP_RE.sub('', text)
    return text.replace(_GAP_MARK, ' ')


def flatten_formula(text: str) -> str:
    """Разворачивает запись формулы в плоскую строку для поиска.

    Индексы и степени теряют фигурные скобки, дробь становится делением. Так
    ``H_{2}O`` совпадает с ``H2O`` из запроса, а ``\\frac{a}{b}`` — с ``a/b``.

    Разряженные буквы склеиваются в самом начале, пока latex-отбивки ещё на
    месте: ниже ``normalize_notation`` заменит их обычными пробелами, и границы
    слов будет не отличить от разрядки внутри слова.
    """
    text = normalize_notation(join_spaced_letters(text))

    # Индексы сворачиваются раньше дробей: пока внутри числителя остаются свои
    # фигурные скобки, \frac{m_{практ}}{m_{теор}} не разбирается.
    for _ in range(4):
        new = re.sub(r'_\{([^{}]*)\}', r'\1', text)
        new = re.sub(r'\^\{([^{}]*)\}', r'^\1', new)
        if new == text:
            break
        text = new
    text = re.sub(r'_([A-Za-zА-Яа-я0-9])', r'\1', text)

    for _ in range(4):  # вложенные дроби разворачиваются послойно
        new = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', text)
        if new == text:
            break
        text = new

    text = text.replace('\\', ' ')
    return _SPACE_RE.sub(' ', text).strip()


def _parse_unit(raw: str):
    """Разбирает единицу измерения, в том числе составную."""
    raw = _SPACE_RE.sub('', raw)
    parts = re.split(r'([/·⋅])', raw)
    canonical, factor, simple = [], 1.0, True

    for part in parts:
        if part in ('/', '·', '⋅'):
            canonical.append('/' if part == '/' else '*')
            simple = False
            continue
        base = part.rstrip('0123456789')
        power = part[len(base):]
        known = UNITS.get(base)
        if known is None:
            return None
        canonical.append(known[0] + power)
        if not canonical[:-1]:
            factor = known[1]
        else:
            simple = False

    return {
        'canonical': ''.join(canonical),
        'si_factor': factor if simple else None,
    }


def extract_units(text: str) -> list[dict]:
    """Находит величины с единицами измерения.

    Единицы выносятся в метаданные отдельно от текста: по ним можно отбирать
    блоки при поиске, а приведение к общему написанию позволяет сопоставлять
    «кДж/моль» из документа с «kJ/mol» из запроса.
    """
    found = []
    for match in MEASUREMENT_RE.finditer(text):
        parsed = _parse_unit(match.group('unit'))
        if parsed is None:
            continue

        value = float(match.group('value').replace(',', '.'))
        exponent = match.group('exp')
        if exponent:
            value *= 10 ** int(exponent.replace('−', '-'))

        item = {
            'value': value,
            'unit': _SPACE_RE.sub('', match.group('unit')),
            'canonical': parsed['canonical'],
            'text': match.group(0).strip(),
        }
        if parsed['si_factor'] is not None:
            item['si_value'] = value * parsed['si_factor']
        found.append(item)
    return found


def chemistry_keywords(text: str) -> dict:
    """Состав химической записи: символы элементов и их названия."""
    parsed = chemical_parse(latinize_chemistry(text))
    elements = sorted(set(parsed['elements']))
    names = []
    for element in elements:
        names.extend(ELEMENT_NAMES.get(element, ()))
    return {'elements': elements, 'element_names': names}


def normalize_record(record: dict) -> dict:
    """Дополняет блок очищенным текстом, поисковой строкой и метаданными."""
    raw = record.get('text', '')
    block_type = str(record.get('type', ''))
    is_formula = 'Formula' in block_type

    cleaned = clean_text(raw)
    record['text_clean'] = cleaned
    record['has_cid_artifacts'] = bool(_CID_RE.search(raw))

    if is_formula:
        search = flatten_formula(latinize_chemistry(cleaned))
    else:
        search = cleaned
    record['text_search'] = search

    record['units'] = extract_units(cleaned)

    if 'chemistry' in block_type:
        record.update(chemistry_keywords(cleaned))
    else:
        record['elements'] = []
        record['element_names'] = []

    return record


def normalize_blocks(records: list[dict]) -> list[dict]:
    return [normalize_record(r) for r in records]
