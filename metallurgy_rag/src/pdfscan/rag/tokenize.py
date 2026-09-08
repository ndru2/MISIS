"""Доменный лексический токенизатор для BM25.

Модельный токенизатор ``bge-m3`` считает бюджет чанка. Этот — другой: из текста
получаются термы для sparse-вектора. Общий word-split ломает химию и единицы
(``Fe2O3`` → ``fe``, ``2``, ``3``; ``кДж/моль`` → мусор), поэтому сначала
защищаются доменные атомы, и только незащищённые слова стеммятся.
"""

from __future__ import annotations

import re

from pdfscan.formulas.features import PERIODIC_TABLE, parse_species

# Плейсхолдер не пересекается с текстом корпуса: управляющий символ + номер.
_HOLD = '\x00P{0}\x00'
_HOLD_RE = re.compile(r'\x00P(\d+)\x00')

# Вещества вроде Fe2O3 / Fe₃O₄ / Al2O3: несколько элементных символов подряд.
_ELEMENT = r'(?:' + '|'.join(sorted(PERIODIC_TABLE, key=len, reverse=True)) + r')'
_SUB = r'(?:[0-9]+|[₀-₉]+)'
_CHEM_CHUNK = _ELEMENT + r'(?:' + _SUB + r')?'
_CHEM_RE = re.compile(r'\b(?:' + _CHEM_CHUNK + r'){2,}(?![a-zA-Z])')
_RATIO_RE = re.compile(
    r'(?:' + _CHEM_CHUNK + r'){1,8}\s*/\s*(?:' + _CHEM_CHUNK + r'){1,8}')

# Единицы, которые нельзя рвать по слэшу или пробелу.
_UNIT_RE = re.compile(
    r'(?:'
    r'мас\.?\s*%|wt\.?\s*%|масс\.?\s*%|'
    r'кДж/моль|кДж·моль-1|kJ/mol|kJ·mol-1|'
    r'Н·м|N·m|'
    r'т/ч|t/h|т/сут|t/d|'
    r'м3/ч|m3/h|'
    r'ppm|MVA|'
    r'°[CС]|degC'
    r')',
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(
    r'(?:'
    r'\d+(?:[.,]\d+)?\s*[·⋅×x]\s*10\^?-?\d+|'
    r'\d+\s*[–−-]\s*\d+\s*%|'
    r'\d+[.,]\d+'
    r')'
)

_STANDARD_RE = re.compile(
    r'(?:GOST|ГОСТ|ISO|DIN|ASTM|ТУ)\s*[\d.\-–/]+',
    re.IGNORECASE,
)

_EQNUM_RE = re.compile(r'\[(?:\d+[A-Za-z]?|\d+[.\-]\d+)\]')

_WORD_RE = re.compile(r'[^\W\d_]+', re.UNICODE)

# Короткий стеммер: только незащищённые слова. Стеммить Fe3O4 нельзя.
_RU_SUFFIX = (
    'иями', 'ями', 'ами', 'ией', 'ой', 'ий', 'ый', 'ое', 'ее', 'ие',
    'ов', 'ев', 'ей', 'ом', 'ем', 'ах', 'ях', 'ию', 'ью', 'ия', 'ья',
    'ть', 'ти', 'ла', 'ли', 'ло', 'ся',
)
_EN_SUFFIX = ('ing', 'ers', 'ies', 'ied', 'ed', 'es', 's')


def _stem(word: str) -> str:
    lower = word.lower()
    if len(lower) <= 4:
        return lower
    cyrillic = any('а' <= c <= 'я' or c == 'ё' for c in lower)
    suffixes = _RU_SUFFIX if cyrillic else _EN_SUFFIX
    for suffix in suffixes:
        if lower.endswith(suffix) and len(lower) - len(suffix) >= 4:
            return lower[: -len(suffix)]
    return lower


def _protect_spans(text: str) -> tuple[str, list[str]]:
    """Подменяет доменные атомы плейсхолдерами, длинные совпадения первыми."""
    spans = []
    for regex in (_RATIO_RE, _CHEM_RE, _UNIT_RE, _STANDARD_RE, _NUMBER_RE, _EQNUM_RE):
        for match in regex.finditer(text):
            spans.append((match.start(), match.end(), match.group(0)))
    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    kept, occupied = [], [False] * (len(text) + 1)
    for start, end, value in spans:
        if any(occupied[start:end]):
            continue
        for pos in range(start, end):
            occupied[pos] = True
        kept.append((start, end, value))
    kept.sort()

    pieces, held, cursor = [], [], 0
    for start, end, value in kept:
        pieces.append(text[cursor:start])
        pieces.append(_HOLD.format(len(held)))
        held.append(value)
        cursor = end
    pieces.append(text[cursor:])
    return ''.join(pieces), held


def chemistry_subtokens(token: str) -> list[str]:
    """Fe2O3 → fe, o; Fe/SiO2 → fe, si, o. Сам токен сюда не входит."""
    found = []
    for part in re.split(r'[/·⋅]', token):
        parsed = parse_species(part.strip())
        if parsed:
            found.extend(element.lower() for element in parsed[0])
    # Уникальные, в порядке появления.
    seen, ordered = set(), []
    for item in found:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def flatten_protected(token: str) -> str:
    """Канонический терм защищённого атома: без пробелов, нижний регистр."""
    token = token.strip().lower()
    token = token.replace('°с', '°c').replace('мас.%', 'мас%').replace('мас. %', 'мас%')
    token = re.sub(r'\s+', '', token)
    # Юникодные индексы → обычные цифры, чтобы Fe₃O₄ совпал с Fe2O3-запросом Fe3O4.
    trans = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
    return token.translate(trans)


def tokenize(text: str) -> list[str]:
    """Термы для BM25: защищённые атомы + стемы обычных слов + подтокены химии."""
    masked, held = _protect_spans(text or '')
    tokens = []

    cursor = 0
    for match in _HOLD_RE.finditer(masked):
        prefix = masked[cursor:match.start()]
        tokens.extend(_plain_tokens(prefix))
        original = held[int(match.group(1))]
        flat = flatten_protected(original)
        tokens.append(flat)
        tokens.extend(chemistry_subtokens(original))
        cursor = match.end()
    tokens.extend(_plain_tokens(masked[cursor:]))
    return [token for token in tokens if token]


def _plain_tokens(text: str) -> list[str]:
    tokens = []
    for word in _WORD_RE.findall(text):
        tokens.append(_stem(word))
    for number in re.findall(r'\d+[.,]?\d*', text):
        tokens.append(number.replace(',', '.'))
    return tokens


def tokenize_unique(text: str) -> list[str]:
    """Как ``tokenize``, но без повторов — удобно смотреть контракт глазами."""
    seen, ordered = set(), []
    for token in tokenize(text):
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered
