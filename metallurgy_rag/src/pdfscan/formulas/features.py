"""Structural features and text helpers for formula classification."""

import re
from functools import lru_cache

GREEK_LATEX = {
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε', 'varepsilon': 'ε',
    'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'vartheta': 'ϑ', 'iota': 'ι', 'kappa': 'κ',
    'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'pi': 'π', 'rho': 'ρ', 'sigma': 'σ',
    'varsigma': 'ς', 'tau': 'τ', 'upsilon': 'υ', 'phi': 'φ', 'varphi': 'φ', 'chi': 'χ',
    'psi': 'ψ', 'omega': 'ω', 'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ',
    'Xi': 'Ξ', 'Pi': 'Π', 'Sigma': 'Σ', 'Upsilon': 'Υ', 'Phi': 'Φ', 'Psi': 'Ψ', 'Omega': 'Ω',
}

LATEX_OPERATORS = {
    'cdot': '⋅', 'times': '×', 'div': '÷', 'pm': '±', 'mp': '∓', 'leq': '≤', 'le': '≤',
    'geq': '≥', 'ge': '≥', 'neq': '≠', 'ne': '≠', 'approx': '≈', 'equiv': '≡',
    'propto': '∝', 'infty': '∞', 'partial': '∂', 'int': '∫', 'sum': '∑', 'prod': '∏',
    'sqrt': '√', 'rightarrow': '→', 'to': '→', 'leftarrow': '←', 'leftrightarrow': '↔',
    'Rightarrow': '⇒', 'uparrow': '↑', 'downarrow': '↓', 'circ': '°', 'degree': '°',
    'ldots': '…', 'dots': '…', 'cdots': '…', 'in': '∈', 'forall': '∀', 'exists': '∃',
}

# Буквы, которые в кириллице и латинице выглядят одинаково. Химическую запись
# часто набирают вперемешку: «Н₂О» может быть набрано русскими Н и О.
CHEM_HOMOGLYPHS = str.maketrans({
    'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'H', 'К': 'K', 'М': 'M', 'О': 'O',
    'Р': 'P', 'Т': 'T', 'Х': 'X', 'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c',
    'у': 'y', 'х': 'x',
})

STATE_MARKERS = ('т', 'ж', 'г', 'тв', 'к', 'р', 'aq', 's', 'l', 'g', 'cr', 'ад', 'газ')

_LATEX_WRAPPER_RE = re.compile(
    r'\\(?:mathrm|mathit|mathbf|mathsf|text|textrm|textit|rm|it|bf|operatorname)\s*\{([^{}]*)\}')
_LATEX_COMMAND_RE = re.compile(r'\\([A-Za-z]+)')
_LATEX_SPACING_RE = re.compile(r'\\[,;:!]|\\qquad|\\quad|\\ ')
_REACTION_SIGN_RE = re.compile(r'[=+]|→|⟶|↔|⇄|⇌|⇒')
_STATE_RE = re.compile(r'\((?:%s)\)' % '|'.join(STATE_MARKERS))
_CHARGE_RE = re.compile(r'\^\{[^{}]*\}|\^[\d+−–-]+')
_SPLIT_RE = re.compile(r'[=+;–−-]|→|⟶|↔|⇄|⇌|⇒')

PERIODIC_TABLE = {
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
    'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
    'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
    'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
}

AMBIGUOUS_ELEMENTS = {'F', 'H', 'P', 'V', 'S', 'C', 'O', 'K', 'I', 'N', 'W', 'U', 'T'}

# Основы слов ищутся от начала слова, поэтому русские записаны без окончаний, а
# английские — до места, где начинается разнобой форм: «oxid» покрывает oxide,
# oxidation и oxidising.
CHEM_KEYWORDS = [
    'реакц', 'соединени', 'веществ', 'молекул', 'атом', 'раствор', 'сплав', 'легир', 'оксид',
    'реакци', 'легирован', 'электролит', 'катализ', 'ион', 'валент',
    'reaction', 'compound', 'substance', 'molecul', 'atom', 'solution', 'alloy', 'oxid',
    'electrolyt', 'catalys', 'catalyz', 'ion', 'valen', 'reduc', 'reagent', 'acid',
    'sulph', 'sulf', 'chlorid', 'mole', 'stoichiometr', 'equilibri',
]

PHYS_KEYWORDS = [
    'сила', 'напряжен', 'давлен', 'объем', 'энерги', 'мощн', 'ток', 'скорост', 'ускорен',
    'деформац', 'пластич', 'прочност', 'модуль', 'упруг', 'вязкост', 'тепло', 'крутящ',
    'force', 'stress', 'strain', 'pressure', 'volume', 'energy', 'power', 'current',
    'velocit', 'speed', 'acceler', 'deformation', 'plastic', 'strength', 'modulus',
    'elastic', 'viscos', 'heat', 'thermal', 'torque', 'mass', 'temperature',
]

MATH_KEYWORDS = [
    'статистик', 'вероятност', 'матриц', 'векторн', 'предел', 'производн', 'интеграл',
    'функци', 'уравнен', 'теорем', 'доказательств',
    'statistic', 'probabilit', 'matri', 'vector', 'limit', 'derivativ', 'integral',
    'function', 'equation', 'theorem', 'proof', 'ratio', 'estimate', 'coefficient',
]

TECH_SYMBOLS = set('=+-*/^\\_→Δ∫∑°≈·|[]{}()<>!')

TEXT_WHITELIST_PATTERNS = [
    r'^\d{2,4}$',
    # Одинокий номер формулы: сама формула на этой странице вставлена картинкой
    # либо отошла в соседний блок, а «(3.4)» само по себе формулой не является.
    r'^\(\s*\d+(?:[.\-–]\d+)*\s*\)$',
    r'^[PpСс]\.?\s*\d+$',
    r'^[-\d]+\s*[PpСс]\.?\s*\d*$',
    r'^[A-ZА-Я]{1,3}\d+[A-ZА-ЯХ\d-]*$',
    r'^[A-ZА-Я]{2,}\s*[\d.-]+$',
    r'^[A-Za-zА-Яа-я]+\d+/\d+[A-Za-zА-Яа-я]*$',
    r'^IT\d+$',
    r'^Ra\s+[\d.]+',
    r'^Rz\s+\d+',
    r'^M\d+x[\d.]+$',
    r'^Tr\d+x\d+',
    r'^G\s+\d',
    r'^[ⓘ∥⟂⌒/○/]\s*[\d.]+',
]

STRUCTURAL_COLUMNS = [
    'formula_len', 'context_len', 'digit_ratio', 'cyrillic_ratio', 'context_cyrillic_ratio',
    'tech_symbol_ratio',
    'has_equation', 'has_latex', 'has_arrow', 'has_integral', 'has_sum', 'has_greek',
    'n_periodic_elements', 'n_unambiguous_elements', 'n_ambiguous_elements',
    'chem_keyword_hits', 'phys_keyword_hits', 'math_keyword_hits',
    'chem_context_hits', 'phys_context_hits', 'math_context_hits',
    'density', 'alpha_count', 'subscript_like', 'superscript_like',
    # Грамматика химической записи
    'n_species', 'species_ratio', 'chem_parse_junk', 'is_chem_equation', 'has_state_marker',
    # Начертание из текстового слоя. У синтетических примеров разметки нет,
    # поэтому рядом идёт признак её наличия: без него нули «нет засечек» и
    # «неизвестно» слились бы в одно значение.
    'math_font_ratio', 'italic_ratio', 'n_subscript', 'n_superscript', 'n_fonts',
    'has_layout',
]

LAYOUT_COLUMNS = ['math_font_ratio', 'italic_ratio', 'n_subscript', 'n_superscript', 'n_fonts']


def normalize_notation(text: str) -> str:
    """Приводит запись формулы к нотации, которую выдаёт парсер.

    Обучающие примеры записаны в LaTeX, а из текстового слоя приходит юникод.
    Без общей нотации модель училась бы на одном языке записи, а работала на
    другом: признак «в формуле есть греческая буква» срабатывал бы только на
    обучении, где буквы записаны как ``\\sigma``.
    """
    if not text:
        return ''

    text = _LATEX_SPACING_RE.sub(' ', str(text))
    # Граница слова обязательна, иначе \right откусит начало у \rightarrow.
    text = re.sub(r'\\(?:left|right)(?![A-Za-z])', '', text)

    # Обёртки бывают вложенными: \mathrm{\Delta H}.
    for _ in range(3):
        unwrapped = _LATEX_WRAPPER_RE.sub(r'\1', text)
        if unwrapped == text:
            break
        text = unwrapped

    def expand(match):
        name = match.group(1)
        return GREEK_LATEX.get(name) or LATEX_OPERATORS.get(name) or match.group(0)

    # \frac парсер выдаёт сам, поэтому нераспознанные команды остаются как есть.
    text = _LATEX_COMMAND_RE.sub(expand, text)
    return re.sub(r'\s+', ' ', text).strip()


def _strip_decorations(token: str) -> str:
    """Снимает с вещества индексы, заряд и пометку агрегатного состояния."""
    token = re.sub(r'_\{([^{}]*)\}', r'\1', token)
    token = re.sub(r'\^\{[^{}]*\}', '', token)
    token = re.sub(r'\^[\d+−–\-]+', '', token)
    token = _STATE_RE.sub('', token)
    token = re.sub(r'_(\d+)', r'\1', token)
    return token.replace('·', '').replace('⋅', '').replace('*', '')


def parse_species(token: str):
    """Разбирает токен как химическое вещество.

    Возвращает пару «элементы, есть ли внутренний индекс» либо ``None``, если
    запись не раскладывается на символы элементов целиком: частичное совпадение
    ничего не значит, ведь почти любая физическая формула содержит буквы,
    совпадающие с обозначениями элементов.
    """
    token = token.strip(' .,;:').translate(CHEM_HOMOGLYPHS)
    token = _strip_decorations(token)
    token = re.sub(r'^\d+', '', token)  # стехиометрический коэффициент
    if not token:
        return None
    if token in ('e', 'ē'):  # электрон в полуреакции
        return [], False
    # Комплексы записывают в скобках: [TaF7].
    if not (token[0].isupper() or token[0] in '(['):
        return None

    elements, subscripted, position = [], False, 0
    while position < len(token):
        symbol = token[position]
        if symbol in '()[]↑↓':
            position += 1
            continue
        if symbol.isdigit():
            subscripted = True
            position += 1
            continue

        match = re.match(r'[A-Z][a-z]?', token[position:])
        if not match:
            return None
        symbol = match.group(0)
        if symbol not in PERIODIC_TABLE and len(symbol) == 2:
            symbol = symbol[0]
        if symbol not in PERIODIC_TABLE:
            return None
        elements.append(symbol)
        position += len(symbol)

    return (elements, subscripted) if elements else None


def chemical_parse(text: str) -> dict:
    """Раскладывает запись на вещества по обе стороны знака реакции."""
    # Заряд снимается до разбиения: плюс внутри «Н^{+}» иначе разорвал бы
    # вещество пополам и полуреакция перестала бы разбираться.
    normalized = _CHARGE_RE.sub('', normalize_notation(text))

    species, elements, junk, subscripted = [], [], 0, False
    for part in _SPLIT_RE.split(normalized):
        part = part.strip()
        if not part:
            continue
        parsed = parse_species(part)
        if parsed is None:
            junk += 1
            continue
        found, has_subscript = parsed
        species.append(part)
        elements.extend(found)
        subscripted = subscripted or has_subscript
    return {'species': species, 'elements': elements, 'junk': junk, 'subscripted': subscripted}


def looks_like_chemical_equation(text: str) -> bool:
    """Химическое уравнение: несколько веществ вокруг знака реакции.

    Грамматика символов элементов различает химию надёжнее ключевых слов:
    ``2NH_{4}VO_{3} = V_{2}O_{5} + 2NH_{3} + Н_{2}О`` разбирается без остатка,
    а физическая формула из тех же букв — нет.
    """
    if not _REACTION_SIGN_RE.search(normalize_notation(text)):
        return False

    parsed = chemical_parse(text)
    species = len(parsed['species'])
    if species < 2:
        return False
    # Уравнение часто дописывают тепловым эффектом или единицами измерения,
    # поэтому один неразобранный кусок допустим, если вещества всё же
    # составляют подавляющую часть записи.
    if parsed['junk'] and not (species >= 3 and species / (species + parsed['junk']) >= 0.75):
        return False
    # Односимвольные обозначения читаются и как элементы, и как физические
    # величины, поэтому «P = IV» отсекается. Настоящую химию выдаёт либо
    # однозначный символ, либо индекс в составе вещества.
    return parsed['subscripted'] or any(
        element not in AMBIGUOUS_ELEMENTS for element in parsed['elements'])


def is_technical_text_not_formula(text: str) -> bool:
    text = text.strip()
    if not text:
        return True
    for pattern in TEXT_WHITELIST_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    return False


def build_bert_input(formula: str, context: str, max_context_len: int = 400) -> str:
    formula = (formula or '').strip()
    context = (context or '').strip()[:max_context_len]
    return f"[FORMULA] {formula} [CONTEXT] {context}"


@lru_cache(maxsize=8)
def _keyword_regex(keywords: tuple[str, ...]):
    return re.compile(r'\b(?:' + '|'.join(keywords) + ')')


def _keyword_hits(text: str, keywords: list[str]) -> int:
    """Считает основы слов, встретившиеся в тексте.

    Совпадение обязано начинаться с начала слова. Иначе английское «ion»
    находилось бы внутри equation, function и solution, и любая математика
    выглядела бы химией.
    """
    return len(_keyword_regex(tuple(keywords)).findall(text.lower()))


# Обозначения стандартов набирают латиницей и в русском тексте, поэтому ссылка
# вида «ISO 9001» встречается в документах на обоих языках.
STANDARD_REFERENCE_RE = re.compile(
    r'^(?:GOST|ISO|DIN|ASTM|ANSI|EN|BS|JIS|IEC|SAE|AISI|UNS)\s*[\d.\-–/]+\s*$',
    re.IGNORECASE)

_LATEX_COMMAND_ONLY_RE = re.compile(r'\\[A-Za-z]+')
_INDEX_GROUP_RE = re.compile(r'[_^]\{[^{}]*\}|[_^][A-Za-z0-9]')
_WORD_RE = re.compile(r'[^\W\d_]{4,}')


def prose_words(text: str) -> list[str]:
    """Возвращает самостоятельные слова записи, годные для счёта прозы.

    Из текста убираются команды LaTeX и содержимое индексов: ``m_{практ}`` и
    ``\\frac`` состоят из букв, но словами не являются, и без этой чистки
    формула с подписанными индексами считалась бы обычным предложением.
    Письмо не важно, поэтому правило одинаково работает на русском и
    английском.
    """
    stripped = _INDEX_GROUP_RE.sub(' ', _LATEX_COMMAND_ONLY_RE.sub(' ', text))
    return _WORD_RE.findall(stripped)


def _cyrillic_ratio(text: str) -> float:
    """Доля кириллицы среди букв: по ней видно язык окружающего текста.

    Ключевые слова считаются сразу по двум языкам, и без этой подсказки модель
    не отличила бы «в тексте нет химических слов» от «текст на другом языке».
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    cyrillic = sum(1 for c in letters if 'А' <= c <= 'я' or c in 'ёЁ')
    return cyrillic / len(letters)


def extract_structural_features(formula: str, context: str, layout: dict | None = None) -> dict:
    formula = normalize_notation(formula)
    context = (context or '').strip()
    clean = formula.replace(' ', '')
    formula_len = max(len(formula), 1)

    digit_count = sum(c.isdigit() for c in clean)
    alpha_count = sum(c.isalpha() for c in clean)
    cyrillic_count = sum(1 for c in formula if 'А' <= c <= 'я' or c in 'ёЁ')
    tech_count = sum(c in TECH_SYMBOLS for c in clean)

    parsed = chemical_parse(formula)
    n_parts = len(parsed['species']) + parsed['junk']

    return {
        'formula_len': formula_len,
        'context_len': len(context),
        'digit_ratio': digit_count / formula_len,
        'cyrillic_ratio': cyrillic_count / formula_len,
        'context_cyrillic_ratio': _cyrillic_ratio(context),
        'tech_symbol_ratio': tech_count / formula_len,
        'has_equation': int(any(c in clean for c in '=→≈><')),
        'has_latex': int(any(c in clean for c in '\\^_{}')),
        'has_arrow': int(any(c in formula for c in '→⟶↑↓')),
        'has_integral': int('∫' in formula),
        'has_sum': int(any(c in formula for c in '∑∏')),
        'has_greek': int(any('α' <= c <= 'ω' or 'Α' <= c <= 'Ω' for c in formula)),
        'n_periodic_elements': len(parsed['elements']),
        'n_unambiguous_elements': sum(
            1 for el in parsed['elements'] if el not in AMBIGUOUS_ELEMENTS),
        'n_ambiguous_elements': sum(
            1 for el in parsed['elements'] if el in AMBIGUOUS_ELEMENTS),
        'chem_keyword_hits': _keyword_hits(formula, CHEM_KEYWORDS),
        'phys_keyword_hits': _keyword_hits(formula, PHYS_KEYWORDS),
        'math_keyword_hits': _keyword_hits(formula, MATH_KEYWORDS),
        'chem_context_hits': _keyword_hits(context, CHEM_KEYWORDS),
        'phys_context_hits': _keyword_hits(context, PHYS_KEYWORDS),
        'math_context_hits': _keyword_hits(context, MATH_KEYWORDS),
        'density': (tech_count + digit_count) / max(alpha_count, 1),
        'alpha_count': alpha_count,
        'subscript_like': int(bool(re.search(r'[_\{][a-zA-Z0-9]', formula))),
        'superscript_like': int(bool(re.search(r'\^\{?[0-9+\-]', formula))),
        'n_species': len(parsed['species']),
        'species_ratio': len(parsed['species']) / max(n_parts, 1),
        'chem_parse_junk': parsed['junk'],
        'is_chem_equation': int(looks_like_chemical_equation(formula)),
        'has_state_marker': int(bool(_STATE_RE.search(formula))),
        **layout_features(layout),
    }


def layout_features(layout: dict | None) -> dict:
    """Признаки начертания и отметка о том, что разметка вообще известна."""
    if not layout:
        return dict.fromkeys(LAYOUT_COLUMNS, 0.0) | {'has_layout': 0.0}
    return {
        'math_font_ratio': float(layout.get('math_font_ratio', 0.0)),
        'italic_ratio': float(layout.get('italic_ratio', 0.0)),
        'n_subscript': float(layout.get('n_subscript', 0)),
        'n_superscript': float(layout.get('n_superscript', 0)),
        'n_fonts': float(len(layout.get('fonts', {}) or {})),
        'has_layout': 1.0,
    }


def structural_features_matrix(formulas: list[str], contexts: list[str], layouts=None):
    import numpy as np

    if layouts is None:
        layouts = [None] * len(formulas)
    rows = [extract_structural_features(f, c, l)
            for f, c, l in zip(formulas, contexts, layouts)]
    return np.array([[row[col] for col in STRUCTURAL_COLUMNS] for row in rows], dtype=np.float32)
