"""Восстановление содержимого блоков из текстового слоя PDF (этап 1.5).

Макет, классы и Bounding Box по-прежнему приходят от unstructured. Здесь у него
забирается только право решать, какие символы находятся внутри рамки: для
цифровых PDF текст блока пересобирается из символов текстового слоя вместе с
индексами, степенями и информацией о шрифтах.

Базовая линия символа берётся из матрицы текста: в отличие от bottom/top она
не зависит от метрик шрифта, поэтому символы одной строки, набранные разными
гарнитурами, дают одно и то же значение.
"""

import re
import unicodedata
from collections import Counter, defaultdict

import pdfplumber

# Гарнитуры, однозначно указывающие на математический набор.
MATH_FONT_MARKERS = (
    'CMMI', 'CMSY', 'CMEX', 'MSAM', 'MSBM', 'RSFS', 'EUFM', 'EUSM',
    'MATHITALIC', 'MATHEMATICAL', 'SYMBOL', 'MTMI', 'MTSY', 'EUCLIDSYMBOL',
)

ITALIC_FONT_MARKERS = ('ITALIC', 'OBLIQUE', '-IT', ',IT')

SUPERSCRIPT_MAP = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶',
    '7': '⁷', '8': '⁸', '9': '⁹', '+': '⁺', '-': '⁻', '−': '⁻', '–': '⁻',
    '=': '⁼', '(': '⁽', ')': '⁾', 'n': 'ⁿ', 'i': 'ⁱ',
}

SUBSCRIPT_MAP = {
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆',
    '7': '₇', '8': '₈', '9': '₉', '+': '₊', '-': '₋', '−': '₋', '–': '₋',
    '=': '₌', '(': '₍', ')': '₎', 'a': 'ₐ', 'e': 'ₑ', 'o': 'ₒ', 'x': 'ₓ',
}

# Кириллические буквы, неотличимые по начертанию от латинских.
CYRILLIC_TO_LATIN = str.maketrans({
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O',
    'Р': 'P', 'С': 'C', 'Т': 'T', 'У': 'Y', 'Х': 'X', 'І': 'I', 'Ј': 'J',
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
    'ѕ': 's', 'і': 'i', 'ј': 'j',
})

# Шрифт Symbol отдаёт свои знаки в приватной зоне юникода (U+F020..U+F0FF),
# где младший байт — это код по кодировке Adobe Symbol. Без расшифровки
# греческие буквы и знаки операций выпадают из текста как «пустые» символы.
SYMBOL_ENCODING = {
    0x40: '≅', 0x41: 'Α', 0x42: 'Β', 0x43: 'Χ', 0x44: 'Δ', 0x45: 'Ε', 0x46: 'Φ',
    0x47: 'Γ', 0x48: 'Η', 0x49: 'Ι', 0x4a: 'ϑ', 0x4b: 'Κ', 0x4c: 'Λ', 0x4d: 'Μ',
    0x4e: 'Ν', 0x4f: 'Ο', 0x50: 'Π', 0x51: 'Θ', 0x52: 'Ρ', 0x53: 'Σ', 0x54: 'Τ',
    0x55: 'Υ', 0x56: 'ς', 0x57: 'Ω', 0x58: 'Ξ', 0x59: 'Ψ', 0x5a: 'Ζ',
    0x5c: '∴', 0x5e: '⊥',
    0x61: 'α', 0x62: 'β', 0x63: 'χ', 0x64: 'δ', 0x65: 'ε', 0x66: 'φ', 0x67: 'γ',
    0x68: 'η', 0x69: 'ι', 0x6a: 'ϕ', 0x6b: 'κ', 0x6c: 'λ', 0x6d: 'μ', 0x6e: 'ν',
    0x6f: 'ο', 0x70: 'π', 0x71: 'θ', 0x72: 'ρ', 0x73: 'σ', 0x74: 'τ', 0x75: 'υ',
    0x76: 'ϖ', 0x77: 'ω', 0x78: 'ξ', 0x79: 'ψ', 0x7a: 'ζ', 0x7e: '∼',
    0x22: '∀', 0x24: '∃', 0x27: '∋', 0x2a: '∗', 0x2d: '−',
    0xa2: '′', 0xa3: '≤', 0xa4: '⁄', 0xa5: '∞', 0xab: '↔', 0xac: '←', 0xad: '↑',
    0xae: '→', 0xaf: '↓', 0xb0: '°', 0xb1: '±', 0xb2: '″', 0xb3: '≥', 0xb4: '×',
    0xb5: '∝', 0xb6: '∂', 0xb7: '•', 0xb8: '÷', 0xb9: '≠', 0xba: '≡', 0xbb: '≈',
    0xbc: '…', 0xc6: '∅', 0xc7: '∩', 0xc8: '∪', 0xc9: '⊃', 0xca: '⊇', 0xcc: '⊂',
    0xcd: '⊆', 0xce: '∈', 0xcf: '∉', 0xd0: '∠', 0xd1: '∇', 0xd5: '∏', 0xd6: '√',
    0xd7: '⋅', 0xd8: '¬', 0xd9: '∧', 0xda: '∨', 0xdb: '⇔', 0xdc: '⇐', 0xdd: '⇑',
    0xde: '⇒', 0xdf: '⇓', 0xe0: '◊', 0xe1: '⟨', 0xe5: '∑', 0xf1: '⟩', 0xf2: '∫',
    0xa1: 'ϒ', 0xa2: '′', 0xa6: 'ƒ', 0xc0: 'ℵ', 0xc1: 'ℑ', 0xc2: 'ℜ', 0xc3: '℘',
    0xc4: '⊗', 0xc5: '⊕', 0xcb: '⊄', 0xd2: '®', 0xd3: '©', 0xd4: '™',
    # Высокие скобки набраны из кусочков: верхний кусок отдаёт саму скобку,
    # продолжение и нижний кусок отбрасываются, иначе одна скобка размножится
    # по всем строкам, которые она охватывает.
    0xe6: '(', 0xe7: '', 0xe8: '', 0xf6: ')', 0xf7: '', 0xf8: '',
    0xe9: '[', 0xea: '', 0xeb: '', 0xf9: ']', 0xfa: '', 0xfb: '',
    0xec: '{', 0xed: '', 0xee: '', 0xef: '', 0xfc: '}', 0xfd: '', 0xfe: '',
}

# Wingdings в технических документах служит источником маркеров списка, и его
# кодировка не имеет ничего общего с Adobe Symbol: код 0xA7 там не «трефы»,
# а маленький чёрный квадрат.
WINGDINGS_ENCODING = {
    0x6c: '●', 0x6d: '❍', 0x6e: '■', 0x6f: '□', 0x71: '❑', 0x75: '◆',
    0xa7: '▪', 0xa8: '□', 0xfc: '✓', 0xfd: '✗',
    0xe0: '⇦', 0xe1: '⇨', 0xe2: '⇧', 0xe3: '⇩',
}

_CYRILLIC_RE = re.compile(r'[А-Яа-яЁё]')
_LATIN_RE = re.compile(r'[A-Za-z]')
# Глиф шрифта, у которого нет таблицы соответствия юникоду: символ в PDF есть,
# но какой именно — из файла не восстановить.
_CID_RE = re.compile(r'\(cid:\d+\)')

# Доля кегля, на которую символ должен отклониться от базовой линии строки,
# чтобы считаться индексом или степенью.
SUPERSCRIPT_RISE = 0.18
SUBSCRIPT_DROP = 0.08
# Максимальный кегль спутника относительно кегля несущей строки.
SATELLITE_SIZE_RATIO = 0.85
# Допуск при склейке символов в одну базовую линию, в пунктах.
BASELINE_TOLERANCE = 0.6


def _is_math_font(fontname):
    name = (fontname or '').upper()
    return any(marker in name for marker in MATH_FONT_MARKERS)


def _is_italic_font(fontname):
    name = (fontname or '').upper()
    return any(marker in name for marker in ITALIC_FONT_MARKERS)


def _base_fontname(fontname):
    """Отбрасывает префикс подмножества шрифта вида ``BKOOCH+Times-Roman``."""
    return (fontname or '').split('+')[-1]


def normalize_homoglyphs(text):
    """Чинит смешение кириллицы и латиницы внутри одного токена.

    Правится только смешанный токен, все кириллические буквы которого имеют
    латинского двойника: ``ТаF`` почти наверняка набран вместо ``TaF``. Если
    в токене есть хотя бы одна кириллическая буква без двойника, это настоящее
    русское слово рядом с латиницей (``PDFформат``), и его трогать нельзя.
    """
    def fix(match):
        token = match.group(0)
        if len(token) > 20 or not _LATIN_RE.search(token):
            return token
        cyrillic = _CYRILLIC_RE.findall(token)
        if not cyrillic or any(ord(ch) not in CYRILLIC_TO_LATIN for ch in cyrillic):
            return token
        return token.translate(CYRILLIC_TO_LATIN)

    return re.sub(r'[A-Za-zА-Яа-яЁё]+', fix, text)


def normalize_text(text):
    """Приводит юникод к единому виду, не меняя содержательных символов."""
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\u00a0', ' ').replace('\ufb01', 'fi').replace('\ufb02', 'fl')
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _decode_symbol_char(text, fontname):
    """Разворачивает знак декоративного шрифта из приватной зоны юникода.

    Symbol и Wingdings кладут свои знаки в диапазон U+F020..U+F0FF, где младший
    байт — код по собственной кодировке шрифта. Таблицы у них разные, поэтому
    расшифровка обязана смотреть на гарнитуру.
    """
    if len(text) != 1 or not ('\uf020' <= text <= '\uf0ff'):
        return text

    code = ord(text) - 0xf000
    name = (fontname or '').upper()

    if 'WINGDINGS' in name or 'WEBDINGS' in name or 'DINGBAT' in name:
        # Незнакомый дингбат в таком документе — почти наверняка маркер списка.
        return WINGDINGS_ENCODING.get(code, '•')
    if code in SYMBOL_ENCODING:
        return SYMBOL_ENCODING[code]
    # Цифры и знаки препинания в Symbol совпадают с ASCII, а все буквы
    # перечислены в таблице выше.
    return chr(code) if 0x20 <= code < 0x7f else text


def _char_baseline(char, page_height):
    """Базовая линия символа в координатах сверху вниз."""
    matrix = char.get('matrix')
    if matrix and len(matrix) >= 6:
        return page_height - matrix[5]
    return char.get('bottom', 0.0)


def _cluster_by_baseline(chars, page_height):
    """Разбивает символы на группы с общей базовой линией."""
    groups = []
    for char in sorted(chars, key=lambda c: _char_baseline(c, page_height)):
        baseline = _char_baseline(char, page_height)
        if groups and abs(baseline - groups[-1]['baseline']) <= BASELINE_TOLERANCE:
            groups[-1]['chars'].append(char)
        else:
            groups.append({'baseline': baseline, 'chars': [char]})

    for group in groups:
        sizes = [c['size'] for c in group['chars']]
        group['size'] = sorted(sizes)[len(sizes) // 2]
        group['x0'] = min(c['x0'] for c in group['chars'])
        group['x1'] = max(c['x1'] for c in group['chars'])
    return groups


def _claimed_by_fraction(char, baseline, carrier, rules):
    """Принадлежит ли символ дроби, а не индексу предполагаемого носителя.

    Числитель и знаменатель мельче основной строки и смещены по вертикали, то
    есть выглядят в точности как степень и индекс. Отличает их то, что они
    стоят над чертой и под ней, тогда как настоящий индекс делит с носителем и
    сторону черты, и саму дробь.
    """
    for rule in rules:
        inside = rule['x0'] - 1.5 <= char['x0'] and char['x1'] <= rule['x1'] + 1.5
        if not inside or abs(baseline - rule['y']) > 2.5 * max(char['size'], 1.0):
            continue
        carrier_inside = any(rule['x0'] - 1.5 <= c['x0'] and c['x1'] <= rule['x1'] + 1.5
                             for c in carrier['chars'])
        same_side = (carrier['baseline'] - rule['y']) * (baseline - rule['y']) > 0
        if carrier_inside and same_side:
            continue
        return True
    return False


def _attach_satellites(groups, rules=()):
    """Пришивает группы индексов и степеней к их несущим строкам.

    Возвращает список логических строк, где каждый символ помечен ролью
    ``base``, ``sub`` или ``sup``.
    """
    carriers = []
    satellites = []
    for index, group in enumerate(groups):
        neighbours = groups[max(0, index - 2):index + 3]
        largest = max(g['size'] for g in neighbours)
        if group['size'] <= largest * SATELLITE_SIZE_RATIO:
            satellites.append(group)
        else:
            carriers.append(group)

    if not carriers:
        carriers, satellites = groups, []

    for group in carriers:
        group['role_chars'] = [(c, 'base') for c in group['chars']]

    # Носитель всегда крупнее своего спутника, а сам может оказаться спутником
    # строки покрупнее: степень числителя мельче числителя, а тот мельче строки
    # с дробью. Разбор от крупных к мелким гарантирует, что к моменту поиска
    # носителя тот уже занял своё место в списке.
    satellites.sort(key=lambda g: (-g['size'], g['baseline'], g['x0']))

    for satellite in satellites:
        orphans = []
        for char in sorted(satellite['chars'], key=lambda c: c['x0']):
            best, best_score = None, None
            for carrier in carriers:
                # Индекс всегда заметно мельче своего носителя. Без этой
                # проверки сноска целиком уезжает в индексы к своему номеру.
                if satellite['size'] > carrier['size'] * SATELLITE_SIZE_RATIO:
                    continue
                offset = satellite['baseline'] - carrier['baseline']
                if offset < -SUPERSCRIPT_RISE * carrier['size']:
                    role = 'sup'
                elif offset > SUBSCRIPT_DROP * carrier['size']:
                    role = 'sub'
                else:
                    continue
                if abs(offset) > 0.75 * carrier['size']:
                    continue
                if _claimed_by_fraction(char, satellite['baseline'], carrier, rules):
                    continue

                gap = _gap_to_carrier(char, carrier)
                if gap is None or gap > 1.5 * carrier['size']:
                    continue
                # Горизонтальный зазор важнее вертикального: индекс стоит
                # вплотную к своему основанию, тогда как по вертикали к нему
                # может оказаться ближе числитель соседней дроби.
                score = gap + 0.5 * abs(offset)
                if best_score is None or score < best_score:
                    best, best_score = (carrier, role), score

            if best is None:
                orphans.append(char)
            else:
                carrier, role = best
                carrier['role_chars'].append((char, role))

        if orphans:
            satellite['chars'] = orphans
            satellite['role_chars'] = [(c, 'base') for c in orphans]
            satellite['x0'] = min(c['x0'] for c in orphans)
            satellite['x1'] = max(c['x1'] for c in orphans)
            carriers.append(satellite)

    carriers.sort(key=lambda g: (g['baseline'], g['x0']))
    return carriers


def _gap_to_carrier(char, carrier):
    """Расстояние от символа до ближайшего знака несущей строки слева.

    Основание индекса почти всегда стоит слева от него, поэтому смотрим сперва
    влево и только при отсутствии соседа — вправо. В расчёт идут и уже
    привязанные индексы, иначе многобуквенный индекс вроде ``NH4VO3``
    оборвётся на первом же символе, отстоящем от основания.
    """
    anchors = [c for c, _ in carrier['role_chars'] if not c['text'].isspace()]
    left = [c['x1'] for c in anchors if c['x1'] <= char['x0'] + 1.0]
    if left:
        return max(char['x0'] - max(left), 0.0)
    right = [c['x0'] for c in anchors if c['x0'] >= char['x1'] - 1.0]
    if right:
        return max(min(right) - char['x1'], 0.0)
    return None


def _render_line(role_chars, line_size):
    """Собирает строку в двух представлениях: LaTeX-подобном и юникодном.

    LaTeX-подобная форма нужна классификатору — обучающая выборка
    ``train_metallurgy.py`` записана именно в ней. Юникодная удобнее для
    чтения и для эмбеддингов.
    """
    units = sorted(role_chars, key=lambda rc: rc[0]['x0'])

    # Часть документов вообще не кодирует пробелы, разделяя слова только
    # отступом. Там пробел приходится синтезировать по зазору, а где пробелы
    # проставлены явно, порог поднимаем, чтобы не удваивать их.
    has_spaces = any(c['text'].isspace() for c, _ in units)
    space_threshold = (0.30 if has_spaces else 0.15) * max(line_size, 1.0)

    latex_parts, plain_parts = [], []
    pending_role, pending = None, []
    previous = None
    n_sub = n_sup = 0

    def flush():
        nonlocal pending_role, pending
        if not pending:
            return
        body = ''.join(pending)
        if pending_role == 'sup':
            latex_parts.append('^{%s}' % body)
            plain_parts.append(
                ''.join(SUPERSCRIPT_MAP.get(ch, ch) for ch in body)
                if all(ch in SUPERSCRIPT_MAP for ch in body) else '^{%s}' % body
            )
        else:
            latex_parts.append('_{%s}' % body)
            plain_parts.append(
                ''.join(SUBSCRIPT_MAP.get(ch, ch) for ch in body)
                if all(ch in SUBSCRIPT_MAP for ch in body) else '_{%s}' % body
            )
        pending_role, pending = None, []

    for char, role in units:
        text = char['text']
        plain_text = char.get('text_plain', text)
        if previous is not None:
            gap = char['x0'] - previous['x1']
            if gap > space_threshold and not text.isspace():
                flush()
                if latex_parts and not latex_parts[-1].endswith(' '):
                    latex_parts.append(' ')
                    plain_parts.append(' ')
        previous = char

        if role == 'base' or text.isspace():
            flush()
            latex_parts.append(text)
            plain_parts.append(plain_text)
            continue

        if role != pending_role:
            flush()
            pending_role = role
        pending.append(text)
        if role == 'sub':
            n_sub += 1
        else:
            n_sup += 1

    flush()
    return ''.join(latex_parts), ''.join(plain_parts), n_sub, n_sup


def _pseudo_char(text, x0, x1, top, bottom, size, text_plain=None):
    """Символ, синтезированный из графики страницы (дробь, косая черта)."""
    return {
        'text': text,
        'text_plain': text_plain if text_plain is not None else text,
        'x0': x0, 'x1': x1, 'top': top, 'bottom': bottom,
        'size': size, 'fontname': 'synthetic', 'upright': True,
        'width': x1 - x0, 'height': bottom - top,
    }


def _collect_rules(page):
    """Делит графику страницы на дробные черты и косые черты.

    В наборе формул дробь и знак деления часто рисуются линиями, а не
    набираются символами, поэтому в тексте их нет вообще. В зависимости от
    того, чем сверстан документ, черта приходит как линия, как прямоугольник
    или как изображение толщиной в половину пункта.
    """
    # Линейка во всю ширину полосы — это колонтитул или отбивка сноски,
    # а не дробь.
    max_fraction_width = 0.5 * page.width

    horizontal, diagonal = [], []
    for obj in list(page.lines) + list(page.rects) + list(page.images):
        width = obj['x1'] - obj['x0']
        height = obj['bottom'] - obj['top']
        if width < 2:
            continue
        if height <= 1.0 and 3 <= width <= max_fraction_width:
            horizontal.append({'x0': obj['x0'], 'x1': obj['x1'],
                               'y': (obj['top'] + obj['bottom']) / 2})
        elif obj.get('object_type') != 'image' and height >= 3 and 1 < width <= height:
            diagonal.append(obj)
    return horizontal, diagonal


def _insert_slashes(lines, diagonal):
    """Возвращает косые черты в те строки, где они были нарисованы."""
    for obj in diagonal:
        host, best = None, None
        for line in lines:
            distance = abs(line['baseline'] - obj['bottom'])
            if distance > 0.5 * line['size']:
                continue
            if obj['x1'] < line['x0'] - line['size'] or obj['x0'] > line['x1'] + line['size']:
                continue
            if best is None or distance < best:
                host, best = line, distance
        if host is None:
            continue
        size = host['size']
        host['role_chars'].append((
            _pseudo_char('/', obj['x0'], obj['x1'], obj['top'], obj['bottom'], size), 'base'
        ))


def _merge_fractions(lines, horizontal):
    """Сворачивает числитель, черту и знаменатель в одно выражение."""
    for rule in sorted(horizontal, key=lambda r: r['x1'] - r['x0']):
        x0, x1, y = rule['x0'], rule['x1'], rule['y']
        candidates, host, host_distance = [], None, None

        for line in lines:
            size = max(line['size'], 1.0)
            offset = line['baseline'] - y
            inside, has_outside = [], False
            for char, role in line['role_chars']:
                if char['text'].isspace():
                    continue
                if x0 - 1.5 <= char['x0'] and char['x1'] <= x1 + 1.5:
                    inside.append((char, role))
                else:
                    has_outside = True

            # Несущая строка стоит на уровне самой черты: на ней знак равенства
            # и всё, что стоит рядом с дробью.
            if has_outside and abs(offset) < 1.2 * size:
                if host_distance is None or abs(offset) < host_distance:
                    host, host_distance = line, abs(offset)
            if inside:
                candidates.append((line, inside, offset, size))

        numerator = [(line, inside) for line, inside, offset, size in candidates
                     if line is not host and -1.7 * size < offset < -0.05 * size]
        denominator = [(line, inside) for line, inside, offset, size in candidates
                       if line is not host and 0.05 * size < offset < 1.7 * size]

        if not numerator or not denominator:
            continue

        def render(parts):
            parts.sort(key=lambda item: item[0]['baseline'])
            role_chars = [rc for _, chunk in parts for rc in chunk]
            size = max((line['size'] for line, _ in parts), default=10.0)
            latex, plain, _, _ = _render_line(role_chars, size)
            return latex.strip(), plain.strip()

        num_latex, num_plain = render(numerator)
        den_latex, den_plain = render(denominator)
        if not num_latex or not den_latex:
            continue

        consumed = {id(c) for _, chunk in numerator + denominator for c, _ in chunk}
        for line, _ in numerator + denominator:
            line['role_chars'] = [rc for rc in line['role_chars'] if id(rc[0]) not in consumed]

        size = host['size'] if host else max(l['size'] for l, _ in numerator)
        fraction = _pseudo_char(
            r'\frac{%s}{%s}' % (num_latex, den_latex),
            x0, x1, y - size, y + size, size,
            text_plain='(%s)/(%s)' % (num_plain, den_plain),
        )
        if host is None:
            host = {'baseline': y, 'size': size, 'role_chars': []}
            lines.append(host)
        host['role_chars'].append((fraction, 'base'))

    return [line for line in lines
            if any(not c['text'].isspace() for c, _ in line['role_chars'])]


def page_lines(page):
    """Возвращает логические строки страницы, собранные из текстового слоя."""
    chars = [c for c in page.chars if c.get('upright', True)]
    if not chars:
        return []

    page_height = page.height
    for char in chars:
        char['_baseline'] = _char_baseline(char, page_height)
        char['text'] = _decode_symbol_char(char['text'], char['fontname'])
    # Отброшенные продолжения высоких скобок больше не несут текста.
    chars = [c for c in chars if c['text']]
    if not chars:
        return []

    horizontal, diagonal = _collect_rules(page)

    groups = _cluster_by_baseline(chars, page_height)
    lines = []
    for group in _attach_satellites(groups, horizontal):
        role_chars = group.get('role_chars') or [(c, 'base') for c in group['chars']]
        base_sizes = [c['size'] for c, role in role_chars if role == 'base']
        lines.append({
            'baseline': group['baseline'],
            'size': sorted(base_sizes)[len(base_sizes) // 2] if base_sizes else group['size'],
            'x0': min(c['x0'] for c, _ in role_chars),
            'x1': max(c['x1'] for c, _ in role_chars),
            'role_chars': role_chars,
        })

    _insert_slashes(lines, diagonal)
    lines = _merge_fractions(lines, horizontal)

    result = []
    for line in lines:
        role_chars = line['role_chars']
        latex, plain, n_sub, n_sup = _render_line(role_chars, line['size'])
        if not latex.strip():
            continue

        fonts = [_base_fontname(c['fontname']) for c, _ in role_chars if not c['text'].isspace()]
        visible = max(len(fonts), 1)
        line.update({
            'x0': min(c['x0'] for c, _ in role_chars),
            'x1': max(c['x1'] for c, _ in role_chars),
            'top': min(c['top'] for c, _ in role_chars),
            'bottom': max(c['bottom'] for c, _ in role_chars),
            'text_latex': latex,
            'text_plain': plain,
            'n_sub': n_sub,
            'n_sup': n_sup,
            'math_font_ratio': sum(_is_math_font(f) for f in fonts) / visible,
            'italic_ratio': sum(_is_italic_font(f) for f in fonts) / visible,
            'fonts': Counter(fonts),
        })
        result.append(line)

    result.sort(key=lambda ln: (ln['baseline'], ln['x0']))
    return result


def is_digital_page(page, min_chars=40, min_coverage=0.005):
    """Отличает страницу с настоящим текстовым слоем от скана.

    Скан даёт либо пустой слой, либо несколько символов «мусорного» текста,
    поэтому проверяется и количество символов, и доля площади под ними.
    """
    chars = page.chars
    if len(chars) < min_chars:
        return False
    page_area = max(page.width * page.height, 1.0)
    text_area = sum(max(c['x1'] - c['x0'], 0) * max(c['bottom'] - c['top'], 0) for c in chars)
    return text_area / page_area >= min_coverage


def detect_languages(pdf_path, sample_pages=8, min_letters=200, dominance=0.9):
    """Определяет языки документа для распознавания текста.

    Читает буквы из текстового слоя и смотрит, какое письмо преобладает. Если
    слоя нет или письма перемешаны, возвращаются оба языка: распознавание с
    двумя словарями хуже, чем с одним верным, но несравнимо лучше, чем с одним
    неверным. Порядок важен — первым идёт основной язык страницы.
    """
    import pdfplumber

    # Считаются только слова длиной от четырёх букв. Обозначения веществ и
    # величин — Fe, CO, TiCl, σ — набирают латиницей и в русском тексте, и по
    # отдельным буквам химический учебник выглядел бы наполовину английским.
    word = re.compile(r'([А-Яа-яЁё]{4,})|([A-Za-z]{4,})')

    cyrillic = latin = 0
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:sample_pages]:
                for cyr, lat in word.findall(page.extract_text() or ''):
                    if cyr:
                        cyrillic += len(cyr)
                    else:
                        latin += len(lat)
    except Exception:
        return ['rus', 'eng']

    total = cyrillic + latin
    if total < min_letters:
        return ['rus', 'eng']
    if cyrillic / total >= dominance:
        return ['rus']
    if latin / total >= dominance:
        return ['eng']
    return ['rus', 'eng'] if cyrillic >= latin else ['eng', 'rus']


def element_bbox_in_pdf(element, page_width, page_height):
    """Переводит рамку элемента unstructured в координаты PDF (сверху вниз).

    Unstructured отдаёт координаты либо в пикселях отрендеренной страницы,
    либо в пунктах, и с началом отсчёта сверху или снизу, поэтому система
    координат приводится по описанию из метаданных.
    """
    coords = getattr(getattr(element, 'metadata', None), 'coordinates', None)
    if not coords or not getattr(coords, 'points', None):
        return None

    points = coords.points
    x0 = min(p[0] for p in points)
    x1 = max(p[0] for p in points)
    y0 = min(p[1] for p in points)
    y1 = max(p[1] for p in points)

    system = getattr(coords, 'system', None)
    system_width = getattr(system, 'width', None) or page_width
    system_height = getattr(system, 'height', None) or page_height
    scale_x = page_width / system_width
    scale_y = page_height / system_height

    if system is not None and 'Bottom' in type(system).__name__:
        top = (system_height - y1) * scale_y
        bottom = (system_height - y0) * scale_y
    else:
        top = y0 * scale_y
        bottom = y1 * scale_y

    return x0 * scale_x, top, x1 * scale_x, bottom


def _owning_block(char, blocks):
    """Блок, которому принадлежит символ; при вложенности выбирается меньший."""
    cx = (char['x0'] + char['x1']) / 2
    cy = (char['top'] + char['bottom']) / 2

    best, best_area = None, None
    for block in blocks:
        x0, top, x1, bottom = block['bbox']
        if x0 <= cx <= x1 and top <= cy <= bottom:
            area = max((x1 - x0) * (bottom - top), 0.0)
            if best_area is None or area < best_area:
                best, best_area = block, area
    if best is not None:
        return best

    # Символ у самой границы рамки: разрешаем небольшой промах.
    tolerance = max(char['size'], 1.0)
    nearest, nearest_distance = None, None
    for block in blocks:
        x0, top, x1, bottom = block['bbox']
        dx = max(x0 - cx, 0.0, cx - x1)
        dy = max(top - cy, 0.0, cy - bottom)
        distance = (dx * dx + dy * dy) ** 0.5
        if distance <= tolerance and (nearest_distance is None or distance < nearest_distance):
            nearest, nearest_distance = block, distance
    return nearest


def _segment_line(line, blocks):
    """Режет строку на куски по принадлежности блокам.

    Индексы и степени наследуют блок своего базового символа, поэтому формула
    не рассыпается, даже если модель макета отвела индексу отдельную рамку.
    """
    segments = []
    owner = None
    for char, role in sorted(line['role_chars'], key=lambda rc: rc[0]['x0']):
        if role == 'base' and not char['text'].isspace():
            found = _owning_block(char, blocks)
            if found is not None:
                owner = found
        if segments and segments[-1][0] is owner:
            segments[-1][1].append((char, role))
        else:
            segments.append((owner, [(char, role)]))
    return segments


def rebind_elements(elements, pdf_path):
    """Заменяет текст блоков на восстановленный из текстового слоя.

    Страницы без текстового слоя пропускаются: там результат OCR — это всё,
    что есть. Возвращает статистику по документу.
    """
    stats = {
        'digital_pages': [], 'scanned_pages': [],
        'rebound': 0, 'unchanged': 0, 'orphan_lines': 0, 'cid_artifacts': 0,
        'image_blocks': 0,
    }

    by_page = defaultdict(list)
    for element in elements:
        page_number = getattr(getattr(element, 'metadata', None), 'page_number', None)
        if page_number:
            by_page[page_number].append(element)

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_elements = by_page.get(page_number, [])
            if not page_elements:
                continue

            if not is_digital_page(page):
                stats['scanned_pages'].append(page_number)
                stats['unchanged'] += len(page_elements)
                for element in page_elements:
                    element.text_source = 'ocr'
                continue
            stats['digital_pages'].append(page_number)

            blocks = []
            for element in page_elements:
                # На странице с текстовым слоем результат OCR — это вынужденная
                # замена, а не единственный источник, как на скане.
                element.text_source = 'ocr_fallback'
                bbox = element_bbox_in_pdf(element, page.width, page.height)
                if bbox:
                    blocks.append({'el': element, 'bbox': bbox, 'segments': []})
            if not blocks:
                stats['unchanged'] += len(page_elements)
                continue

            for line in page_lines(page):
                for owner, role_chars in _segment_line(line, blocks):
                    if owner is None:
                        if any(not c['text'].isspace() for c, _ in role_chars):
                            stats['orphan_lines'] += 1
                        continue
                    owner['segments'].append((line['baseline'], line['size'], role_chars))

            for block in blocks:
                if not block['segments']:
                    # Под рамкой нет ни одного символа: на странице с текстовым
                    # слоем это значит, что содержимое нарисовано картинкой, и
                    # текст блока остаётся сомнительным результатом OCR.
                    stats['unchanged'] += 1
                    stats['image_blocks'] += 1
                    continue
                stats['cid_artifacts'] += _apply_segments(block['el'], block['segments'])
                block['el'].text_source = 'text_layer'
                stats['rebound'] += 1

    return stats


def _apply_segments(element, segments):
    """Собирает текст блока из его сегментов и складывает разбор в атрибут."""
    segments.sort(key=lambda item: (item[0], min(c['x0'] for c, _ in item[2])))

    latex_lines, plain_lines = [], []
    fonts = Counter()
    n_sub = n_sup = 0
    math_chars = italic_chars = visible_chars = 0
    sizes = []

    for _, line_size, role_chars in segments:
        latex, plain, subs, sups = _render_line(role_chars, line_size)
        latex, plain = normalize_text(latex), normalize_text(plain)
        if not latex:
            continue
        latex_lines.append(normalize_homoglyphs(latex))
        plain_lines.append(normalize_homoglyphs(plain))
        n_sub += subs
        n_sup += sups
        sizes.append(line_size)
        for char, _ in role_chars:
            if char['text'].isspace():
                continue
            name = _base_fontname(char['fontname'])
            fonts[name] += 1
            visible_chars += 1
            math_chars += _is_math_font(name)
            italic_chars += _is_italic_font(name)

    if not latex_lines:
        return 0

    text = ' '.join(latex_lines)
    element.text = text
    element.text_layer = {
        'lines_latex': latex_lines,
        'lines_plain': plain_lines,
        'text_plain': ' '.join(plain_lines),
        'font_size': sorted(sizes)[len(sizes) // 2] if sizes else 0.0,
        'n_subscript': n_sub,
        'n_superscript': n_sup,
        'math_font_ratio': math_chars / max(visible_chars, 1),
        'italic_ratio': italic_chars / max(visible_chars, 1),
        'fonts': dict(fonts),
        'cid_artifacts': len(_CID_RE.findall(text)),
    }
    return element.text_layer['cid_artifacts']
