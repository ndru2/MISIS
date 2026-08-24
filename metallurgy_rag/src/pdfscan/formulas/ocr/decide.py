"""Решение, лучше ли распознанный вариант того, что уже стоит в блоке.

Это главная часть прохода, и её нельзя пропустить. Модель формул ошибается
иначе, чем распознавание страницы: она не рассыпает текст, а уверенно выдаёт
гладкую, но чужую разметку — иногда обрывок, иногда бесконечный повтор одного
множителя. Замена без проверки испортила бы часть корпуса молча, и найти это
потом было бы нечем.

Правила устроены так, чтобы отказ был дешевле ошибки. Текст, собранный из
символьного слоя PDF, не заменяется никогда — кроме случая, когда от формулы там
остался только её номер, то есть содержимого нет вовсе. Спорные случаи получают
решение ``unsure`` и по умолчанию не применяются: плохая, но заметная формула
лучше подменённой.
"""

import re

from pdfscan.prepare import textstats

from . import config, select

# Обёртки макета, которые модель формул ставит всегда, независимо от того, что
# нашла. Одиночная ссылка «[1]» возвращается как «\begin{aligned}&[1]\end{aligned}»,
# и если считать эти команды структурой, то пустой ответ окажется богаче любого
# исходного текста, а замена пройдёт на ровном месте. Поэтому перед подсчётом
# разметка выравнивания снимается. Осмысленные окружения — cases, matrix и
# прочие — не в этом списке и остаются: они действительно говорят о структуре.
_WRAPPER_ENVIRONMENTS = (
    'aligned', 'align', 'alignat', 'gathered', 'gather', 'split',
    'equation', 'displaymath',
)
# У этих двух сразу за именем идёт спецификация колонок вида «{r l r}»: её тоже
# надо снять, иначе она сойдёт за группировку. У остальных окружений следующая
# фигурная скобка — уже содержимое, и трогать её нельзя.
_COLUMN_ENVIRONMENTS = ('array', 'tabular')


def _env_names(names) -> str:
    return '|'.join(name + r'\*?' for name in names)


_WRAPPER_RE = re.compile(
    rf'\\begin\{{(?:{_env_names(_COLUMN_ENVIRONMENTS)})\}}\s*(?:\{{[^{{}}]*\}})?'
    rf'|\\begin\{{(?:{_env_names(_WRAPPER_ENVIRONMENTS)})\}}'
    rf'|\\end\{{(?:{_env_names(_WRAPPER_ENVIRONMENTS + _COLUMN_ENVIRONMENTS)})\}}'
)
# Начертание, кегль и отбивки: к смыслу записи они не добавляют ничего, а
# «\mathrm» модель ставит вокруг каждой второй буквы.
_STYLE_RE = re.compile(
    r'\\(?:displaystyle|textstyle|scriptstyle|scriptsize|small|large|normalsize)\b'
    r'|\\(?:mathrm|mathbf|mathit|mathsf|mathtt|bf|it|rm|sf|tt)\b'
    r'|\\(?:quad|qquad)\b'
    r'|\\\\'
    r'|\\[,;:!]'
    r'|[&~]'
)

# Признаки того, что в записи есть математическая структура, а не плоская
# строка. Распознавание страницы теряет именно их: дробь превращается в две
# строки, индекс — в обычную цифру.
_STRUCTURE_TOKENS = (
    r'\\frac', r'\\dfrac', r'\\sqrt', r'\\sum', r'\\int', r'\\prod', r'\\lim',
    r'\\left', r'\\right', r'\\cdot', r'\\times', r'\\partial', r'\\Delta',
    r'\\alpha', r'\\beta', r'\\gamma', r'\\delta', r'\\sigma', r'\\tau',
    r'\\lambda', r'\\mu', r'\\rho', r'\\theta', r'\\omega', r'\\pi',
    r'\\rightarrow', r'\\leftrightarrow', r'\\approx', r'\\leq', r'\\geq',
    r'\\begin', r'\^', r'_', r'=',
)
_STRUCTURE_RE = re.compile('|'.join(_STRUCTURE_TOKENS))

# Знаки, которых в ответе быть не может. Иероглифы, кана и полноширинные формы
# приходят из обучающих данных модели: в корпусе по металлургии на русском и
# английском им взяться неоткуда, и их появление означает, что модель отвечает
# не по картинке. Тег картинки — то же самое, сказанное прямо: на диаграммах
# модель возвращает «\text{<img>image.png</img>}», расписываясь в бессилии.
_ALIEN_RE = re.compile(r'[\u3000-\u30ff\u4e00-\u9fff\uff00-\uffef]|<img')

# Формула что-то утверждает: в ней есть знак отношения либо хотя бы буквенное
# обозначение — переменная, элемент, функция. Ответ, в котором нет ни того, ни
# другого, приходит с подписей осей графика или из строки таблицы, где формулы
# не было вовсе: модель складывает подвернувшиеся числа в столбик дробей и
# выдаёт разметку тем увереннее, чем меньше в ней смысла.
_RELATION_RE = re.compile(
    r'=|<|>|\\to\b|\\rightarrow\b|\\leftrightarrow\b|\\xrightarrow\b'
    r'|\\longrightarrow\b|\\approx\b|\\leq\b|\\geq\b|\\equiv\b|\\propto\b'
)
_COMMAND_RE = re.compile(r'\\[a-zA-Z]+')
_LETTER_RE = re.compile(r'[A-Za-zА-Яа-я]')

# Повтор одного и того же куска — типичный срыв модели формул: «x + x + x + …».
# Повторяющаяся единица берётся вместе с разделителем, иначе одиночные символы,
# а это самый частый случай срыва, в неё не укладываются.
_RUNAWAY_RE = re.compile(r'(\S+(?:\s+\S+){0,3}\s+)\1{4,}')
# Тот же срыв без пробелов: «xxxxxxxxxxxx». Фигурные скобки и служебные знаки
# исключены — в матрицах и выравниваниях они повторяются законно.
_RUNAWAY_TIGHT_RE = re.compile(r'([^\s{}&\\])\1{11,}')

# Растягивающиеся скобки идут только парой, и непарность означает обрыв ответа.
# Граница слова здесь обязательна: без неё «\right» находится внутри
# «\rightarrow», и любая реакция со стрелкой — а это лучшее, что модель умеет
# восстанавливать, — выглядит как оборванная запись и отбрасывается.
_LEFT_RE = re.compile(r'\\left(?![A-Za-z])')
_RIGHT_RE = re.compile(r'\\right(?![A-Za-z])')


def strip_layout(text: str) -> str:
    """Снимает разметку выравнивания и начертания, оставляя саму запись."""
    text = _WRAPPER_RE.sub(' ', text or '')
    return _STYLE_RE.sub(' ', text)


def structure_score(text: str) -> int:
    """Сколько признаков математической структуры в записи.

    Считается по записи без разметки макета, иначе оценка мерила бы старание
    модели оформить ответ, а не то, что она в нём распознала.
    """
    return len(_STRUCTURE_RE.findall(strip_layout(text)))


def asserts_something(text: str) -> bool:
    """Есть ли в записи знак отношения или буквенное обозначение.

    Буквы ищутся по записи, с которой снята разметка макета и имена команд:
    иначе «array» из ``\\begin{array}{l}`` сойдёт за обозначение, и столбик
    голых дробей пройдёт проверку.
    """
    if _RELATION_RE.search(text or ''):
        return True
    bare = _COMMAND_RE.sub(' ', strip_layout(text))
    return bool(_LETTER_RE.search(bare))


def _bare_content(text: str) -> str:
    """Запись без разметки, пробелов и группирующих скобок — для сравнения."""
    return re.sub(r'[\s{}]', '', strip_layout(text))


def same_content(existing: str, candidate: str) -> bool:
    """Совпадают ли записи по содержанию, различаясь только оформлением.

    Модель часто возвращает ровно то же, что уже стоит в блоке, добавив
    начертание: «Cu(s)» приходит как «\\mathrm{C u(s)}». Замена здесь ничего не
    исправляет, зато переводит блок в разметку и прячет исходник.
    """
    return _bare_content(existing) == _bare_content(candidate)


def _braces_balanced(text: str) -> bool:
    depth = 0
    for index, char in enumerate(text):
        if char in '{}' and index and text[index - 1] == '\\':
            continue          # экранированная скобка, не структурная
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def validate(candidate, cfg=config.DEFAULT) -> tuple:
    """Проверяет, что кандидат вообще годен. Возвращает ``(годен, причина)``."""
    text = (candidate or '').strip()
    if not text:
        return False, 'модель ничего не вернула'
    if len(text) > cfg.max_candidate_chars:
        return False, f'слишком длинный ответ ({len(text)} символов)'
    alien = _ALIEN_RE.search(text)
    if alien:
        return False, f'в ответе посторонние знаки ({alien.group(0)!r})'
    if not _braces_balanced(text):
        return False, 'непарные фигурные скобки'
    if len(_LEFT_RE.findall(text)) != len(_RIGHT_RE.findall(text)):
        return False, 'непарные \\left и \\right'
    if _RUNAWAY_RE.search(text) or _RUNAWAY_TIGHT_RE.search(text):
        return False, 'ответ ушёл в повтор одного куска'
    if not asserts_something(text):
        return False, 'в ответе нет ни обозначений, ни знака отношения'
    return True, ''


def _existing_is_broken(text: str) -> tuple:
    """Признаки того, что текущая запись формулы негодна."""
    if '(cid:' in text:
        return True, 'в исходном битые символы шрифта'
    if textstats.repeat_run_max(text) >= 6:
        return True, 'в исходном длинная цепочка одного символа'
    if textstats.doubling_ratio(text) >= 0.30:
        return True, 'в исходном удвоены буквы'
    if len(text.strip()) > 6 and structure_score(text) == 0:
        return True, 'в исходном нет ни одного признака формулы'
    return False, ''


def _preserve_number(existing: str, candidate: str) -> str:
    """Возвращает номер формулы на место, если модель его потеряла.

    На номер ссылается текст вокруг («подставив (3.2), получим»), и потерять его
    значит разорвать связь между формулой и рассуждением.
    """
    number = select.trailing_number(existing)
    if not number:
        return candidate
    digits = re.sub(r'\D', '', number)
    if digits and digits in re.sub(r'\D', '', candidate):
        return candidate
    return f'{candidate} {number}'


def decide(block: dict, candidate, cfg=config.DEFAULT) -> dict:
    """Сравнивает кандидата с текущим текстом блока.

    Решение — одно из ``replace``, ``keep``, ``unsure``, и к нему всегда идёт
    причина: по ней потом читается отчёт.
    """
    existing = block.get('text') or ''
    valid, why = validate(candidate, cfg)
    if not valid:
        return _verdict('keep', why, existing, candidate)

    candidate = candidate.strip()
    if existing.strip() and same_content(existing, candidate):
        return _verdict('keep', 'кандидат отличается только оформлением',
                        existing, candidate)

    empty_or_number = (not existing.strip()
                       or select.is_bare_number(existing))
    from_text_layer = block.get('text_source') == select.RELIABLE_SOURCE

    if from_text_layer and not empty_or_number and '(cid:' not in existing:
        return _verdict('keep', 'текст собран из символьного слоя PDF',
                        existing, candidate)

    if empty_or_number:
        return _verdict('replace', 'от формулы оставался только номер',
                        existing, _preserve_number(existing, candidate))

    broken, broken_why = _existing_is_broken(existing)
    if broken:
        return _verdict('replace', broken_why,
                        existing, _preserve_number(existing, candidate))

    theirs, ours = structure_score(existing), structure_score(candidate)
    if ours > theirs:
        return _verdict(
            'replace', f'у кандидата больше структуры ({ours} против {theirs})',
            existing, _preserve_number(existing, candidate))

    return _verdict(
        'unsure', f'структура не хуже и не лучше ({ours} против {theirs})',
        existing, _preserve_number(existing, candidate))


def _verdict(decision, reason, existing, candidate) -> dict:
    return {
        'decision': decision,
        'reason': reason,
        'existing': existing,
        'candidate': candidate,
        'structure_existing': structure_score(existing),
        'structure_candidate': structure_score(candidate or ''),
    }


def applies(verdict: dict, cfg=config.DEFAULT) -> bool:
    """Надо ли на самом деле писать этот вариант в файл."""
    if verdict['decision'] == 'replace':
        return True
    return verdict['decision'] == 'unsure' and cfg.accept_unsure
