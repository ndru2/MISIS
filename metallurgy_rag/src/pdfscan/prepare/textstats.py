"""Признаки текста, по которым отличают язык от распознавательного шума.

Словаря здесь нет намеренно. Корпус двуязычный, металлургический и с химией,
поэтому любой внешний словарь отвергал бы половину законных слов — «файнштейн»,
«фаялит», «ISASMELT». Вместо принадлежности словарю считается правдоподобие
самой последовательности символов: у распознанной картинки статистика ломается
целиком, а не в отдельных словах.
"""

import hashlib
import re
import unicodedata

CYRILLIC_VOWELS = frozenset('аеёиоуыэюя')
LATIN_VOWELS = frozenset('aeiouy')

_CYRILLIC_RE = re.compile(r'[а-яёА-ЯЁ]')
_LATIN_RE = re.compile(r'[a-zA-Z]')
_WORD_RE = re.compile(r'[^\W\d_]+', re.UNICODE)
_DIGIT_RE = re.compile(r'\d')
_NON_KEY_RE = re.compile(r'[^\w\s#]', re.UNICODE)
_SPACE_RE = re.compile(r'\s+')

# Алфавит символьной модели. Точная буква важна только внутри своего письма,
# поэтому цифры, пробелы и всё остальное сворачиваются в три класса: иначе
# таблица триграмм разрастается на порядок без пользы для оценки.
_BOUNDARY = '^'
_ALPHABET = (_BOUNDARY
             + 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
             + 'abcdefghijklmnopqrstuvwxyz'
             + '# .')
ALPHABET_SIZE = len(_ALPHABET)
BOUNDARY_INDEX = 0
OTHER_INDEX = _ALPHABET.index('.')


class _CharTable(dict):
    """Таблица для ``str.translate``, сводящая любой символ к классу алфавита."""

    def __missing__(self, code):
        return OTHER_INDEX


def _build_table() -> _CharTable:
    table = _CharTable()
    for index, char in enumerate(_ALPHABET):
        table[ord(char)] = index
    for char in '0123456789':
        table[ord(char)] = _ALPHABET.index('#')
    for char in ' \t\n\r\f\v\u00a0':
        table[ord(char)] = _ALPHABET.index(' ')
    return table


_CHAR_TABLE = _build_table()


def encode(text: str) -> bytes:
    """Переводит текст в последовательность индексов алфавита модели.

    ``translate`` отдаёт строку, символы которой имеют номера от нуля до размера
    алфавита, поэтому latin-1 переводит её в байты один к одному. Побайтовый
    проход по тексту здесь был бы на порядок медленнее: корпус — сорок миллионов
    символов.
    """
    lowered = unicodedata.normalize('NFKC', text or '').casefold()
    return lowered.translate(_CHAR_TABLE).encode('latin-1')


def script(text: str) -> str:
    """Определяет письмо текста по преобладанию букв.

    Химическая запись латиницей встречается в русском тексте постоянно, поэтому
    решает не наличие букв, а их количество.
    """
    cyrillic = len(_CYRILLIC_RE.findall(text or ''))
    latin = len(_LATIN_RE.findall(text or ''))
    if cyrillic + latin < 3:
        return 'none'
    return 'cyr' if cyrillic >= latin else 'lat'


def norm_key(text: str) -> str:
    """Ключ для сравнения строк на повторяемость.

    Номер страницы в колонтитуле меняется, сам колонтитул — нет, поэтому цифры
    сводятся к одному символу. Регистр и знаки убираются: распознавание путает
    их чаще всего.
    """
    text = unicodedata.normalize('NFKC', text or '').casefold()
    text = _DIGIT_RE.sub('#', text)
    text = _NON_KEY_RE.sub(' ', text)
    return _SPACE_RE.sub(' ', text).strip()


def content_hash(text: str) -> str:
    """Отпечаток содержимого для поиска точных дублей."""
    return hashlib.blake2b(norm_key(text).encode('utf-8'), digest_size=16).hexdigest()


def doubling_ratio(text: str) -> float:
    """Доля соседних одинаковых букв.

    Распознавание сдвоенных начертаний даёт «EEDDIITTEEDD BBYY»: каждая буква
    удвоена, и доля совпадающих пар подскакивает с обычных двух процентов до
    половины. Ни один язык так не выглядит.
    """
    letters = ''.join(_WORD_RE.findall((text or '').casefold()))
    if len(letters) < 8:
        return 0.0
    same = sum(1 for a, b in zip(letters, letters[1:]) if a == b)
    return same / (len(letters) - 1)


def vowel_ratio(text: str, text_script: str) -> float:
    """Доля гласных среди букв.

    В русском и английском она держится около 0.4. Шум распознавания даёт либо
    почти одни согласные, либо цепочки одинаковых гласных.
    """
    letters = ''.join(_WORD_RE.findall((text or '').casefold()))
    if not letters:
        return 0.0
    vowels = CYRILLIC_VOWELS if text_script == 'cyr' else LATIN_VOWELS
    return sum(1 for char in letters if char in vowels) / len(letters)


def consonant_run_max(text: str, text_script: str) -> int:
    """Самая длинная цепочка согласных внутри слова.

    «Взгляд» даёт четыре, «вспыхнуть» — четыре, дальше начинается шум.
    """
    vowels = CYRILLIC_VOWELS if text_script == 'cyr' else LATIN_VOWELS
    longest = 0
    for word in _WORD_RE.findall((text or '').casefold()):
        run = 0
        for char in word:
            if char in vowels:
                run = 0
            else:
                run += 1
                longest = max(longest, run)
    return longest


def short_token_ratio(text: str) -> float:
    """Доля токенов в один символ.

    Когда распознавание рассыпает строку, вместо слов остаются отдельные буквы
    и палки: «Ц | | | | б о».
    """
    tokens = (text or '').split()
    if len(tokens) < 5:
        return 0.0
    return sum(1 for token in tokens if len(token) == 1) / len(tokens)


def nonalpha_ratio(text: str) -> float:
    """Доля символов, не являющихся буквой, цифрой или пробелом."""
    text = text or ''
    if not text:
        return 0.0
    other = sum(1 for char in text
                if not (char.isalnum() or char.isspace()))
    return other / len(text)


def repeat_run_max(text: str) -> int:
    """Самая длинная цепочка одного и того же символа.

    Рамки таблиц и линейки на сканах превращаются в «_____» и «ЦЦЦЦ».
    """
    longest, run, previous = 0, 0, None
    for char in (text or ''):
        if char.isspace():
            run, previous = 0, None
            continue
        if char == previous:
            run += 1
        else:
            run, previous = 1, char
        longest = max(longest, run)
    return longest


def signals(text: str) -> dict:
    """Считает все структурные признаки текста разом."""
    text_script = script(text)
    return {
        'script': text_script,
        'n_chars': len(text or ''),
        'doubling_ratio': doubling_ratio(text),
        'vowel_ratio': vowel_ratio(text, text_script),
        'consonant_run_max': consonant_run_max(text, text_script),
        'short_token_ratio': short_token_ratio(text),
        'nonalpha_ratio': nonalpha_ratio(text),
        'repeat_run_max': repeat_run_max(text),
    }
