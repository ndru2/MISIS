"""Нормализация блоков перед индексацией.

Главное проверяемое свойство: запись формулы должна становиться находимой, но
обычный текст трогать нельзя. Односимвольные русские слова — «в», «и», «о» —
делают склейку разряженных букв опасной за пределами формул.
"""

from pdfscan.rag.normalize import (clean_text, flatten_formula,
                                   join_spaced_letters, normalize_record)


def test_spaced_letters_are_joined_into_words():
    """Модель формул разряжает буквы: без склейки слово для поиска не найти."""
    spaced = r'\mathrm{W e i g h t\;l o s s\;a t\;a n y\;t i m e}'
    assert join_spaced_letters(spaced) == r'\mathrm{Weight loss at any time}'
    assert flatten_formula(spaced) == 'Weight loss at any time'


def test_word_boundary_survives_the_join():
    """Отбивка между словами обязана стать пробелом, а не исчезнуть."""
    joined = join_spaced_letters(r'\mathrm{T h e o r e t i c a l\;w e i g h t}')
    assert joined == r'\mathrm{Theoretical weight}'
    assert 'Theoreticalweight' not in joined


def test_separators_other_than_spacing_commands_also_hold():
    """Перевод строки и разделитель колонок — тоже граница слова."""
    assert join_spaced_letters(r'M S&M O') == 'MS MO'
    assert join_spaced_letters(r'C a O\\F e O') == 'CaO FeO'


def test_upright_and_italic_letters_are_joined():
    """«R T» — это RT, множитель в формуле, а не две переменные подряд."""
    text = r'\Delta\mathrm{G}^{\circ}=-2.303\;R T\log K'
    assert 'RT' in join_spaced_letters(text)
    assert 'RT' in flatten_formula(text)


def test_normal_formulas_are_left_alone():
    formulas = (r'\frac{a}{b} = c', 'H_{2}O + Fe_{2}O_{3}',
                r'\mathrm{Cu^{2+}(aq)}')
    for formula in formulas:
        assert join_spaced_letters(formula) == formula


def test_prose_is_not_touched_by_the_join():
    """В прозе «в и о том» склейка превратила бы текст в бессмыслицу."""
    prose = 'дошёл до и о том, что в и о нём известно'
    assert clean_text(prose) == prose

    record = normalize_record({'text': prose, 'type': 'NarrativeText'})
    assert record['text_search'] == prose


def test_formula_block_gets_a_searchable_string():
    record = normalize_record({
        'text': r'\mathrm{W e i g h t\;l o s s}',
        'type': 'Formula (math)',
    })
    assert 'Weight loss' in record['text_search']
