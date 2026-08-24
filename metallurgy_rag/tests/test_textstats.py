"""Признаки текста и кодирование под символьную модель."""

import pytest

from pdfscan.prepare import textstats


def test_encode_gives_one_byte_per_char():
    """Регресс: кодирование падало на попытке сделать bytes из строки.

    Символьная модель обучается на выходе этой функции, поэтому её поломка
    останавливала всю стадию очистки.
    """
    encoded = textstats.encode('Fe2O3 привет hello')
    assert isinstance(encoded, bytes)
    assert len(encoded) == len('Fe2O3 привет hello')
    assert max(encoded) < textstats.ALPHABET_SIZE


def test_encode_folds_digits_and_punctuation():
    """Цифры и знаки сводятся к служебным символам алфавита."""
    assert textstats.encode('12345') == textstats.encode('99999')
    assert len(textstats.encode('')) == 0


@pytest.mark.parametrize('text, expected', [
    ('металлургия меди', 'cyr'),
    ('copper smelting process', 'lat'),
    ('=+= 123 ===', 'none'),
    # Химия латиницей внутри русской фразы не должна менять письмо: иначе блок
    # оценивался бы английской моделью и выглядел бы шумом.
    ('содержание Fe2O3 в шлаке составляет 12 процентов', 'cyr'),
])
def test_script_detection(text, expected):
    assert textstats.script(text) == expected


def test_norm_key_hides_page_numbers():
    """Колонтитул с разным номером страницы должен давать один ключ."""
    left = textstats.norm_key('Труды конференции, с. 214')
    right = textstats.norm_key('Труды конференции, с. 987')
    assert left == right


def test_doubling_ratio_catches_ocr_stutter():
    assert textstats.doubling_ratio('ммееддьь ннииккеелльь') > 0.5
    assert textstats.doubling_ratio('медь никель кобальт') == 0.0


def test_repeat_run_max():
    assert textstats.repeat_run_max('таблица.........5') >= 9
    assert textstats.repeat_run_max('обычный текст') <= 2
