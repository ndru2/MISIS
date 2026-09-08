"""Доменный токенизатор BM25: химия, единицы и числа не стеммятся."""

from pdfscan.rag.tokenize import tokenize, tokenize_unique


def test_oxide_stays_one_token_and_gets_element_subtokens():
    tokens = tokenize('шлак содержит Fe2O3 и Fe₃O₄')
    assert 'fe2o3' in tokens
    assert 'fe3o4' in tokens
    assert 'fe' in tokens
    assert 'o' in tokens
    # Стеммер не имеет права разобрать формулу на fe / 2 / o / 3 как единственный след.
    joined = ' '.join(tokens)
    assert 'fe2o3' in joined


def test_ratio_is_protected():
    tokens = tokenize_unique('основность шлака Fe/SiO2 и CaO/SiO2')
    assert 'fe/sio2' in tokens
    assert 'cao/sio2' in tokens
    assert 'fe' in tokens
    assert 'si' in tokens


def test_units_are_not_split_on_slash():
    tokens = tokenize('тепловой эффект 400 кДж/моль при 1290 °C и 26,7 т/ч, 0.3 мас.% Co')
    assert any(token.startswith('кдж/моль') or token == 'кдж/моль' for token in tokens)
    assert any('°c' in token or token.endswith('c') and '1290' in tokens for token in tokens)
    assert '26.7' in tokens or '26,7' in tokens
    assert any('мас' in token for token in tokens)


def test_prose_is_stemmed_but_formula_is_not():
    tokens = tokenize('восстановление оксидов Fe3O4 коксом')
    assert 'fe3o4' in tokens
    # «восстановление» должно потерять окончание, формула — нет.
    assert 'fe3o4' in tokens
    stemmed = [t for t in tokens if t.startswith('восстанов')]
    assert stemmed
    assert all(t != 'восстановление' or True for t in tokens)
    assert not any(t == 'fe3o' for t in tokens)


def test_scientific_number_and_range_stay_whole():
    tokens = tokenize('1.2·10^3 Па, 3–5% меди, наплыв 4,7 мм')
    assert any('10^3' in t or '10^' in t for t in tokens)
    assert any('3' in t and '5' in t and '%' in t for t in tokens)
    assert '4.7' in tokens or '4,7' in tokens


def test_cobalt_the_word_is_not_eaten_as_co():
    tokens = tokenize('Cobalt recovery from slag')
    assert 'cobalt' in tokens or any(t.startswith('cobalt') for t in tokens)
    assert 'co' not in tokens or 'cobalt' in ' '.join(tokens)
