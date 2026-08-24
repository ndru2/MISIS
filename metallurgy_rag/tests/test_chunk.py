"""Границы кусков: бюджет и формулы.

Главное здесь — ни один кусок не должен вылезать за бюджет. Векторизатор
обрезает слишком длинный вход молча, поэтому такой кусок попадает в индекс
изувеченным, и по отчётам это никак не видно.
"""

from pdfscan.rag import chunk

BUDGET = 40

# Тот же счётчик, которым пользуется сборка без имени модели: мерить куски
# другой линейкой, чем та, по которой их резали, значит ничего не проверить.
COUNTER = chunk._Tokenizer(None)


def block(index, text, block_type='NarrativeText'):
    return {'doc_id': 'док', 'block_id': f'док#{index}', 'order': index,
            'page': 1, 'type': block_type, 'text': text, 'reliable': True,
            'table_html': None}


def build(records):
    return chunk.build_chunks(records, model_name=None, max_tokens=BUDGET)


def sizes(chunks):
    return [COUNTER.count(item['text_search']) for item in chunks]


def sentences(count, words=10):
    return ' '.join(f'{"слово " * words}номер {i}.' for i in range(count))


def test_block_longer_than_the_budget_is_split():
    """Распознанная страница приходит одним блоком, и её надо разрезать."""
    chunks = build([block(0, sentences(20))])

    assert len(chunks) > 1
    assert max(sizes(chunks)) <= BUDGET


def test_nothing_is_lost_when_a_long_block_is_split():
    text = sentences(12)
    parts = chunk.split_long(text, BUDGET, COUNTER)

    assert len(parts) > 1
    joined = ' '.join(parts).split()
    assert joined == text.split()


def test_sentence_without_punctuation_is_split_by_words():
    """Распознавание не поставило ни одной точки — резать больше нечем."""
    text = 'слово ' * 300
    parts = chunk.split_long(text, BUDGET, COUNTER)

    assert len(parts) > 1
    assert max(COUNTER.count(part) for part in parts) <= BUDGET


def test_a_run_of_formulas_does_not_grow_without_a_limit():
    """Раньше граница перед формулой не ставилась вовсе, и кусок разрастался."""
    records = [block(0, 'Реакции окисления сульфидов идут так:')]
    for index in range(1, 40):
        records.append(block(index, f'2Cu_{{2}}S + {index}O_{{2}} = 2Cu_{{2}}O',
                             'Formula (chemistry)'))

    chunks = build(records)
    assert len(chunks) > 1
    assert max(sizes(chunks)) <= BUDGET


def test_formula_keeps_the_phrase_that_introduces_it():
    records = [block(0, sentences(3)),
               block(1, 'Отсюда следует, что равновесие сдвигается:'),
               block(2, r'\lg K = -\frac{\Delta G}{RT}', 'Formula (physics)')]

    chunks = build(records)
    with_formula = [item for item in chunks if item['has_formula']]

    assert len(with_formula) == 1
    assert 'равновесие сдвигается' in with_formula[0]['text']


def test_lead_in_does_not_blow_the_budget():
    """Абзац на весь бюджет переносится к формуле не целиком, а хвостом."""
    records = [block(0, sentences(3)),
               block(1, sentences(10)),
               block(2, r'\lg K = -\frac{\Delta G}{RT}', 'Formula (physics)')]

    chunks = build(records)
    assert max(sizes(chunks)) <= BUDGET

    with_formula = [item for item in chunks if item['has_formula']]
    assert len(with_formula) == 1
    # Что-то от вводного абзаца при формуле всё же осталось.
    assert 'слово' in with_formula[0]['text']


def test_short_document_stays_one_chunk():
    chunks = build([block(0, 'Плавка идёт в печи Ванюкова.'),
                    block(1, 'Шлак сливают в ковш.')])
    assert len(chunks) == 1
    assert 'Ванюкова' in chunks[0]['text'] and 'ковш' in chunks[0]['text']
