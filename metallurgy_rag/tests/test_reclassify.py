"""Пересчёт подтипа формулы после замены текста.

Проверяется главное: трогаются только изменённые блоки, а решение принимается в
том же порядке, что и при разборе, — сначала грамматика уравнения, потом модель.
"""

import json

from pdfscan.formulas import reclassify


def block(index, text, applied=None, block_type='Formula'):
    record = {'block_id': f'doc#{index}', 'order': index, 'type': block_type,
              'text': text, 'layout': None}
    if applied is not None:
        record['formula_ocr'] = {'applied': applied, 'decision': 'replace'}
    return record


def write_blocks(path, blocks):
    with open(path, 'w', encoding='utf-8') as handle:
        for item in blocks:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    return path


def test_only_replaced_blocks_are_reconsidered():
    """Блок, которого замена не коснулась, разметку менять не должен."""
    assert reclassify.text_was_replaced(block(1, 'a', applied=True))
    assert not reclassify.text_was_replaced(block(2, 'a', applied=False))
    assert not reclassify.text_was_replaced(block(3, 'a'))


def test_context_takes_two_before_and_stops_at_a_long_one():
    blocks = [block(0, 'первый'), block(1, 'второй'), block(2, 'формула'),
              block(3, 'после'), block(4, 'ж' * 300), block(5, 'далеко')]
    context = reclassify.context_around(blocks, 2)

    assert 'первый' in context and 'второй' in context
    assert 'после' in context
    assert 'далеко' not in context      # длинный блок обрывает просмотр


def test_chemistry_is_decided_by_the_rule_without_the_model(tmp_path):
    """Уравнение разбирается грамматикой, и модель для него не вызывается."""
    path = write_blocks(tmp_path / 'blocks.jsonl', [
        block(0, 'После окисления получаем:'),
        block(1, r'2Cu_{2}S + 3O_{2} = 2Cu_{2}O + 2SO_{2}', applied=True),
        block(2, 'где Cu — медь'),
    ])

    decided, jobs = reclassify.collect([path], progress=False)
    assert list(decided.values()) == [reclassify.CHEMISTRY_TYPE]
    assert jobs == []


def test_non_chemistry_goes_to_the_model_with_its_context(tmp_path):
    path = write_blocks(tmp_path / 'blocks.jsonl', [
        block(0, 'Скорость осаждения частицы:'),
        block(1, r'u = \frac{(\rho_1 - \rho_2) g d^{2}}{18\eta}', applied=True),
        block(2, 'где d — диаметр частицы'),
    ])

    decided, jobs = reclassify.collect([path], progress=False)
    assert decided == {}
    assert len(jobs) == 1
    assert 'диаметр' in jobs[0]['context']
    assert 'Скорость осаждения' in jobs[0]['context']


def test_new_label_is_written_with_a_trace(tmp_path):
    path = write_blocks(tmp_path / 'blocks.jsonl', [
        block(0, r'2Cu_{2}S + 3O_{2} = 2Cu_{2}O + 2SO_{2}', applied=True),
    ])

    moves = reclassify.write_back(
        [path], {(str(path), 'doc#0'): reclassify.CHEMISTRY_TYPE},
        progress=False)

    written = reclassify.read_blocks(path)[0]
    assert written['type'] == reclassify.CHEMISTRY_TYPE
    assert written['formula_ocr']['type_before'] == 'Formula'
    assert written['formula_ocr']['type_after'] == reclassify.CHEMISTRY_TYPE
    assert sum(moves.values()) == 1


def test_unchanged_label_is_counted_but_not_rewritten(tmp_path):
    path = write_blocks(tmp_path / 'blocks.jsonl', [
        block(0, 'a = b', applied=True, block_type='Formula (math)'),
    ])

    moves = reclassify.write_back(
        [path], {(str(path), 'doc#0'): 'Formula (math)'}, progress=False)

    written = reclassify.read_blocks(path)[0]
    assert 'type_before' not in written['formula_ocr']
    assert all('без изменений' in move for move in moves)
