"""Контракт parent-child search units без Qdrant и embedding-моделей."""

from pdfscan.rag.units import build_search_units


def test_formula_table_and_quantity_are_children_of_chunks():
    records = [
        {'doc_id': 'd', 'block_id': 'd#0', 'page': 1, 'type': 'Title',
         'text': 'Шлаки', 'reliable': True},
        {'doc_id': 'd', 'block_id': 'd#1', 'page': 1, 'type': 'NarrativeText',
         'text': 'Энергия реакции составляет 120 кДж/моль.', 'reliable': True},
        {'doc_id': 'd', 'block_id': 'd#2', 'page': 1, 'type': 'Formula (chemistry)',
         'text': r'FeO + C = Fe + CO \tag{5.2}', 'formula_latex': r'FeO + C = Fe + CO \tag{5.2}',
         'formula_class': 'chemistry', 'reliable': True},
        {'doc_id': 'd', 'block_id': 'd#3', 'page': 2, 'type': 'Table',
         'text': 'Таблица 1', 'caption': 'Таблица 1 Состав шлака', 'reliable': True,
         'table_html': '<table><tr><td>Шлак</td><td>Cu, мас.%</td></tr><tr><td>FSF</td><td>19.46</td></tr></table>'},
    ]
    units = build_search_units(records, max_tokens=200)
    by_type = {}
    for unit in units:
        by_type.setdefault(unit['unit_type'], []).append(unit)

    assert by_type['chunk']
    formula = by_type['formula'][0]
    assert formula['vector_kind'] == 'chemistry'
    assert formula['equation_number'] == '5.2'
    assert formula['parent_chunk_id']

    row = by_type['table_row'][0]
    cell = next(item for item in by_type['table_cell'] if item['cell_value'] == '19.46')
    assert row['vector_kind'] == cell['vector_kind'] == 'table'
    assert 'Cu, мас.%: 19.46' in row['text_search']
    assert cell['row_id'] == row['row_id']
    assert cell['parent_chunk_id']

    quantity = by_type['quantity'][0]
    assert quantity['vector_kind'] == 'unit'
    assert quantity['value'] == 120.0
    assert quantity['unit_raw'] == 'кДж/моль'
