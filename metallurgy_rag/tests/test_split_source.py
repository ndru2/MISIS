"""Раскладка corpus_split → блоки для чанкера и графа."""

from pathlib import Path

from pdfscan.rag.split_source import as_index_record, load_document


def test_heading_and_chemistry_keep_types_the_chunker_knows():
    heading = as_index_record({
        'doc_id': 'd', 'order': 0, 'page': 1, 'topic': 'медь',
        'subtype': 'heading', 'text': 'Плавка', 'section': 'Плавка',
    }, 'text')
    assert heading['type'] == 'Title'
    assert heading['text_level'] == 1
    assert heading['block_id'] == 'd#0'

    formula = as_index_record({
        'doc_id': 'd', 'order': 3, 'page': 2, 'topic': 'медь',
        'latex': r'\mathrm{Fe}_{3}\mathrm{O}_{4} + C',
        'formula_class': 'chemistry', 'number': '1',
    }, 'formulas_chem')
    assert formula['type'] == 'Formula (chemistry)'
    assert formula['formula_latex'].startswith(r'\mathrm{Fe}')
    assert formula['category'] == 'медь'

    math = as_index_record({
        'doc_id': 'd', 'order': 4, 'page': 2,
        'latex': r'u = \frac{g d^{2}}{18}', 'formula_class': 'physics',
    }, 'formulas_math')
    assert math['type'] == 'Formula'


def test_nomenclature_is_marked_as_legend():
    record = as_index_record({
        'doc_id': 'd', 'order': 5, 'page': 2,
        'subtype': 'nomenclature', 'text': 'g — ускорение свободного падения',
    }, 'text')
    assert record['is_legend'] is True
    assert record['type'] == 'NarrativeText'


def test_table_keeps_html_and_caption():
    record = as_index_record({
        'doc_id': 'd', 'order': 1, 'page': 1,
        'caption': 'Таблица 1', 'footnote': 'мас.%',
        'table_html': '<table><tr><td>Cu</td></tr></table>',
    }, 'tables')
    assert record['type'] == 'Table'
    assert 'Cu' in record['table_html']
    assert record['caption'] == 'Таблица 1'


def test_load_document_restores_reading_order(tmp_path: Path):
    split = tmp_path
    (split / 'text').mkdir()
    (split / 'formulas_chem').mkdir()
    (split / 'formulas_math').mkdir()
    (split / 'tables').mkdir()
    (split / 'text' / 'doc.jsonl').write_text(
        '{"doc_id":"doc","order":0,"page":0,"subtype":"heading","text":"A","topic":"t"}\n'
        '{"doc_id":"doc","order":2,"page":0,"subtype":"text","text":"после","topic":"t"}\n',
        encoding='utf-8')
    (split / 'formulas_chem' / 'doc.jsonl').write_text(
        '{"doc_id":"doc","order":1,"page":0,"latex":"FeO + C","formula_class":"chemistry"}\n',
        encoding='utf-8')

    records = load_document(split, 'doc')
    assert [r['type'] for r in records] == ['Title', 'Formula (chemistry)', 'NarrativeText']
    assert [r['text'] for r in records] == ['A', 'FeO + C', 'после']
