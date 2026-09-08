"""Адаптер MinerU: типы блоков, inline-формулы, таблицы, разрядка LaTeX."""

import json
from pathlib import Path

from pdfscan.parse.mineru import (convert_document, load_mineru_file,
                                  repair_mineru_latex)

FIXTURES = Path(__file__).parent / 'fixtures' / 'mineru'


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding='utf-8'))


def test_repair_collapses_spaced_temperature():
    repaired = repair_mineru_latex(r'1 2 9 0 ^ { \mathrm { o } } \mathrm { C }')
    assert '1290' in repaired
    assert 'circ' in repaired
    assert 'C' in repaired
    assert '1 2 9 0' not in repaired


def test_repair_collapses_oxide_indices():
    repaired = repair_mineru_latex(
        r'(\mathrm{Fe} _ {3} \mathrm{O} _ {4}) + \mathrm{C} = (3 \mathrm{FeO})')
    assert r'{3}' in repaired
    assert r'{4}' in repaired
    assert r'_ {3}' not in repaired


def test_repair_collapses_subscript_letters_but_keeps_legend_words():
    stokes = repair_mineru_latex(
        r'u = \frac {(\rho_ {c u} - \rho_ {s l a g})}{1 8 \eta_ {s l a g}} g d ^ {2}')
    assert r'\rho_{cu}' in stokes.replace(' ', '')
    assert '18' in stokes.replace(' ', '')

    legend = repair_mineru_latex(
        r'g = \text { acceleration   due   to   gravity,   m   s } ^{-2}')
    assert 'acceleration due to gravity' in legend


def test_chikashi_maps_types_and_keeps_inline_math_in_the_paragraph():
    records = convert_document(_load('chikashi_v2.json'), 'chikashi')
    types = [r['type'] for r in records]

    assert 'Title' in types
    assert 'Table' in types
    assert 'Header' in types
    assert 'PageNumber' in types
    assert any(t.startswith('Formula') for t in types)

    prose = [r for r in records if r['type'] == 'NarrativeText']
    tapped = next(r for r in prose if 'Slag is tapped' in r['text'])
    assert '$' in tapped['text']
    assert '1290' in tapped['text']
    assert tapped['spans']
    assert any(span['type'] == 'formula' for span in tapped['spans'])


def test_table_is_one_block_with_caption_and_html():
    records = convert_document(_load('chikashi_v2.json'), 'chikashi')
    tables = [r for r in records if r['type'] == 'Table']
    assert len(tables) == 1
    table = tables[0]
    assert 'Table I' in table['caption']
    assert 'SiO' in table['table_html']
    assert '19.46' in table['table_html']
    assert table['page'] == 2


def test_reactions_are_chemistry_and_legends_are_not():
    records = convert_document(_load('chikashi_v2.json'), 'chikashi')
    formulas = [r for r in records if str(r['type']).startswith('Formula')]
    reactions = [r for r in formulas if '[1]' in r['text'] or '[2]' in r['text']]
    legends = [r for r in formulas if r['is_legend']]

    assert reactions
    assert all(r['type'] == 'Formula (chemistry)' for r in reactions)
    assert legends
    assert all(r['type'] == 'Formula' for r in legends)
    assert all('acceleration' in r['text'] or 'diameter' in r['text']
               for r in legends)


def test_image_pixels_are_dropped_caption_stays():
    data = [[
        {
            'type': 'chart',
            'content': {
                'content': '100.00 + 90.00 + Jul-09',
                'chart_caption': [{'type': 'text', 'content': 'Figure 1: Control of copper in SCF slag'}],
                'chart_footnote': [],
            },
            'bbox': [1, 2, 3, 4],
        },
        {
            'type': 'image',
            'content': {'content': '', 'image_caption': [], 'image_footnote': []},
            'bbox': [1, 2, 3, 4],
        },
    ]]
    records = convert_document(data, 'doc')
    assert len(records) == 1
    assert records[0]['type'] == 'FigureCaption'
    assert 'Figure 1' in records[0]['text']
    assert 'Jul-09' not in records[0]['text']


def test_empty_paragraphs_are_dropped():
    data = [[{'type': 'paragraph', 'content': {'paragraph_content': [{'type': 'text', 'content': '   '}]}, 'bbox': [0, 0, 1, 1]}]]
    assert convert_document(data, 'doc') == []


def test_vanyukov_russian_table_and_formula():
    records = convert_document(_load('vanyukov_v2.json'), 'vanyukov')
    assert any(r['type'] == 'Title' and 'Ванюкова' in r['text'] for r in records)
    table = next(r for r in records if r['type'] == 'Table')
    assert 'т/ч' in table['table_html']
    assert '26,7' in table['table_html']
    formula = next(r for r in records if str(r['type']).startswith('Formula'))
    assert 'Me' in formula['text'].replace(' ', '') or 'M e' in formula['text']
    assert any(r['type'] == 'Footer' for r in records)


def test_v1_fallback_reads_flat_list():
    v1 = [
        {'type': 'text', 'text': 'Introduction', 'text_level': 1, 'bbox': [1, 2, 3, 4], 'page_idx': 0},
        {'type': 'equation', 'text': '$$\\mathrm{Fe} _ {2} \\mathrm{O} _ {3}$$',
         'text_format': 'latex', 'bbox': [1, 2, 3, 4], 'page_idx': 1},
        {'type': 'table', 'table_caption': ['Table 1'], 'table_footnote': [],
         'table_body': '<table><tr><td>Cu</td></tr></table>', 'bbox': [1, 2, 3, 4], 'page_idx': 1},
    ]
    records = convert_document(v1, 'flat')
    assert records[0]['type'] == 'Title'
    assert records[0]['page'] == 1
    assert any(r['type'].startswith('Formula') and '{2}' in r['text'] for r in records)
    assert any(r['type'] == 'Table' and r['caption'] == 'Table 1' for r in records)


def test_load_mineru_file_roundtrip(tmp_path):
    src = FIXTURES / 'chikashi_v2.json'
    records = load_mineru_file(src, doc_id='chikashi')
    assert records[0]['doc_id'] == 'chikashi'
    assert records[0]['block_id'] == 'chikashi#0'
    assert records[0]['parser'] == 'mineru'
