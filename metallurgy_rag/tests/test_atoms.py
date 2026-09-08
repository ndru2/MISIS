"""Атомарный чанкер: таблицы и формулы без token-overlap."""

import json
import re
from pathlib import Path

from pdfscan.parse.mineru import convert_document
from pdfscan.rag.atoms import build_atoms, build_atomic_chunks

FIXTURES = Path(__file__).parent / 'fixtures' / 'mineru'


def _chikashi():
    data = json.loads((FIXTURES / 'chikashi_v2.json').read_text(encoding='utf-8'))
    return convert_document(data, 'chikashi')


def _vanyukov():
    data = json.loads((FIXTURES / 'vanyukov_v2.json').read_text(encoding='utf-8'))
    return convert_document(data, 'vanyukov')


def _wide_table_blocks(n_rows=20):
    header = '<tr><td>Refinery</td><td>Cu %</td><td>Current kA</td></tr>'
    rows = ''.join(f'<tr><td>Plant {i}</td><td>{i}.1</td><td>{i}00</td></tr>'
                   for i in range(n_rows))
    html = f'<table>{header}{rows}</table>'
    return convert_document([[
        {
            'type': 'paragraph',
            'content': {'paragraph_content': [
                {'type': 'text', 'content': 'Industrial data are shown in Table 13.5 below.'},
            ]},
            'bbox': [1, 2, 3, 4],
        },
        {
            'type': 'table',
            'content': {
                'table_caption': [{'type': 'text', 'content': 'Table 13.5 Selected industrial copper electrorefining data'}],
                'html': html,
            },
            'bbox': [1, 2, 3, 4],
        },
    ]], 'davenport')


def test_table_and_formula_are_separate_atoms():
    atoms = build_atoms(_chikashi())
    kinds = [a['kind'] for a in atoms]
    assert 'table' in kinds
    assert 'formula' in kinds
    table = next(a for a in atoms if a['kind'] == 'table')
    assert any('Table' in r['type'] for r in table['records'])
    # Ссылка «shown in Table I» уходит внутрь табличного атома, не остаётся прозой.
    assert any('shown in Table I' in (r.get('text') or '') for r in table['records'])


def test_stokes_formula_keeps_lead_in_and_legend():
    atoms = build_atoms(_chikashi())
    stokes = next(
        a for a in atoms if a['kind'] == 'formula'
        and any('[4]' in (r.get('text') or '') for r in a['records'])
    )
    blob = '\n'.join(r.get('text') or '' for r in stokes['records'])
    assert 'Stokes law' in blob
    assert r'\rho' in blob or 'rho' in blob.lower()
    assert 'acceleration due to gravity' in blob
    assert 'diameter of copper droplet' in blob
    assert 'density of copper' in blob


def test_consecutive_reactions_form_one_atom():
    atoms = build_atoms(_chikashi())
    reactions = next(
        a for a in atoms if a['kind'] == 'formula'
        and any('[1]' in (r.get('text') or '') for r in a['records'])
    )
    texts = [r.get('text') or '' for r in reactions['records']]
    assert any('[1]' in t for t in texts)
    assert any('[2]' in t for t in texts)
    assert any('coke is charged' in t.lower() for t in texts)


def test_no_token_overlap_between_table_parts():
    chunks = build_atomic_chunks(_wide_table_blocks(), model_name=None, max_tokens=80)
    table_chunks = [c for c in chunks if c.get('has_table')]
    assert len(table_chunks) > 1
    # Каждая часть повторяет шапку, но строки данных не пересекаются.
    plants = []
    for chunk in table_chunks:
        found = set(re.findall(r'Plant \d+', chunk['text']))
        plants.append(found)
        assert found
        assert 'Table 13.5' in chunk['text'] or 'Refinery' in chunk['text']
        assert chunk['parent_text']
        assert 'Plant 0' in chunk['parent_text']
    for left, right in zip(plants, plants[1:]):
        assert left.isdisjoint(right)


def test_formula_chunk_does_not_copy_the_equation_into_the_next_chunk():
    chunks = build_atomic_chunks(_chikashi(), model_name=None, max_tokens=80)
    with_stokes = [c for c in chunks if '[4]' in c['text']]
    assert len(with_stokes) == 1
    after = [c for c in chunks if c['order'] > with_stokes[0]['order']]
    assert all('[4]' not in c['text'] for c in after)


def test_heading_resets_overlap():
    chunks = build_atomic_chunks(_chikashi(), model_name=None, max_tokens=120)
    chemistry = [c for c in chunks if 'Carbothermic' in (c.get('section') or '')
                 or 'Carbothermic' in ' '.join(c.get('section_path') or [])]
    assert chemistry
    # Куски после нового раздела не тащат хвост предыдущей прозы про Table I.
    assert all('Table I: Typical' not in c['text'] for c in chemistry)


def test_vanyukov_table_stays_one_chunk_when_it_fits():
    chunks = build_atomic_chunks(_vanyukov(), model_name=None, max_tokens=400)
    tables = [c for c in chunks if c.get('has_table')]
    assert len(tables) == 1
    assert '26,7' in tables[0]['text'] or '26,7' in tables[0]['text_search']
    assert 'т/ч' in tables[0]['text'] or 'т/ч' in (tables[0].get('text_search') or '')
    formula = [c for c in chunks if c.get('has_formula')]
    assert formula
    assert any('заверш' in c['text'] for c in formula)


def test_figure_caption_with_equals_does_not_loop():
    records = [
        {'doc_id': 'x', 'block_id': 'x#0', 'page': 1, 'type': 'NarrativeText',
         'text': 'Шлак сливают в ковш.', 'table_html': None, 'reliable': True},
        {'doc_id': 'x', 'block_id': 'x#1', 'page': 1, 'type': 'FigureCaption',
         'text': 'Cu = 19.46 мас.%', 'table_html': None, 'reliable': True},
        {'doc_id': 'x', 'block_id': 'x#2', 'page': 1, 'type': 'NarrativeText',
         'text': 'Далее металл разливают.', 'table_html': None, 'reliable': True},
    ]
    atoms = build_atoms(records)
    kinds = [a['kind'] for a in atoms]
    assert 'figure' in kinds
    assert kinds.count('figure') == 1
    chunks = build_atomic_chunks(records, model_name=None, max_tokens=200)
    assert chunks


def test_page_numbers_are_not_indexed():
    chunks = build_atomic_chunks(_chikashi(), model_name=None, max_tokens=200)
    blob = '\n'.join(c['text'] for c in chunks)
    assert '\n185\n' not in f'\n{blob}\n'
    # Номер страницы из колонтитула не должен стать отдельным куском.
    assert all(c['text'].strip() != '185' for c in chunks)
