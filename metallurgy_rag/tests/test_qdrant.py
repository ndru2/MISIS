"""Сборка sparse-вектора и payload для Qdrant — без живого сервера."""

from pdfscan.rag.qdrant_index import SparseVocab, chunk_payload


def test_vocab_assigns_stable_ids_and_term_frequencies():
    vocab = SparseVocab()
    first, values = vocab.encode(['fe2o3', 'шлак', 'fe2o3'])
    assert len(first) == 2
    assert sorted(values) == [1.0, 2.0]
    again, _ = vocab.encode(['fe2o3'])
    assert again == [vocab.token_to_id['fe2o3']]


def test_chunk_payload_exposes_filters():
    payload = chunk_payload({
        'doc_id': 'chikashi',
        'chunk_id': 'chikashi#c0',
        'text': 'Table I slag',
        'text_search': 'Table I slag SiO2',
        'has_table': True,
        'has_formula': False,
        'has_chemistry': True,
        'elements': ['Cu', 'Fe', 'Si'],
        'units': [{'unit': 'мас.%', 'canonical': '%'}],
        'year': 2011,
        'lang': 'en',
        'pages': [2],
        'section_path': ['Operations'],
        'category': 'Cu-Co сырье',
        'doc_type': 'article',
    })
    assert payload['has_table'] is True
    assert payload['year'] == 2011
    assert 'Cu' in payload['elements']
    assert 'мас.%' in payload['units']
    assert payload['category'] == 'Cu-Co сырье'
    assert payload['source'] == 'chunk'


def test_payload_clips_parent_text():
    payload = chunk_payload({
        'parent_text': 'шлак ' * 3000,
        'text': 'строка таблицы',
    })
    assert payload['parent_text'].endswith('…')
    assert len(payload['parent_text']) <= 4001
