from pdfscan.rag import retrieval


class FakeReranker:
    model_name = 'fake-reranker'

    def __init__(self, scores):
        self.scores = scores

    def score(self, query, texts):
        assert query
        assert len(texts) == len(self.scores)
        return self.scores


def test_router_combines_chemistry_table_and_text():
    profile = retrieval.route_query('Cu content in FSF slag Table I')
    assert profile.vector_kinds == ('chemistry', 'table', 'text')
    assert profile.channels[-1] == 'bm25'


def test_router_uses_numeric_range_for_simple_unit():
    profile = retrieval.route_query('температура выше 1200 °C')
    assert profile.vector_kinds == ('unit', 'text')
    assert profile.filters == {
        'numeric': {'canonical': 'degC', 'gte': 1200.0},
        'unit_type': 'quantity',
    }


def test_retrieve_deduplicates_child_and_parent(monkeypatch):
    hits = [
        {'unit_id': 'd#c0#formula', 'unit_type': 'formula', 'parent_chunk_id': 'd#c0',
         'doc_id': 'd', 'pages': [1], 'text': 'FeO + C', 'score': 0.8,
         'formula_flat': 'FeO + C'},
        {'unit_id': 'd#c0', 'unit_type': 'chunk', 'parent_chunk_id': None,
         'doc_id': 'd', 'pages': [1], 'text': 'Объяснение реакции', 'score': 0.7},
        {'unit_id': 'd#c1', 'unit_type': 'chunk', 'parent_chunk_id': None,
         'doc_id': 'd', 'pages': [2], 'text': 'Другой фрагмент', 'score': 0.6},
    ]
    monkeypatch.setattr(retrieval.qdrant_index, 'search', lambda *args, **kwargs: hits)
    monkeypatch.setattr(retrieval.qdrant_index, 'retrieve_units', lambda *args, **kwargs: {
        'd#c0': hits[1],
    })
    profile, evidence = retrieval.retrieve(
        'FeO + C', k=2, relevance_threshold=0.5,
        reranker=FakeReranker([0.92, 0.31]),
    )
    assert [item.unit_id for item in evidence] == ['d#c0#formula']
    assert evidence[0].context == 'Объяснение реакции'
    assert evidence[0].relevance_score == 0.92
    assert profile.candidate_count == 2
    assert profile.accepted_count == 1
    assert profile.reranker == 'fake-reranker'


def test_relevance_gate_returns_no_evidence_when_no_candidate_passes(monkeypatch):
    hits = [{'unit_id': 'd#c0', 'unit_type': 'chunk', 'doc_id': 'd', 'pages': [1],
             'text': 'История производства меди', 'score': 0.8}]
    monkeypatch.setattr(retrieval.qdrant_index, 'search', lambda *args, **kwargs: hits)
    monkeypatch.setattr(retrieval.qdrant_index, 'retrieve_units', lambda *args, **kwargs: {})
    profile, evidence = retrieval.retrieve(
        'Какова молярная масса меди?', relevance_threshold=0.55,
        reranker=FakeReranker([0.12]),
    )
    assert evidence == []
    assert profile.candidate_count == 1
    assert profile.accepted_count == 0
