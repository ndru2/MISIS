from types import SimpleNamespace

import pytest

from pdfscan.rag import graph, qdrant_index, retrieval
from pdfscan.rag.experiments import ALL_COMBINATIONS, summarize


class Reranker:
    model_name = 'test'

    def score(self, query, texts):
        return [0.9] * len(texts)


@pytest.mark.parametrize('methods', ALL_COMBINATIONS)
def test_every_combination_isolated(monkeypatch, methods):
    calls = []

    def flat(query, **kwargs):
        calls.append('flat')
        assert bool(kwargs['vector_kinds']) == ('vector' in methods)
        assert kwargs['use_bm25'] == ('bm25' in methods)
        channels = list(kwargs['vector_kinds']) + (['bm25'] if kwargs['use_bm25'] else [])
        return {name: [{'unit_id': 'flat', 'text': 'Cu slag'}] for name in channels}

    def kg(query, **kwargs):
        calls.append('graph')
        return [{'unit_id': 'kg:1', 'text': 'Cu graph', 'graph': {'path': 'Cu → slag'}}]

    monkeypatch.setattr(qdrant_index, 'search', flat)
    monkeypatch.setattr(graph, 'search', kg)
    monkeypatch.setattr(qdrant_index, 'retrieve_units',
                        lambda *a, **kw: pytest.fail('No parents, no Qdrant call'))
    profile, evidence = retrieval.retrieve('Cu slag', methods=methods, reranker=Reranker())
    assert ('flat' in calls) == bool(set(methods) & {'vector', 'bm25'})
    assert ('graph' in calls) == ('graph' in methods)
    assert profile.methods == methods
    assert profile.accepted_count == len(evidence)
    for item in evidence:
        if item.unit_id.startswith('kg:'):
            assert item.retrieval_channels == ('graph',)
            assert item.structured['graph']['path']
        else:
            assert 'graph' not in item.retrieval_channels


def test_fusion_ranks_and_provenance():
    hits = retrieval.fuse_rankings({
        'text': [{'unit_id': 'a'}, {'unit_id': 'b'}],
        'bm25': [{'unit_id': 'b'}, {'unit_id': 'b'}],
        'graph': [{'unit_id': 'c'}],
    })
    assert [hit['unit_id'] for hit in hits] == ['b', 'a', 'c']
    assert hits[0]['score'] == pytest.approx(1 / 62 + 1 / 61)
    assert hits[0]['retrieval_channels'] == ['text', 'bm25']


@pytest.mark.parametrize('methods', [[], ['unknown']])
def test_invalid_methods(methods):
    with pytest.raises(ValueError):
        retrieval.retrieve('test', methods=methods)


def test_graph_failure_not_silently_downgraded(monkeypatch):
    def fail(*a, **kw):
        raise ConnectionError('Neo4j unavailable')
    monkeypatch.setattr(graph, 'search', fail)
    with pytest.raises(ConnectionError):
        retrieval.retrieve('test', methods=['graph'])


@pytest.mark.parametrize('vector,bm25', [(True, False), (False, True), (True, True)])
def test_qdrant_does_not_load_disabled_dependencies(monkeypatch, vector, bm25):
    calls = []
    class Client:
        def query_points(self, **kwargs):
            calls.append(kwargs['using'])
            return SimpleNamespace(points=[])
    class Vocab:
        def encode(self, tokens):
            assert bm25
            return [1], [1.0]
    def encode(*a, **kw):
        assert vector
        return [[0.1, 0.2]], 2
    monkeypatch.setattr(qdrant_index, '_encode', encode)
    result = qdrant_index.search('шлак', vector_kinds=('text',) if vector else (),
                                  use_bm25=bm25, return_rankings=True,
                                  vocab=Vocab(), client=Client())
    assert len(result) == int(vector) + int(bm25)
    assert ('bm25' in calls) == bm25


def test_summary_does_not_count_errors_as_no_evidence():
    result = summarize([
        {'experiment': 'graph', 'error': 'offline', 'seconds': 1},
        {'experiment': 'graph', 'seconds': 2, 'evidence': [1],
         'profile': {'candidate_count': 4, 'accepted_count': 1}},
    ])['graph']
    assert result['errors'] == 1
    assert result['evidence_found_rate'] == 1
    assert result['mean_judge_coverage'] is None


def test_graph_adapter_preserves_source(monkeypatch):
    fact = {'uid': 's1', 'subject_text': 'Cu', 'predicate': 'in', 'object_text': 'slag',
            'sentence': 'Cu in slag', 'doc_id': 'book', 'page': 4, 'sentence_uid': 'p1'}
    path = {'facts': [fact], 'score': 2, 'nodes': [{'label': 'Cu'}, {'label': 'slag'}],
            'rels': [{'predicate': 'in'}]}
    monkeypatch.setattr(graph.graph_core, 'search', lambda *a, **kw: ([path], [{'id': 'cu'}], []))
    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    driver = SimpleNamespace(session=lambda **kw: Session())
    hit, = graph.search('Cu', driver=driver)
    assert hit['unit_id'] == 'kg:s1'
    assert hit['pages'] == [4]
    assert hit['doc_id'] == 'book'
    assert hit['graph']['sentence_uid'] == 'p1'


def test_experiment_runner_keeps_going_and_records_failures(monkeypatch):
    import io
    import json
    from pdfscan.rag import experiments
    args = SimpleNamespace(retrieval_only=True, methods=None, collection='test', k=3,
                           qdrant_url='http://localhost:6333', relevance_threshold=0.5,
                           provider='ollama', model='test', judge_model=None,
                           coverage_threshold=0.8, max_iterations=2)
    def retrieve(query, **kwargs):
        if 'graph' in kwargs['methods']:
            raise ConnectionError('graph unavailable')
        return retrieval.QueryProfile(query, (), {}, (), accepted_count=0), []
    monkeypatch.setattr(experiments, 'retrieve', retrieve)
    output = io.StringIO()
    records = experiments.run_experiments([{'id': '1', 'question': 'Cu'}],
                                          ALL_COMBINATIONS, args, output)
    assert len(records) == 7
    assert sum('error' in row for row in records) == 4
    assert len([json.loads(line) for line in output.getvalue().splitlines()]) == 7


def test_ui_method_selection():
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/pdfscan/web/app.py').run()
    assert not app.exception
    app.multiselect[0].set_value(['graph']).run()
    assert not app.chat_input[0].disabled
    app.multiselect[0].set_value([]).run()
    assert app.chat_input[0].disabled


def test_shared_candidate_budget_after_fusion(monkeypatch):
    monkeypatch.setattr(qdrant_index, 'search', lambda *a, **kw: {
        'dense_text': [{'unit_id': f'v{i}', 'text': 'vector'} for i in range(40)],
        'bm25': [{'unit_id': f'b{i}', 'text': 'index'} for i in range(40)],
    })
    monkeypatch.setattr(graph, 'search', lambda *a, **kw: [
        {'unit_id': f'kg:{i}', 'text': 'graph'} for i in range(40)])
    profile, _ = retrieval.retrieve('question', methods=retrieval.METHODS, reranker=Reranker())
    assert profile.candidate_count == 40
    assert sum(profile.channel_counts.values()) == 120


def test_reranker_bounds_model_inputs(monkeypatch):
    import sys
    from pdfscan.rag.rerank import CrossEncoderReranker
    seen = {}
    class Encoder:
        def __init__(self, name, **kwargs): seen.update(kwargs)
        def predict(self, pairs, **kwargs):
            seen.update(kwargs)
            return [0.0 for _ in pairs]
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(CrossEncoder=Encoder))
    reranker = CrossEncoderReranker(max_length=512, batch_size=2, device='cpu')
    assert reranker.score('question', ['text']) == [0.5]
    assert seen['max_length'] == 512
    assert seen['batch_size'] == 2
    assert seen['device'] == 'cpu'


def test_ui_provider_switch_resets_model_and_url():
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/pdfscan/web/app.py').run()
    app.selectbox[0].set_value('ollama').run()
    assert not app.exception
    assert app.text_input(key='generator_model').value == 'qwen3:8b'
    assert '11434' in app.text_input(key='generator_url').value
    app.selectbox[0].set_value('openai-compatible').run()
    assert not app.exception
    assert app.text_input(key='generator_model').value == 'nn-tech/MetalGPT-1:featherless-ai'
    assert app.text_input(key='generator_url').value == 'https://router.huggingface.co/v1'
    assert app.slider[0].label == 'Число фрагментов источников'
