from fastapi.testclient import TestClient

from pdfscan.rag.answering import RagAnswer
from pdfscan.rag.retrieval import Evidence, QueryProfile
from pdfscan.web.api import ApiSettings, create_app


class FakeLLM:
    model = 'fake-qwen'

    def generate(self, messages, *, temperature=0.1):
        return 'Ответ [E1]'


def test_health_exposes_server_configuration():
    app = create_app(ApiSettings(collection='test_collection',
                                 model='test-model'))
    response = TestClient(app).get('/health')
    assert response.status_code == 200
    body = response.json()
    assert body['qdrant_collection'] == 'test_collection'
    assert 'coverage_threshold' in body
    # judge не задан отдельно -> совпадает с генератором
    assert body['judge_model'] == 'test-model'
    assert body['judge_provider'] == 'ollama'


def test_separate_judge_has_own_timeout_and_provider(monkeypatch):
    import pdfscan.web.api as api

    settings = ApiSettings(
        provider='ollama', model='qwen3:8b', timeout_seconds=90,
        judge_provider='openai-compatible', judge_model='strong-judge',
        judge_base_url='https://judge.example/v1', judge_api_key='test-key',
        judge_timeout_seconds=240,
    )
    calls = []
    monkeypatch.setattr(api, 'create_llm', lambda **kwargs: calls.append(kwargs) or FakeLLM())
    app = create_app(settings)
    # Exercise the nested factory through the request path with no evidence.
    no_evidence = RagAnswer(
        'вопрос', 'нет документов', (), QueryProfile('вопрос', (), {}, ()), (),
        'fake-qwen', generation_mode='no_evidence')
    monkeypatch.setattr(api, 'answer_question', lambda *args, **kwargs: no_evidence)
    response = TestClient(app).post('/api/answer', json={'question': 'вопрос'})
    assert response.status_code == 200
    assert calls[1]['provider'] == 'openai-compatible'
    assert calls[1]['model'] == 'strong-judge'
    assert calls[1]['timeout_seconds'] == 240


def test_kg_openrouter_variables_configure_independent_judge(monkeypatch):
    import pdfscan.web.api as api

    monkeypatch.delenv('RAG_JUDGE_PROVIDER', raising=False)
    monkeypatch.delenv('RAG_JUDGE_MODEL', raising=False)
    monkeypatch.delenv('RAG_JUDGE_BASE_URL', raising=False)
    monkeypatch.delenv('RAG_JUDGE_API_KEY', raising=False)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-openrouter-key')
    monkeypatch.setenv('OPENROUTER_MODEL', 'qwen/qwen3.6-35b-a3b')
    settings = api.ApiSettings.from_env()
    assert settings.provider == 'ollama'
    assert settings.model == 'qwen3:8b'
    assert settings.judge_provider == 'openai-compatible'
    assert settings.judge_model == 'qwen/qwen3.6-35b-a3b'
    assert settings.judge_base_url == 'https://openrouter.ai/api/v1'
    assert settings.judge_api_key == 'test-openrouter-key'


def test_request_can_choose_configured_judge_model(monkeypatch):
    import pdfscan.web.api as api

    settings = ApiSettings(
        judge_provider='openai-compatible',
        judge_model='default-judge', judge_base_url='https://judge.example/v1',
        judge_api_key='test-key',
    )
    calls = []
    monkeypatch.setattr(api, 'create_llm', lambda **kwargs: calls.append(kwargs) or FakeLLM())
    no_evidence = RagAnswer(
        'вопрос', 'нет документов', (), QueryProfile('вопрос', (), {}, ()), (),
        'fake-qwen', generation_mode='no_evidence')
    monkeypatch.setattr(api, 'answer_question', lambda *args, **kwargs: no_evidence)
    response = TestClient(create_app(settings)).post(
        '/api/answer', json={'question': 'вопрос', 'judge_model': 'chosen-judge'})
    assert response.status_code == 200
    assert calls[1]['model'] == 'chosen-judge'
    assert calls[1]['api_key'] == 'test-key'


def test_answer_returns_evidence(monkeypatch):
    import pdfscan.web.api as api

    profile = QueryProfile('вопрос', ('text',), {}, ('text', 'bm25'))
    evidence = Evidence('u-1', 'chunk', 'Источник', 'Контекст', None, 'book',
                        (12,), 0.9, ('text', 'bm25'), {})
    answer = RagAnswer(
        'вопрос', 'Ответ [E1]', ('u-1',), profile, (evidence,), 'fake-qwen',
        generation_mode='hybrid', judge_score=0.9, evidence_used=1,
        iterations=1)
    monkeypatch.setattr(api, 'create_llm', lambda **kwargs: FakeLLM())
    monkeypatch.setattr(api, 'answer_question', lambda *args, **kwargs: answer)
    response = TestClient(create_app()).post(
        '/api/answer', json={'question': 'вопрос'})
    assert response.status_code == 200
    payload = response.json()
    assert payload['answer'] == 'Ответ [E1]'
    assert payload['evidence'][0]['unit_id'] == 'u-1'
    assert payload['generation_mode'] == 'hybrid'
    assert payload['judge_score'] == 0.9


def test_answer_reports_no_evidence_without_fallback(monkeypatch):
    import pdfscan.web.api as api

    profile = QueryProfile('вопрос', ('text',), {}, ('text', 'bm25'))
    no_evidence = RagAnswer(
        'вопрос', 'нет документов', (), profile, (), 'fake-qwen',
        generation_mode='no_evidence')
    monkeypatch.setattr(api, 'create_llm', lambda **kwargs: FakeLLM())
    monkeypatch.setattr(
        api, 'answer_question', lambda *args, **kwargs: no_evidence)
    fallback_calls = []
    monkeypatch.setattr(
        api, 'generate_llm_only_answer',
        lambda *args, **kwargs: fallback_calls.append(1))
    response = TestClient(create_app()).post(
        '/api/answer', json={'question': 'вопрос', 'allow_llm_fallback': False})
    assert response.status_code == 200
    assert response.json()['generation_mode'] == 'no_evidence'
    assert fallback_calls == []  # без opt-in fallback не вызывается


def test_answer_uses_explicit_fallback_when_requested(monkeypatch):
    import pdfscan.web.api as api

    profile = QueryProfile('вопрос', ('text',), {}, ('text', 'bm25'))
    no_evidence = RagAnswer(
        'вопрос', 'нет документов', (), profile, (), 'fake-qwen',
        generation_mode='no_evidence')
    fallback_answer = RagAnswer(
        'вопрос', 'Ответ без RAG', (), profile, (), 'fake-qwen',
        generation_mode='llm_only')
    monkeypatch.setattr(api, 'create_llm', lambda **kwargs: FakeLLM())
    monkeypatch.setattr(
        api, 'answer_question', lambda *args, **kwargs: no_evidence)
    monkeypatch.setattr(
        api, 'generate_llm_only_answer',
        lambda *args, **kwargs: fallback_answer)
    response = TestClient(create_app()).post(
        '/api/answer', json={'question': 'вопрос', 'allow_llm_fallback': True})
    assert response.status_code == 200
    payload = response.json()
    assert payload['generation_mode'] == 'llm_only'
    assert payload['answer'] == 'Ответ без RAG'


def test_api_rejects_empty_or_unknown_methods():
    client = TestClient(create_app())
    for methods in ([], ['bad']):
        response = client.post('/api/answer', json={'question': 'вопрос', 'methods': methods})
        assert response.status_code == 422


def test_api_passes_selected_methods(monkeypatch):
    import pdfscan.web.api as api
    monkeypatch.setattr(api, 'create_llm', lambda **kwargs: FakeLLM())
    def answer(question, llm, **kwargs):
        assert kwargs['retrieval_kwargs']['methods'] == ['graph']
        profile = QueryProfile(question, (), {}, ('graph',), methods=('graph',))
        return RagAnswer(question, 'нет документов', (), profile, (), 'fake-qwen',
                         generation_mode='no_evidence')
    monkeypatch.setattr(api, 'answer_question', answer)
    response = TestClient(create_app()).post('/api/answer',
                 json={'question': 'вопрос', 'methods': ['graph']})
    assert response.status_code == 200
    assert response.json()['profile']['methods'] == ['graph']
