from fastapi.testclient import TestClient

from pdfscan.rag.answering import RagAnswer
from pdfscan.rag.retrieval import Evidence, QueryProfile
from pdfscan.web.api import ApiSettings, create_app


class FakeLLM:
    model = 'fake-qwen'

    def generate(self, messages, *, temperature=0.1):
        return 'Ответ [E1]'


def test_health_exposes_server_configuration():
    app = create_app(ApiSettings(collection='test_collection', model='test-model'))
    response = TestClient(app).get('/health')
    assert response.status_code == 200
    assert response.json()['qdrant_collection'] == 'test_collection'


def test_answer_returns_evidence(monkeypatch):
    import pdfscan.web.api as api

    profile = QueryProfile('вопрос', ('text',), {}, ('text', 'bm25'))
    evidence = Evidence('u-1', 'chunk', 'Источник', 'Контекст', None, 'book', (12,),
                        0.9, ('text', 'bm25'), {})
    answer = RagAnswer('вопрос', 'Ответ [E1]', ('u-1',), profile, (evidence,), 'fake-qwen')
    monkeypatch.setattr(api, 'create_llm', lambda **kwargs: FakeLLM())
    monkeypatch.setattr(api, 'answer_question', lambda *args, **kwargs: answer)
    response = TestClient(create_app()).post('/api/answer', json={'question': 'вопрос'})
    assert response.status_code == 200
    payload = response.json()
    assert payload['answer'] == 'Ответ [E1]'
    assert payload['evidence'][0]['unit_id'] == 'u-1'
