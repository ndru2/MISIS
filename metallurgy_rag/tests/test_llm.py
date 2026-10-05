import httpx
import pytest

from pdfscan.rag.llm import LLMError, OpenAICompatibleLLM


def test_openai_client_explicitly_disables_streaming(monkeypatch):
    def post(url, **kwargs):
        assert kwargs['json']['stream'] is False
        return httpx.Response(200, json={'choices': [{'message': {'content': 'Ответ'}}]},
                              request=httpx.Request('POST', url))
    monkeypatch.setattr(httpx, 'post', post)
    assert OpenAICompatibleLLM('metal', 'https://example.test/v1').generate([]) == 'Ответ'


def test_invalid_json_becomes_identifiable_llm_error(monkeypatch):
    monkeypatch.setattr(httpx, 'post', lambda url, **kw: httpx.Response(
        200, content=b'{}{}', request=httpx.Request('POST', url)))
    with pytest.raises(LLMError, match='metal.*некорректный JSON'):
        OpenAICompatibleLLM('metal', 'https://example.test/v1').generate([])
