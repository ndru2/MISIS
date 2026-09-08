"""Заменяемые chat-клиенты для генератора RAG.

Ollama подходит для локального Qwen на macOS. OpenAI-compatible endpoint
покрывает vLLM, LM Studio и будущой сервер без зависимости от их SDK.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


class LLMError(RuntimeError):
    pass


class ChatLLM:
    """Минимальный интерфейс; любой провайдер обязан реализовать generate."""

    model: str

    def generate(self, messages, *, temperature=0.1):  # pragma: no cover - contract only
        raise NotImplementedError


@dataclass
class OllamaLLM:
    model: str = 'qwen3:8b'
    base_url: str = 'http://localhost:11434'
    timeout_seconds: float = 180.0

    def generate(self, messages: list[dict[str, str]], *, temperature: float = 0.1) -> str:
        try:
            response = httpx.post(
                f'{self.base_url.rstrip("/")}/api/chat',
                json={'model': self.model, 'messages': messages, 'stream': False,
                      'options': {'temperature': temperature}},
                timeout=self.timeout_seconds)
            response.raise_for_status()
            text = (response.json().get('message') or {}).get('content') or ''
        except httpx.HTTPError as exc:
            raise LLMError(f'Ollama недоступен по {self.base_url}: {exc}') from exc
        if not text.strip():
            raise LLMError('Ollama вернул пустой ответ')
        return text.strip()


@dataclass
class OpenAICompatibleLLM:
    model: str
    base_url: str
    api_key: str | None = None
    timeout_seconds: float = 180.0

    def generate(self, messages: list[dict[str, str]], *, temperature: float = 0.1) -> str:
        headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}
        try:
            response = httpx.post(
                f'{self.base_url.rstrip("/")}/chat/completions', headers=headers,
                json={'model': self.model, 'messages': messages, 'temperature': temperature},
                timeout=self.timeout_seconds)
            response.raise_for_status()
            text = (((response.json().get('choices') or [{}])[0].get('message') or {})
                    .get('content') or '')
        except httpx.HTTPError as exc:
            raise LLMError(f'LLM endpoint недоступен по {self.base_url}: {exc}') from exc
        if not text.strip():
            raise LLMError('LLM endpoint вернул пустой ответ')
        return text.strip()


def create_llm(*, provider=None, model=None, base_url=None, timeout_seconds=180.0):
    provider = provider or os.environ.get('RAG_LLM_PROVIDER', 'ollama')
    model = model or os.environ.get('RAG_LLM_MODEL', 'qwen3:8b')
    if provider == 'ollama':
        return OllamaLLM(model=model, base_url=base_url or os.environ.get(
            'RAG_OLLAMA_URL', 'http://localhost:11434'), timeout_seconds=timeout_seconds)
    if provider == 'openai-compatible':
        endpoint = base_url or os.environ.get('RAG_LLM_BASE_URL')
        if not endpoint:
            raise LLMError('Для openai-compatible задайте --base-url или RAG_LLM_BASE_URL')
        return OpenAICompatibleLLM(
            model=model, base_url=endpoint, api_key=os.environ.get('RAG_LLM_API_KEY'),
            timeout_seconds=timeout_seconds)
    raise LLMError(f'Неизвестный LLM provider: {provider}')
