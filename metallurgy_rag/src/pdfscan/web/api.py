"""FastAPI-граница для RAG: HTTP не влияет на retrieval или LLM-адаптеры."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from pdfscan.rag.answering import answer_question
from pdfscan.rag.llm import LLMError, create_llm
from pdfscan.rag.qdrant_index import DEFAULT_COLLECTION, DEFAULT_URL
from pdfscan.rag.rerank import DEFAULT_RELEVANCE_THRESHOLD

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiSettings:
    """Настройки окружения; UI может переопределить только LLM-конфигурацию."""

    qdrant_url: str = DEFAULT_URL
    collection: str = DEFAULT_COLLECTION
    provider: str = 'ollama'
    model: str = 'qwen3:8b'
    llm_base_url: str | None = None
    timeout_seconds: float = 180.0
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD

    @classmethod
    def from_env(cls) -> 'ApiSettings':
        return cls(
            qdrant_url=os.environ.get('RAG_QDRANT_URL', DEFAULT_URL),
            collection=os.environ.get('RAG_QDRANT_COLLECTION', DEFAULT_COLLECTION),
            provider=os.environ.get('RAG_LLM_PROVIDER', 'ollama'),
            model=os.environ.get('RAG_LLM_MODEL', 'qwen3:8b'),
            llm_base_url=os.environ.get('RAG_LLM_BASE_URL'),
            timeout_seconds=float(os.environ.get('RAG_LLM_TIMEOUT', '180')),
            relevance_threshold=float(os.environ.get(
                'RAG_RELEVANCE_THRESHOLD', str(DEFAULT_RELEVANCE_THRESHOLD))),
        )


class AnswerRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4_000)
    evidence_k: int = Field(default=6, ge=1, le=12)
    provider: Literal['ollama', 'openai-compatible'] | None = None
    model: str | None = Field(default=None, min_length=1, max_length=200)
    base_url: str | None = Field(default=None, max_length=500)


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings.from_env()
    app = FastAPI(title='Metallurgy RAG API', version='0.1.0')

    @app.get('/health')
    def health():
        """Показывает конфигурацию, но не отправляет запрос к модели."""
        return {
            'status': 'ok',
            'qdrant_collection': settings.collection,
            'provider': settings.provider,
            'model': settings.model,
            'relevance_threshold': settings.relevance_threshold,
        }

    @app.post('/api/answer')
    async def answer(request: AnswerRequest):
        provider = request.provider or settings.provider
        model = request.model or settings.model
        base_url = request.base_url or settings.llm_base_url
        try:
            llm = create_llm(provider=provider, model=model, base_url=base_url,
                             timeout_seconds=settings.timeout_seconds)
            result = await run_in_threadpool(
                answer_question, request.question, llm, k=request.evidence_k,
                retrieval_kwargs={
                    'url': settings.qdrant_url,
                    'collection': settings.collection,
                    'relevance_threshold': settings.relevance_threshold,
                })
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # Qdrant, embedding or unexpected runtime failure.
            LOG.exception('RAG answer failed')
            raise HTTPException(status_code=502, detail=f'RAG не смог обработать запрос: {exc}') from exc
        return {
            **result.as_dict(),
            'provider': provider,
            'qdrant_collection': settings.collection,
        }

    return app


app = create_app()
