"""FastAPI-граница для RAG: HTTP не влияет на retrieval или LLM-адаптеры."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from pdfscan.rag.answering import (
    DEFAULT_MAX_ITERATIONS, answer_question, generate_llm_only_answer)
from pdfscan.rag.judge import DEFAULT_COVERAGE_THRESHOLD, LLMJudge
from pdfscan.rag.llm import LLMError, create_llm
from pdfscan.rag.qdrant_index import DEFAULT_COLLECTION, DEFAULT_URL
from pdfscan.rag.rerank import DEFAULT_RELEVANCE_THRESHOLD

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiSettings:
    """Настройки окружения; UI может переопределить только LLM-конфиг."""

    qdrant_url: str = DEFAULT_URL
    collection: str = DEFAULT_COLLECTION
    provider: str = 'ollama'
    model: str = 'qwen3:8b'
    llm_base_url: str | None = None
    timeout_seconds: float = 180.0
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    # Judge может быть отдельной (например, тяжёлой бесплатной API)
    # моделью; если поля не заданы — используется тот же
    # provider/model, что и у генератора.
    judge_provider: str | None = None
    judge_model: str | None = None
    judge_base_url: str | None = None
    judge_api_key: str | None = None
    judge_timeout_seconds: float | None = None
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD
    max_iterations: int = DEFAULT_MAX_ITERATIONS

    @classmethod
    def from_env(cls) -> 'ApiSettings':
        # The project-level .env can configure an independent OpenRouter Judge
        # while leaving the answer generator unchanged.
        openrouter_model = os.environ.get('OPENROUTER_MODEL')
        openrouter_key = os.environ.get('OPENROUTER_API_KEY')
        use_kg_openrouter = bool(openrouter_model and openrouter_key)
        return cls(
            qdrant_url=os.environ.get('RAG_QDRANT_URL', DEFAULT_URL),
            collection=os.environ.get(
                'RAG_QDRANT_COLLECTION', DEFAULT_COLLECTION),
            provider=os.environ.get('RAG_LLM_PROVIDER', 'ollama'),
            model=os.environ.get('RAG_LLM_MODEL', 'qwen3:8b'),
            llm_base_url=os.environ.get('RAG_LLM_BASE_URL'),
            timeout_seconds=float(os.environ.get('RAG_LLM_TIMEOUT', '180')),
            relevance_threshold=float(os.environ.get(
                'RAG_RELEVANCE_THRESHOLD',
                str(DEFAULT_RELEVANCE_THRESHOLD))),
            judge_provider=(os.environ.get('RAG_JUDGE_PROVIDER')
                            or ('openai-compatible' if use_kg_openrouter else None)),
            judge_model=(os.environ.get('RAG_JUDGE_MODEL')
                         or (openrouter_model if use_kg_openrouter else None)),
            judge_base_url=(os.environ.get('RAG_JUDGE_BASE_URL')
                            or ('https://openrouter.ai/api/v1'
                                if use_kg_openrouter else None)),
            judge_api_key=os.environ.get('RAG_JUDGE_API_KEY') or openrouter_key,
            judge_timeout_seconds=(
                float(os.environ['RAG_JUDGE_TIMEOUT'])
                if os.environ.get('RAG_JUDGE_TIMEOUT') else None
            ),
            coverage_threshold=float(os.environ.get(
                'RAG_JUDGE_THRESHOLD', str(DEFAULT_COVERAGE_THRESHOLD))),
            max_iterations=int(os.environ.get(
                'RAG_JUDGE_MAX_ITERATIONS', str(DEFAULT_MAX_ITERATIONS))),
        )


class AnswerRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4_000)
    methods: list[Literal['vector', 'bm25', 'graph']] = Field(
        default_factory=lambda: ['vector', 'bm25'], min_length=1, max_length=3)
    evidence_k: int = Field(default=6, ge=1, le=12)
    provider: Literal['ollama', 'openai-compatible'] | None = None
    model: str | None = Field(default=None, min_length=1, max_length=200)
    base_url: str | None = Field(default=None, max_length=500)
    # API key remains server-side. The UI may select a different configured
    # Judge model for one request without receiving that key.
    judge_model: str | None = Field(default=None, min_length=1, max_length=200)
    # Явный opt-in пользователя: если гибридный поиск не нашёл
    # релевантных документов, всё равно сгенерировать ответ силами
    # LLM без подтверждения.
    allow_llm_fallback: bool = False


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings.from_env()
    app = FastAPI(title='Metallurgy RAG API', version='0.1.0')

    def _judge_for(provider, model, base_url, request_judge_model=None):
        judge_model = request_judge_model or settings.judge_model
        has_own = (judge_model or settings.judge_provider
                  or settings.judge_base_url)
        if has_own:
            judge_llm = create_llm(
                provider=settings.judge_provider or 'openai-compatible',
                model=judge_model, base_url=settings.judge_base_url,
                api_key=settings.judge_api_key,
                timeout_seconds=(settings.judge_timeout_seconds
                                 or settings.timeout_seconds))
        else:
            judge_llm = create_llm(
                provider=provider, model=model, base_url=base_url,
                timeout_seconds=settings.timeout_seconds)
        return LLMJudge(judge_llm)

    @app.get('/health')
    def health():
        """Показывает конфигурацию, но не отправляет запрос к модели."""
        return {
            'status': 'ok',
            'qdrant_collection': settings.collection,
            'provider': settings.provider,
            'model': settings.model,
            'relevance_threshold': settings.relevance_threshold,
            'coverage_threshold': settings.coverage_threshold,
            'judge_model': settings.judge_model or settings.model,
            'judge_provider': settings.judge_provider or settings.provider,
            'judge_timeout_seconds': (settings.judge_timeout_seconds
                                      or settings.timeout_seconds),
        }

    @app.post('/api/answer')
    async def answer(request: AnswerRequest):
        provider = request.provider or settings.provider
        model = request.model or settings.model
        base_url = request.base_url or settings.llm_base_url
        try:
            llm = create_llm(provider=provider, model=model,
                             base_url=base_url,
                             timeout_seconds=settings.timeout_seconds)
            result = await run_in_threadpool(
                answer_question, request.question, llm, k=request.evidence_k,
                retrieval_kwargs={
                    'methods': request.methods,
                    'url': settings.qdrant_url,
                    'collection': settings.collection,
                    'relevance_threshold': settings.relevance_threshold,
                },
                judge=_judge_for(provider, model, base_url, request.judge_model),
                coverage_threshold=settings.coverage_threshold,
                max_iterations=settings.max_iterations)
            no_evidence = result.generation_mode == 'no_evidence'
            if no_evidence and request.allow_llm_fallback:
                result = await run_in_threadpool(
                    generate_llm_only_answer, request.question, llm,
                    profile=result.profile)
        except LLMError as exc:
            LOG.warning('LLM request failed: %s', exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            # Qdrant, embedding or unexpected runtime failure.
            LOG.exception('RAG answer failed')
            raise HTTPException(
                status_code=502,
                detail=f'RAG не смог обработать запрос: {exc}') from exc
        return {
            **result.as_dict(),
            'provider': provider,
            'qdrant_collection': settings.collection,
        }

    return app


app = create_app()
