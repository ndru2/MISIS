"""Второй этап ранжирования кандидатов после Qdrant RRF.

Qdrant возвращает ближайшие ``top-N`` даже если в корпусе нет ответа. Этот
модуль даёт каждому кандидату независимую relevance-оценку и позволяет не
передавать нерелевантный контекст генератору.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from pdfscan.rag.tokenize import tokenize_unique

DEFAULT_RERANKER_MODEL = os.environ.get('RAG_RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3')
DEFAULT_RELEVANCE_THRESHOLD = float(os.environ.get('RAG_RELEVANCE_THRESHOLD', '0.55'))

_STOP_TOKENS = {
    'как', 'каков', 'какая', 'какой', 'какие', 'что', 'чем', 'чего', 'ли',
    'это', 'для', 'про', 'при', 'в', 'на', 'и', 'или', 'а', 'от', 'до',
}


def _lexical_score(query: str, text: str) -> float:
    """Безопасный fallback, если cross-encoder недоступен локально."""
    query_terms = set(tokenize_unique(query)) - _STOP_TOKENS
    if not query_terms:
        return 0.0
    text_terms = set(tokenize_unique(text))
    return len(query_terms & text_terms) / len(query_terms)


@dataclass
class CrossEncoderReranker:
    """Multilingual cross-encoder; загружается лениво только при первом запросе."""

    model_name: str = DEFAULT_RERANKER_MODEL
    _model: object | None = None
    _failed: bool = False

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not self._failed and self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception:
                # Ретривер остаётся рабочим offline: строгий lexical gate лучше,
                # чем передать LLM произвольные top-K результаты.
                self._failed = True
        if self._model is None:
            return [_lexical_score(query, text) for text in texts]
        raw = self._model.predict([(query, text) for text in texts], show_progress_bar=False)
        # BGE reranker возвращает logits; sigmoid переводит их в [0, 1], что
        # делает порог переносимым между запросами и видимым пользователю.
        return [1.0 / (1.0 + math.exp(-float(value))) for value in raw]


_DEFAULT_RERANKER: CrossEncoderReranker | None = None


def default_reranker() -> CrossEncoderReranker:
    global _DEFAULT_RERANKER
    if _DEFAULT_RERANKER is None:
        _DEFAULT_RERANKER = CrossEncoderReranker()
    return _DEFAULT_RERANKER
