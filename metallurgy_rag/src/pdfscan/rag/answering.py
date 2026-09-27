"""Grounded answer generation поверх retrieval Evidence, с LLM Judge.

Порядок теперь строго search-first:

1. Гибридный retrieval (BM25 + dense через RRF) + cross-encoder
   relevance gate.
2. Если evidence нет — LLM вообще не вызывается для генерации ответа;
   явно сообщаем, что в корпусе нет релевантных документов. LLM-only
   ответ без подтверждения источниками — только явный opt-in через
   ``generate_llm_only_answer`` (вызывающий код решает, предлагать
   его или нет).
3. Если evidence есть — генерируем ответ по top-N фрагментам
   (кумулятивно расширяя N: 1, 2, 3, ...), независимый LLM Judge
   оценивает покрытие вопроса 0..1. Останавливаемся, как только
   покрытие >= порога, иначе возвращаем лучшую из попыток с явной
   пометкой ``hybrid_partial``.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, replace

from pdfscan.rag.judge import DEFAULT_COVERAGE_THRESHOLD, LLMJudge
from pdfscan.rag.llm import ChatLLM
from pdfscan.rag.retrieval import Evidence, QueryProfile, retrieve

_CITATION_RE = re.compile(r'\[E(\d+)\]')

DEFAULT_MAX_ITERATIONS = int(
    os.environ.get('RAG_JUDGE_MAX_ITERATIONS', '3'))

NO_EVIDENCE_MESSAGE = (
    'В корпусе не найдено документов, релевантных вопросу: гибридный '
    'поиск (BM25 + dense-эмбеддинги, RRF) и cross-encoder reranker не '
    'дали ни одного фрагмента выше порога релевантности. '
    'Подтверждённый источниками ответ дать нельзя. Если нужен ответ на '
    'основе общих знаний модели без ссылок на корпус — запросите это '
    'явно (LLM-only режим).'
)

GROUNDED_SYSTEM_PROMPT = '''\
Ты — ассистент по металлургии. Отвечай только на русском языке.
Тебе даны фрагменты, прошедшие гибридный поиск (BM25 + dense-эмбеддинги,
RRF), cross-encoder reranker и порог релевантности. Отвечай ТОЛЬКО на
основе этих фрагментов. После каждого факта, взятого из фрагмента,
ставь его номер в формате [E#]. Если у фрагмента есть строка "Точные
данные:" — это структурные поля (formula_latex, equation_number,
table_id, cell_value и т.п.) именно того объекта, который нашёл
retrieval; приводи формулы и числа по ним, а не по пересказу из текста
вокруг. Если фрагменты не покрывают часть вопроса — прямо укажи, какая
часть не покрыта, и не выдумывай факты, которых там нет.'''

STANDALONE_SYSTEM_PROMPT = '''\
Ты — ассистент по металлургии. Отвечай только на русском языке.
В корпусе не нашлось релевантных документов для этого вопроса. Дай
самостоятельный ответ на основе общих знаний модели. Явно предупреди в
начале ответа, что он не подтверждён источниками корпуса, и не
используй ссылки [E#].'''

# Публичные алиасы для обратной совместимости интеграций/тестов.
DRAFT_SYSTEM_PROMPT = STANDALONE_SYSTEM_PROMPT
AUGMENT_SYSTEM_PROMPT = GROUNDED_SYSTEM_PROMPT
SYSTEM_PROMPT = GROUNDED_SYSTEM_PROMPT


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    citations: tuple[str, ...]
    profile: QueryProfile
    evidence: tuple[Evidence, ...]
    model: str
    # no_evidence | hybrid | hybrid_partial | llm_only
    generation_mode: str = 'no_evidence'
    base_answer: str = ''
    judge_score: float | None = None
    judge_missing: str = ''
    judge_model: str = ''
    evidence_used: int = 0
    iterations: int = 0

    def as_dict(self) -> dict:
        result = asdict(self)
        result['profile'] = asdict(self.profile)
        result['evidence'] = [item.as_dict() for item in self.evidence]
        return result


def _structured_line(structured: dict) -> str:
    """Явные formula_latex/equation_number/table_id и т.п. для LLM.

    Без этого LLM видит только текст родительского chunk'а и должна сама
    угадывать, где именно в нём формула/ячейка, на которую сослался
    retrieval. Structured-поля (см. ``retrieval._structured``) дают точную
    запись напрямую, без риска, что модель процитирует не тот фрагмент.
    """
    parts = [f'{key}={value}' for key, value in structured.items()
             if value not in (None, '', ())]
    return 'Точные данные: ' + '; '.join(parts) if parts else ''


def evidence_context(evidence: list[Evidence], *, per_item_chars=2200,
                     total_chars=12000) -> str:
    """Стабильный компактный контекст, где E-номер — будущая citation."""
    blocks, used = [], 0
    for index, item in enumerate(evidence, 1):
        source = item.context or item.text
        text = source[:per_item_chars]
        header = (f'[E{index}] Документ: {item.document_id}; страницы: '
                 f'{", ".join(map(str, item.pages))}')
        structured = _structured_line(item.structured)
        block = header + (f'\n{structured}' if structured else '') + f'\n{text}'
        if blocks and used + len(block) > total_chars:
            break
        blocks.append(block)
        used += len(block)
    return '\n\n'.join(blocks)


def _citations_from_text(text: str,
                         evidence: list[Evidence]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        evidence[int(number) - 1].unit_id
        for number in _CITATION_RE.findall(text)
        if 0 < int(number) <= len(evidence)))


def generate_llm_only_answer(question: str, llm: ChatLLM, *,
                             profile: QueryProfile | None = None
                             ) -> RagAnswer:
    """Явный opt-in: пользователь согласился на ответ без RAG-проверки.

    Вызывающий код (CLI/API/UI) сам решает, когда это предлагать —
    ``answer_question`` никогда не вызывает эту функцию автоматически.
    """
    text = llm.generate([
        {'role': 'system', 'content': STANDALONE_SYSTEM_PROMPT},
        {'role': 'user', 'content': question},
    ])
    empty_profile = profile or QueryProfile(
        query=question, vector_kinds=(), filters={}, channels=())
    return RagAnswer(question, text, (), empty_profile, (), llm.model,
                     generation_mode='llm_only')


def answer_question(question: str, llm: ChatLLM, *, k=6,
                    retrieval_kwargs=None, judge: LLMJudge | None = None,
                    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
                    max_iterations: int = DEFAULT_MAX_ITERATIONS
                    ) -> RagAnswer:
    """Search-first: retrieval -> генерация -> LLM Judge-цикл покрытия."""
    profile, evidence = retrieve(question, k=k, **(retrieval_kwargs or {}))
    if not evidence:
        return RagAnswer(question, NO_EVIDENCE_MESSAGE, (), profile, (),
                         llm.model, generation_mode='no_evidence')

    judge = judge or LLMJudge(llm)
    limit = max(1, min(max_iterations, len(evidence)))
    best: RagAnswer | None = None
    for iteration in range(1, limit + 1):
        subset = evidence[:iteration]
        context = evidence_context(subset)
        prompt = (f'Вопрос: {question}\n\nФрагменты (top-{iteration} '
                 f'после гибридного поиска и reranker):\n{context}\n\n'
                 f'Дай итоговый ответ.')
        text = llm.generate([
            {'role': 'system', 'content': GROUNDED_SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ])
        verdict = judge.evaluate(question, text, context)
        candidate = RagAnswer(
            question, text, _citations_from_text(text, subset), profile,
            tuple(subset), llm.model, generation_mode='hybrid',
            judge_score=verdict.coverage, judge_missing=verdict.missing,
            judge_model=judge.model_name, evidence_used=len(subset),
            iterations=iteration,
        )
        if best is None or verdict.coverage > (best.judge_score or 0.0):
            best = candidate
        if verdict.coverage >= coverage_threshold:
            return candidate
    # Ни одна попытка не прошла порог покрытия — отдаём лучшую, но
    # помечаем это явно, чтобы UI/API не выдавали её за подтверждённую.
    return replace(best, generation_mode='hybrid_partial')
