"""Grounded answer generation поверх retrieval Evidence."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from pdfscan.rag.llm import ChatLLM
from pdfscan.rag.retrieval import Evidence, QueryProfile, retrieve

_CITATION_RE = re.compile(r'\[E(\d+)\]')
DRAFT_SYSTEM_PROMPT = '''Ты — ассистент по металлургии. Отвечай только на русском языке.
Дай самостоятельный, краткий и точный ответ на вопрос на основе своих общих знаний.
Не используй ссылки [E#] и не утверждай, что факт подтверждён документами корпуса.'''

AUGMENT_SYSTEM_PROMPT = '''Ты — ассистент по металлургии. Отвечай только на русском языке.
Перед тобой самостоятельный черновой ответ модели и фрагменты, которые уже прошли
гибридный поиск, reranker и порог релевантности. Уточни или дополни черновик только
фактами, прямо следующими из фрагментов. После такого факта поставь [E#]. Не меняй
правильные общие факты черновика на случайные детали источников и не добавляй ссылку
к утверждению, которое источники не подтверждают.'''

# Сохранено как публичное имя для интеграций, которым нужен общий prompt.
SYSTEM_PROMPT = AUGMENT_SYSTEM_PROMPT


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    citations: tuple[str, ...]
    profile: QueryProfile
    evidence: tuple[Evidence, ...]
    model: str
    base_answer: str = ''
    generation_mode: str = 'llm_only'

    def as_dict(self) -> dict:
        result = asdict(self)
        result['profile'] = asdict(self.profile)
        result['evidence'] = [item.as_dict() for item in self.evidence]
        return result


def evidence_context(evidence: list[Evidence], *, per_item_chars=2200,
                     total_chars=12000) -> str:
    """Стабильный компактный контекст, где E-номер — будущая citation."""
    blocks, used = [], 0
    for index, item in enumerate(evidence, 1):
        source = item.context or item.text
        text = source[:per_item_chars]
        block = (f'[E{index}] Документ: {item.document_id}; страницы: '
                 f'{", ".join(map(str, item.pages))}\n{text}')
        if blocks and used + len(block) > total_chars:
            break
        blocks.append(block)
        used += len(block)
    return '\n\n'.join(blocks)


def answer_question(question: str, llm: ChatLLM, *, k=6, retrieval_kwargs=None) -> RagAnswer:
    """LLM draft → hybrid retrieval/rerank gate → optional RAG augmentation."""
    draft = llm.generate([
        {'role': 'system', 'content': DRAFT_SYSTEM_PROMPT},
        {'role': 'user', 'content': question},
    ])
    profile, evidence = retrieve(question, k=k, **(retrieval_kwargs or {}))
    if not evidence:
        return RagAnswer(question, draft, (), profile, (), llm.model,
                         base_answer=draft, generation_mode='llm_only')
    context = evidence_context(evidence)
    prompt = (f'Вопрос: {question}\n\nСамостоятельный черновик модели:\n{draft}\n\n'
              f'Проверенные фрагменты RAG:\n{context}\n\nДай единый итоговый ответ.')
    text = llm.generate([
        {'role': 'system', 'content': AUGMENT_SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt},
    ])
    cited = tuple(dict.fromkeys(
        evidence[int(number) - 1].unit_id for number in _CITATION_RE.findall(text)
        if 0 < int(number) <= len(evidence)))
    return RagAnswer(question, text, cited, profile, tuple(evidence), llm.model,
                     base_answer=draft, generation_mode='hybrid')
