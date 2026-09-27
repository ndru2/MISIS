"""LLM Judge: независимая оценка покрытия ответа относительно вопроса.

Judge — это отдельный LLM-вызов (может быть другой моделью/провайдером,
например тяжёлой бесплатной моделью по API), который получает вопрос и
черновой ответ и возвращает число 0..1 — какую долю вопроса ответ
закрывает по существу. ``answering.answer_question`` использует это
число, чтобы решить, достаточно ли top-N evidence или нужно расширить
контекст.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

DEFAULT_COVERAGE_THRESHOLD = float(
    os.environ.get('RAG_JUDGE_THRESHOLD', '0.82'))

JUDGE_SYSTEM_PROMPT = '''\
Ты — независимый эксперт-контролёр (judge) по металлургии.
Тебе даны вопрос пользователя и черновой ответ, подготовленный другой
моделью на основе фрагментов корпуса. Оцени, какую долю вопроса ответ
закрывает по существу — число от 0.0 до 1.0 (1.0 = ответ полностью и
точно закрывает вопрос, 0.0 = не отвечает вовсе). Оценивай только
фактическое покрытие содержания, а не стиль или длину. Верни СТРОГО
JSON без текста вокруг:
{"coverage": <float 0..1>, "missing": "<что не покрыто, коротко; \
пустая строка, если покрыто>"}'''

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)
_NUMBER_RE = re.compile(
    r'(\d{1,3})\s*%|(?<!\d)(0?\.\d+|1(?:\.0+)?)(?!\d)')


@dataclass(frozen=True)
class JudgeVerdict:
    coverage: float
    missing: str = ''
    raw: str = ''


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def parse_verdict(raw: str) -> JudgeVerdict:
    """Разбирает ответ judge-модели; переживает отклонения от JSON."""
    match = _JSON_RE.search(raw or '')
    if match:
        try:
            data = json.loads(match.group(0))
            coverage = _clamp(float(data.get('coverage', 0.0)))
            missing = str(data.get('missing') or '').strip()
            return JudgeVerdict(coverage=coverage, missing=missing, raw=raw)
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    # Фолбэк: модель написала число текстом
    # ("покрытие 0.8" / "coverage: 82%").
    found = _NUMBER_RE.search(raw or '')
    if found:
        percent, fraction = found.groups()
        value = float(percent) / 100 if percent else float(fraction)
        return JudgeVerdict(coverage=_clamp(value), raw=raw)
    return JudgeVerdict(coverage=0.0, raw=raw)


@dataclass
class LLMJudge:
    """Оборачивает любой ``ChatLLM``; можно подставить отдельную модель."""

    llm: object

    @property
    def model_name(self) -> str:
        return getattr(self.llm, 'model', self.llm.__class__.__name__)

    def evaluate(self, question: str, answer: str,
                 context: str = '') -> JudgeVerdict:
        prompt = f'Вопрос: {question}\n\nОтвет для проверки:\n{answer}'
        if context:
            prompt += f'\n\nФрагменты, на которые опирался ответ:\n{context}'
        raw = self.llm.generate([
            {'role': 'system', 'content': JUDGE_SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ], temperature=0.0)
        return parse_verdict(raw)
