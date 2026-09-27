from pdfscan.rag import answering
from pdfscan.rag.judge import JudgeVerdict
from pdfscan.rag.retrieval import Evidence, QueryProfile


class FakeLLM:
    model = 'fake-qwen'

    def __init__(self, responses=None):
        self.responses = list(responses) if responses is not None else None
        self.calls = []

    def generate(self, messages, *, temperature=0.1):
        self.calls.append(messages)
        if self.responses is not None:
            return self.responses.pop(0)
        return 'Ответ на вопрос [E1].'


class FakeJudge:
    """Очередь вердиктов на каждую итерацию расширения evidence."""

    def __init__(self, coverages):
        self.coverages = list(coverages)
        self.calls = []
        self.model_name = 'fake-judge'

    def evaluate(self, question, answer, context=''):
        self.calls.append((question, answer, context))
        missing = '' if self.coverages else 'нет данных'
        return JudgeVerdict(coverage=self.coverages.pop(0), missing=missing)


def _evidence(unit_id='d#c0', score=0.8):
    return Evidence(
        unit_id=unit_id, unit_type='formula', text='FeO + C',
        context='Восстановление FeO углеродом.', parent_chunk_id='d#c0',
        document_id='doc', pages=(12,), score=score,
        retrieval_channels=('chemistry', 'bm25'), structured={},
    )


def test_no_evidence_returns_explicit_message_without_calling_llm(
        monkeypatch):
    profile = QueryProfile('?', ('text',), {}, ('text', 'bm25'))
    monkeypatch.setattr(
        answering, 'retrieve', lambda *args, **kwargs: (profile, []))
    llm = FakeLLM()
    result = answering.answer_question('?', llm)
    assert result.generation_mode == 'no_evidence'
    assert result.answer == answering.NO_EVIDENCE_MESSAGE
    assert result.citations == ()
    assert result.evidence == ()
    assert llm.calls == []  # LLM не вызывается автоматически


def test_generate_llm_only_answer_is_explicit_opt_in():
    llm = FakeLLM(['Самостоятельный ответ без источников.'])
    result = answering.generate_llm_only_answer('Что такое штейн?', llm)
    assert result.generation_mode == 'llm_only'
    assert result.answer == 'Самостоятельный ответ без источников.'
    assert result.citations == ()
    assert 'самостоятельный' in llm.calls[0][0]['content'].lower()


def test_answer_stops_at_top1_when_judge_covers_immediately(monkeypatch):
    profile = QueryProfile(
        'FeO + C', ('chemistry',), {}, ('chemistry', 'bm25'))
    evidence = [_evidence('d#c0', 0.9), _evidence('d#c1', 0.5)]
    monkeypatch.setattr(
        answering, 'retrieve', lambda *args, **kwargs: (profile, evidence))
    llm = FakeLLM(['Ответ на основе первого фрагмента [E1].'])
    judge = FakeJudge([0.9])
    result = answering.answer_question(
        'FeO + C', llm, judge=judge, coverage_threshold=0.82)
    assert result.generation_mode == 'hybrid'
    assert result.evidence_used == 1
    assert result.iterations == 1
    assert result.judge_score == 0.9
    assert result.citations == ('d#c0',)
    assert len(llm.calls) == 1
    assert len(judge.calls) == 1


def test_answer_expands_to_top2_when_top1_does_not_cover(monkeypatch):
    profile = QueryProfile(
        'FeO + C', ('chemistry',), {}, ('chemistry', 'bm25'))
    evidence = [_evidence('d#c0', 0.9), _evidence('d#c1', 0.5)]
    monkeypatch.setattr(
        answering, 'retrieve', lambda *args, **kwargs: (profile, evidence))
    llm = FakeLLM(['Неполный ответ [E1].', 'Полный ответ [E1][E2].'])
    judge = FakeJudge([0.4, 0.9])
    result = answering.answer_question(
        'FeO + C', llm, judge=judge, coverage_threshold=0.82)
    assert result.generation_mode == 'hybrid'
    assert result.evidence_used == 2
    assert result.iterations == 2
    assert result.judge_score == 0.9
    assert set(result.citations) == {'d#c0', 'd#c1'}
    assert len(llm.calls) == 2


def test_answer_returns_best_effort_when_threshold_never_reached(
        monkeypatch):
    profile = QueryProfile(
        'FeO + C', ('chemistry',), {}, ('chemistry', 'bm25'))
    evidence = [_evidence('d#c0', 0.9), _evidence('d#c1', 0.5)]
    monkeypatch.setattr(
        answering, 'retrieve', lambda *args, **kwargs: (profile, evidence))
    llm = FakeLLM(['Ответ 1 [E1].', 'Ответ 2 [E1][E2].'])
    judge = FakeJudge([0.3, 0.6])
    result = answering.answer_question(
        'FeO + C', llm, judge=judge, coverage_threshold=0.82,
        max_iterations=2)
    assert result.generation_mode == 'hybrid_partial'
    assert result.judge_score == 0.6  # лучшая из попыток
    assert result.evidence_used == 2
    assert result.iterations == 2


def test_answer_respects_max_iterations_below_evidence_count(monkeypatch):
    profile = QueryProfile('вопрос', ('text',), {}, ('text', 'bm25'))
    evidence = [_evidence('d#c0'), _evidence('d#c1'), _evidence('d#c2')]
    monkeypatch.setattr(
        answering, 'retrieve', lambda *args, **kwargs: (profile, evidence))
    llm = FakeLLM(['Ответ 1.', 'Ответ 2.'])
    judge = FakeJudge([0.1, 0.2])
    result = answering.answer_question(
        'вопрос', llm, judge=judge, coverage_threshold=0.9,
        max_iterations=2)
    assert result.iterations == 2  # ограничено max_iterations
    assert len(llm.calls) == 2


def test_default_judge_reuses_generation_llm_when_not_provided(monkeypatch):
    profile = QueryProfile('вопрос', ('text',), {}, ('text', 'bm25'))
    evidence = [_evidence()]
    monkeypatch.setattr(
        answering, 'retrieve', lambda *args, **kwargs: (profile, evidence))
    llm = FakeLLM(['Ответ [E1].', '{"coverage": 0.95, "missing": ""}'])
    result = answering.answer_question('вопрос', llm, coverage_threshold=0.82)
    assert result.generation_mode == 'hybrid'
    assert result.judge_model == 'fake-qwen'
    assert len(llm.calls) == 2  # генерация ответа + judge на той же LLM


def test_evidence_context_exposes_structured_fields_for_llm():
    evidence = Evidence(
        unit_id='d#c0#formula', unit_type='formula', text='K_{Ni/Ca}=...',
        context='Коэффициент распределения никеля между шлаком и металлом.',
        parent_chunk_id='d#c0', document_id='doc', pages=(12,), score=0.8,
        retrieval_channels=('math', 'bm25'),
        structured={'formula_latex': 'K_{Ni/Ca}', 'equation_number': '5.22'},
    )
    context = answering.evidence_context([evidence])
    assert 'Точные данные:' in context
    assert 'formula_latex=K_{Ni/Ca}' in context
    assert 'equation_number=5.22' in context
    assert 'Коэффициент распределения' in context


def test_evidence_context_skips_structured_line_when_empty():
    evidence = _evidence()  # structured={} по умолчанию в _evidence()
    context = answering.evidence_context([evidence])
    assert 'Точные данные:' not in context


def test_prompts_describe_grounded_and_standalone_modes():
    assert 'reranker' in answering.GROUNDED_SYSTEM_PROMPT.lower()
    assert 'e#' in answering.GROUNDED_SYSTEM_PROMPT.lower()
    assert 'самостоятельный' in answering.STANDALONE_SYSTEM_PROMPT.lower()
    assert ('не подтверждён источниками'
           in answering.STANDALONE_SYSTEM_PROMPT.lower())
