from pdfscan.rag import answering
from pdfscan.rag.retrieval import Evidence, QueryProfile


class FakeLLM:
    model = 'fake-qwen'

    def __init__(self):
        self.messages = []

    def generate(self, messages, *, temperature=0.1):
        self.messages = messages
        return 'Ответ на вопрос [E1].'


def _evidence(unit_id='d#c0'):
    return Evidence(
        unit_id=unit_id, unit_type='formula', text='FeO + C', context='Восстановление FeO углеродом.',
        parent_chunk_id='d#c0', document_id='doc', pages=(12,), score=0.8,
        retrieval_channels=('chemistry', 'bm25'), structured={},
    )


def test_answer_uses_evidence_and_returns_real_citation(monkeypatch):
    profile = QueryProfile('Что происходит?', ('text',), {}, ('text', 'bm25'))
    monkeypatch.setattr(answering, 'retrieve', lambda *args, **kwargs: (profile, [_evidence()]))
    llm = FakeLLM()
    result = answering.answer_question('Что происходит?', llm)
    assert result.citations == ('d#c0',)
    assert '[E1]' in llm.messages[1]['content']
    assert 'Восстановление FeO' in llm.messages[1]['content']
    assert result.generation_mode == 'hybrid'
    assert result.base_answer


def test_answer_uses_llm_knowledge_when_rag_has_no_evidence(monkeypatch):
    profile = QueryProfile('?', ('text',), {}, ('text', 'bm25'))
    monkeypatch.setattr(answering, 'retrieve', lambda *args, **kwargs: (profile, []))
    llm = FakeLLM()
    result = answering.answer_question('?', llm)
    assert result.citations == ()
    assert result.answer == 'Ответ на вопрос [E1].'
    assert len(llm.messages) == 2
    assert 'самостоятельный' in llm.messages[0]['content'].lower()
    assert result.generation_mode == 'llm_only'


def test_prompts_start_with_llm_and_only_then_augment_with_rag():
    assert 'самостоятельный' in answering.DRAFT_SYSTEM_PROMPT.lower()
    assert 'reranker' in answering.AUGMENT_SYSTEM_PROMPT.lower()
    assert 'источников' in answering.AUGMENT_SYSTEM_PROMPT.lower()
