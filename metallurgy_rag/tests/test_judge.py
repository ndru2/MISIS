from pdfscan.rag.judge import LLMJudge, parse_verdict


def test_parse_verdict_reads_strict_json():
    verdict = parse_verdict('{"coverage": 0.91, "missing": ""}')
    assert verdict.coverage == 0.91
    assert verdict.missing == ''


def test_parse_verdict_reads_json_with_surrounding_text():
    raw = ('Вот моя оценка:\n{"coverage": 0.4, "missing": "нет цифр"}\n'
          'Спасибо.')
    verdict = parse_verdict(raw)
    assert verdict.coverage == 0.4
    assert verdict.missing == 'нет цифр'


def test_parse_verdict_falls_back_to_bare_number_when_json_is_broken():
    verdict = parse_verdict('покрытие составляет 0.75 из 1.0')
    assert verdict.coverage == 0.75


def test_parse_verdict_falls_back_to_percent():
    verdict = parse_verdict('coverage: 82%')
    assert verdict.coverage == 0.82


def test_parse_verdict_clamps_out_of_range_values():
    verdict = parse_verdict('{"coverage": 1.4}')
    assert verdict.coverage == 1.0


def test_parse_verdict_defaults_to_zero_when_unparseable():
    verdict = parse_verdict('модель не смогла оценить')
    assert verdict.coverage == 0.0


class FakeLLM:
    model = 'fake-judge'

    def __init__(self, response):
        self.response = response
        self.messages = None

    def generate(self, messages, *, temperature=0.1):
        self.messages = messages
        return self.response


def test_llm_judge_evaluate_calls_llm_and_parses_result():
    llm = FakeLLM('{"coverage": 0.88, "missing": ""}')
    judge = LLMJudge(llm)
    verdict = judge.evaluate('Вопрос?', 'Ответ [E1].', context='[E1] контекст')
    assert verdict.coverage == 0.88
    assert judge.model_name == 'fake-judge'
    assert llm.messages[0]['role'] == 'system'
    assert 'Вопрос?' in llm.messages[1]['content']
    assert 'Ответ [E1].' in llm.messages[1]['content']
