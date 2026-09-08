from pathlib import Path

from pdfscan.rag.expert_questions import load_questions


def test_loads_questions_from_reference_workbook():
    workbook = Path(__file__).parents[1] / 'Экспертная таблица (Nord+Эксперты).xlsx'
    items = load_questions(workbook)
    assert len(items) >= 30
    assert items[0]['id'] == '1'
    assert 'молекулярные массы' in items[0]['question']
