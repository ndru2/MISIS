"""Резюмирует прогон batch с той позиции, где он оборвался.

Дописывает (append) оставшиеся строки таблицы в тот же JSONL, который
использовал ``pdfscan.rag.answer batch`` (позиционно, без учёта того,
что колонка «№» может повторяться по секциям листа).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path('/Users/andrejsuhanov/Applications/MISIS/metallurgy_rag')
sys.path.insert(0, str(ROOT / 'src'))

from pdfscan.rag.answering import answer_question  # noqa: E402
from pdfscan.rag.expert_questions import load_questions  # noqa: E402
from pdfscan.rag.judge import LLMJudge  # noqa: E402
from pdfscan.rag.llm import create_llm  # noqa: E402

WORKBOOK = ROOT / 'Экспертная таблица (Nord+Эксперты).xlsx'
OUT = ROOT / 'reports/llm_runs/expert_eval.jsonl'


def main():
    questions = load_questions(WORKBOOK, sheet='Лист1')
    done = 0
    if OUT.exists():
        with OUT.open(encoding='utf-8') as handle:
            done = sum(1 for _ in handle)
    remaining = questions[done:]
    print(f'Всего вопросов: {len(questions)}; уже готово: {done}; '
          f'осталось: {len(remaining)}', flush=True)
    llm = create_llm(provider='ollama', model='qwen3:8b',
                      timeout_seconds=180.0)
    judge = LLMJudge(llm)
    with OUT.open('a', encoding='utf-8') as handle:
        for position, item in enumerate(remaining, done + 1):
            result = None
            for attempt in range(1, 6):
                try:
                    result = answer_question(
                        item['question'], llm, k=6, judge=judge)
                    break
                except Exception as exc:  # noqa: BLE001
                    wait = min(30, 5 * attempt)
                    print(f'  ! попытка {attempt} не удалась: {exc!r}; '
                          f'жду {wait}s', flush=True)
                    time.sleep(wait)
            if result is None:
                print(f'  !! пропускаю {item["id"]} после 5 попыток',
                      flush=True)
                continue
            payload = {
                'id': item['id'],
                'provider': 'ollama',
                'qdrant_url': 'http://localhost:6333',
                'qdrant_collection': 'metallurgy_search_units',
                **result.as_dict(),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
            handle.flush()
            print(f'{position}/{len(questions)} {item["id"]}: '
                  f'[{result.generation_mode}] {result.answer[:80]}',
                  flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
