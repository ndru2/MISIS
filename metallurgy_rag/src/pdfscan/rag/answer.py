"""Локальный RAG-ответ и batch-прогон вопросов экспертов.

    python -m pdfscan.rag.answer ask "Что такое автогенный процесс?"
    python -m pdfscan.rag.answer batch --workbook "Экспертная таблица (Nord+Эксперты).xlsx"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from pdfscan.paths import REPORTS_DIR, ROOT
from pdfscan.rag.answering import answer_question
from pdfscan.rag.expert_questions import load_questions
from pdfscan.rag.llm import create_llm
from pdfscan.rag.qdrant_index import DEFAULT_COLLECTION, DEFAULT_URL
from pdfscan.rag.rerank import DEFAULT_RELEVANCE_THRESHOLD

DEFAULT_WORKBOOK = ROOT / 'Экспертная таблица (Nord+Эксперты).xlsx'


def _llm(args):
    return create_llm(provider=args.provider, model=args.model, base_url=args.base_url,
                      timeout_seconds=args.timeout)


def command_ask(args):
    result = answer_question(args.question, _llm(args), k=args.k,
                             retrieval_kwargs=_retrieval_kwargs(args))
    print(result.answer)
    if result.citations:
        print('\nEvidence: ' + ', '.join(result.citations))


def command_batch(args):
    questions = load_questions(args.workbook, sheet=args.sheet)
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        raise SystemExit('В экспертной таблице нет вопросов')
    default = REPORTS_DIR / 'llm_runs' / f'{datetime.now():%Y%m%d_%H%M%S}_{args.model.replace("/", "_")}.jsonl'
    output = Path(args.out or default)
    output.parent.mkdir(parents=True, exist_ok=True)
    llm = _llm(args)
    with output.open('w', encoding='utf-8') as handle:
        for position, item in enumerate(questions, 1):
            result = answer_question(item['question'], llm, k=args.k,
                                     retrieval_kwargs=_retrieval_kwargs(args))
            payload = {
                'id': item['id'],
                'provider': args.provider,
                'qdrant_url': args.qdrant_url,
                'qdrant_collection': args.collection,
                **result.as_dict(),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
            handle.flush()
            print(f'{position}/{len(questions)} {item["id"]}: {result.answer[:90]}')
    print(f'Результаты: {output}')


def _add_llm_args(parser):
    parser.add_argument('--provider', choices=('ollama', 'openai-compatible'), default='ollama')
    parser.add_argument('--model', default='qwen3:8b')
    parser.add_argument('--base-url')
    parser.add_argument('--timeout', type=float, default=180.0)
    parser.add_argument('-k', type=int, default=6, help='число Evidence для LLM')


def _add_retrieval_args(parser):
    parser.add_argument('--qdrant-url', default=DEFAULT_URL)
    parser.add_argument('--collection', default=DEFAULT_COLLECTION)
    parser.add_argument('--relevance-threshold', type=float, default=DEFAULT_RELEVANCE_THRESHOLD,
                        help='минимальная score cross-encoder для передачи Evidence в LLM')


def _retrieval_kwargs(args):
    return {'url': args.qdrant_url, 'collection': args.collection,
            'relevance_threshold': args.relevance_threshold}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Grounded generation поверх metallurgy RAG')
    sub = parser.add_subparsers(dest='command', required=True)
    ask = sub.add_parser('ask', help='ответить на один вопрос')
    ask.add_argument('question')
    _add_llm_args(ask)
    _add_retrieval_args(ask)
    ask.set_defaults(func=command_ask)
    batch = sub.add_parser('batch', help='прогнать вопросы из экспертной таблицы')
    batch.add_argument('--workbook', default=str(DEFAULT_WORKBOOK))
    batch.add_argument('--sheet', default='Лист1')
    batch.add_argument('--limit', type=int)
    batch.add_argument('--out')
    _add_llm_args(batch)
    _add_retrieval_args(batch)
    batch.set_defaults(func=command_batch)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
