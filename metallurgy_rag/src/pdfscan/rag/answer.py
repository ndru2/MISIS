"""Локальный RAG-ответ и batch-прогон вопросов экспертов.

    python -m pdfscan.rag.answer ask "Что такое автогенный процесс?"
    python -m pdfscan.rag.answer batch \\
        --workbook "Экспертная таблица (Nord+Эксперты).xlsx"

Judge может быть отдельной (например, тяжёлой бесплатной API) моделью:

    python -m pdfscan.rag.answer ask "..." \\
        --judge-provider openai-compatible \\
        --judge-base-url https://openrouter.ai/api/v1 \\
        --judge-model deepseek/deepseek-r1:free \\
        --judge-api-key $OPENROUTER_API_KEY
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from pdfscan.paths import REPORTS_DIR, ROOT
from pdfscan.rag.answering import (
    DEFAULT_MAX_ITERATIONS, answer_question, generate_llm_only_answer)
from pdfscan.rag.expert_questions import load_questions
from pdfscan.rag.judge import DEFAULT_COVERAGE_THRESHOLD, LLMJudge
from pdfscan.rag.llm import create_llm
from pdfscan.rag.qdrant_index import DEFAULT_COLLECTION, DEFAULT_URL
from pdfscan.rag.rerank import DEFAULT_RELEVANCE_THRESHOLD

DEFAULT_WORKBOOK = ROOT / 'Экспертная таблица (Nord+Эксперты).xlsx'


def _llm(args):
    return create_llm(provider=args.provider, model=args.model,
                      base_url=args.base_url, timeout_seconds=args.timeout)


def _judge(args, llm):
    """Отдельная judge-модель, если задана; иначе — тот же генератор."""
    if args.judge_model or args.judge_provider or args.judge_base_url:
        judge_llm = create_llm(
            provider=args.judge_provider or 'openai-compatible',
            model=args.judge_model, base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            timeout_seconds=args.judge_timeout or args.timeout)
    else:
        judge_llm = llm
    return LLMJudge(judge_llm)


def _report(result):
    lines = [result.answer]
    if result.generation_mode in ('hybrid', 'hybrid_partial'):
        ok = result.generation_mode == 'hybrid'
        status = 'порог достигнут' if ok else 'ПОРОГ НЕ ДОСТИГНУТ'
        lines.append(
            f'\n[Judge: покрытие {result.judge_score:.2f} ({status}); '
            f'модель judge: {result.judge_model}; '
            f'итераций: {result.iterations}; '
            f'evidence использовано: {result.evidence_used}]')
        if result.judge_missing:
            lines.append(
                f'[Judge отметил как не покрытое: {result.judge_missing}]')
    if result.citations:
        lines.append('\nEvidence: ' + ', '.join(result.citations))
    if result.generation_mode == 'no_evidence':
        lines.append(
            '\n[Нет релевантных документов в корпусе. Запустите с '
            '--allow-llm-fallback, чтобы всё равно получить ответ от '
            'LLM без подтверждения источниками.]')
    return '\n'.join(lines)


def _run(question, llm, args):
    result = answer_question(
        question, llm, k=args.k, retrieval_kwargs=_retrieval_kwargs(args),
        judge=_judge(args, llm), coverage_threshold=args.coverage_threshold,
        max_iterations=args.max_iterations)
    if result.generation_mode == 'no_evidence' and args.allow_llm_fallback:
        result = generate_llm_only_answer(question, llm,
                                          profile=result.profile)
    return result


def command_ask(args):
    llm = _llm(args)
    result = _run(args.question, llm, args)
    print(_report(result))


def command_batch(args):
    questions = load_questions(args.workbook, sheet=args.sheet)
    total = len(questions)
    if args.skip:
        questions = questions[args.skip:]
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        raise SystemExit('В экспертной таблице нет вопросов')
    stamp = f'{datetime.now():%Y%m%d_%H%M%S}'
    safe_model = args.model.replace('/', '_')
    default = REPORTS_DIR / 'llm_runs' / f'{stamp}_{safe_model}.jsonl'
    output = Path(args.out or default)
    output.parent.mkdir(parents=True, exist_ok=True)
    llm = _llm(args)
    mode = 'a' if args.append else 'w'
    with output.open(mode, encoding='utf-8') as handle:
        for position, item in enumerate(questions, 1 + args.skip):
            result = _run(item['question'], llm, args)
            payload = {
                'id': item['id'],
                'provider': args.provider,
                'qdrant_url': args.qdrant_url,
                'qdrant_collection': args.collection,
                **result.as_dict(),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
            handle.flush()
            print(f'{position}/{total} {item["id"]}: '
                  f'[{result.generation_mode}] {result.answer[:80]}')
    print(f'Результаты: {output}')


def _add_llm_args(parser):
    parser.add_argument('--provider', choices=('ollama', 'openai-compatible'),
                        default='ollama')
    parser.add_argument('--model', default='qwen3:8b')
    parser.add_argument('--base-url')
    parser.add_argument('--timeout', type=float, default=180.0)
    parser.add_argument('-k', type=int, default=6,
                        help='число Evidence для LLM')
    parser.add_argument(
        '--allow-llm-fallback', action='store_true',
        help='явный opt-in: если evidence нет, всё равно сгенерировать '
             'ответ силами LLM без подтверждения источниками')


def _add_judge_args(parser):
    parser.add_argument(
        '--judge-provider', choices=('ollama', 'openai-compatible'),
        help='по умолчанию — тот же provider, что и генератор')
    parser.add_argument(
        '--judge-model',
        help='например deepseek/deepseek-r1:free через OpenRouter')
    parser.add_argument(
        '--judge-base-url', help='например https://openrouter.ai/api/v1')
    parser.add_argument('--judge-api-key')
    parser.add_argument('--judge-timeout', type=float)
    parser.add_argument(
        '--coverage-threshold', type=float,
        default=DEFAULT_COVERAGE_THRESHOLD,
        help='минимальное покрытие вопроса ответом по оценке Judge')
    parser.add_argument(
        '--max-iterations', type=int, default=DEFAULT_MAX_ITERATIONS,
        help='сколько раз расширять top-N evidence, пока Judge не '
             'одобрит ответ')


def _add_retrieval_args(parser):
    parser.add_argument('--methods', nargs='+', choices=('vector', 'bm25', 'graph'),
                        default=['vector', 'bm25'], help='любое непустое сочетание методов')
    parser.add_argument('--qdrant-url', default=DEFAULT_URL)
    parser.add_argument('--collection', default=DEFAULT_COLLECTION)
    parser.add_argument(
        '--relevance-threshold', type=float,
        default=DEFAULT_RELEVANCE_THRESHOLD,
        help='минимальная score cross-encoder для передачи Evidence в LLM')


def _retrieval_kwargs(args):
    return {'methods': args.methods, 'url': args.qdrant_url, 'collection': args.collection,
            'relevance_threshold': args.relevance_threshold}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Grounded generation поверх metallurgy RAG')
    sub = parser.add_subparsers(dest='command', required=True)
    ask = sub.add_parser('ask', help='ответить на один вопрос')
    ask.add_argument('question')
    _add_llm_args(ask)
    _add_judge_args(ask)
    _add_retrieval_args(ask)
    ask.set_defaults(func=command_ask)
    batch = sub.add_parser('batch', help='прогнать вопросы из таблицы')
    batch.add_argument('--workbook', default=str(DEFAULT_WORKBOOK))
    batch.add_argument('--sheet', default='Лист1')
    batch.add_argument('--limit', type=int)
    batch.add_argument('--skip', type=int, default=0,
                       help='пропустить первые N вопросов (для докачки)')
    batch.add_argument('--append', action='store_true',
                       help='дописывать в существующий --out, не '
                            'перезаписывать')
    batch.add_argument('--out')
    _add_llm_args(batch)
    _add_judge_args(batch)
    _add_retrieval_args(batch)
    batch.set_defaults(func=command_batch)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
