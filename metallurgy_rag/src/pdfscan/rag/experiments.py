"""Reproducible ablation runs over all seven retrieval combinations.

python -m pdfscan.rag.experiments --questions questions.txt --retrieval-only
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from statistics import mean

from pdfscan.rag.answer import (
    DEFAULT_WORKBOOK, _add_judge_args, _add_llm_args, _add_retrieval_args,
    _llm, _retrieval_kwargs, _run)
from pdfscan.rag.expert_questions import load_questions
from pdfscan.rag.retrieval import METHODS, normalize_methods, retrieve

ALL_COMBINATIONS = [combo for size in range(1, 4) for combo in combinations(METHODS, size)]


def summarize(records):
    summary = {}
    for name in dict.fromkeys(row['experiment'] for row in records):
        rows = [row for row in records if row['experiment'] == name]
        good = [row for row in rows if 'error' not in row]
        judge = [row['judge_score'] for row in good
                 if row.get('generation_mode') in ('hybrid', 'hybrid_partial')
                 and isinstance(row.get('judge_score'), (int, float))]
        summary[name] = {
            'runs': len(rows), 'errors': len(rows) - len(good),
            'mean_seconds': mean(row['seconds'] for row in good) if good else None,
            'evidence_found_rate': mean(bool(row.get('evidence')) for row in good) if good else None,
            'mean_candidates': mean(row['profile']['candidate_count'] for row in good) if good else None,
            'mean_accepted': mean(row['profile']['accepted_count'] for row in good) if good else None,
            'mean_judge_coverage': mean(judge) if judge else None,
            'judge_scored_runs': len(judge),
        }
    return summary


def run_experiments(questions, variants, args, handle):
    records = []
    llm = None if args.retrieval_only else _llm(args)
    for item in questions:
        for variant in variants:
            args.methods = normalize_methods(variant)
            record = {'id': item['id'], 'question': item['question'],
                      'experiment': '+'.join(args.methods), 'methods': args.methods,
                      'retrieval_only': args.retrieval_only,
                      'config': {'collection': args.collection, 'qdrant_url': args.qdrant_url,
                                 'neo4j_uri': os.getenv('NEO4J_URI', 'bolt://localhost:7690'),
                                 'neo4j_database': os.getenv('NEO4J_DATABASE', 'neo4j'),
                                 'k': args.k,
                                 'embedding_device': os.getenv('RAG_EMBEDDING_DEVICE') or 'auto',
                                 'relevance_threshold': args.relevance_threshold,
                                 'provider': args.provider, 'model': args.model,
                                 'judge_model': args.judge_model or args.model,
                                 'coverage_threshold': args.coverage_threshold,
                                 'max_iterations': args.max_iterations}}
            print(f'{item["id"]}: {record["experiment"]}: started', flush=True)
            started = time.perf_counter()
            try:
                if args.retrieval_only:
                    profile, evidence = retrieve(item['question'], k=args.k,
                                                 **_retrieval_kwargs(args))
                    record.update(profile=asdict(profile),
                                  evidence=[entry.as_dict() for entry in evidence])
                else:
                    record.update(_run(item['question'], llm, args).as_dict())
            except Exception as exc:
                # A failed channel is a failed experiment, never silently a different mode.
                record['error'] = f'{type(exc).__name__}: {exc}'
            record['seconds'] = round(time.perf_counter() - started, 3)
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
            records.append(record)
            print(f'{item["id"]}: {record["experiment"]}: '
                  f'{"ERROR " + record["error"] if "error" in record else "ok"}', flush=True)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', help='UTF-8 text file, one question per line')
    parser.add_argument('--workbook', default=str(DEFAULT_WORKBOOK))
    parser.add_argument('--sheet', default='Лист1')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--out', required=True, help='new JSONL file (will not overwrite)')
    parser.add_argument('--retrieval-only', action='store_true', help='skip generator and Judge')
    parser.add_argument('--all-combinations', action='store_true')
    _add_llm_args(parser)
    _add_judge_args(parser)
    _add_retrieval_args(parser)
    args = parser.parse_args(argv)
    if args.questions:
        questions = [{'id': str(i), 'question': line.strip()}
                     for i, line in enumerate(Path(args.questions).read_text().splitlines(), 1)
                     if line.strip()]
    else:
        questions = load_questions(args.workbook, sheet=args.sheet)
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        parser.error('No questions to run')
    variants = ALL_COMBINATIONS if args.all_combinations else [args.methods]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        records = run_experiments(questions, variants, args, handle)
    summary = output.with_suffix('.summary.json')
    summary.write_text(json.dumps(summarize(records), ensure_ascii=False, indent=2))
    print(f'Results: {output}; summary: {summary}')
    if any('error' in row for row in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
