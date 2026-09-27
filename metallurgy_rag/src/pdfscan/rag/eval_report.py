"""Сборка человекочитаемого отчёта по результатам batch-прогона RAG.

    python -m pdfscan.rag.eval_report reports/llm_runs/expert_eval.jsonl

Читает JSONL, произведённый ``pdfscan.rag.answer batch``, и формирует
Markdown-отчёт: вопрос, ответ, режим генерации, оценка Judge, цитаты,
плюс сводная статистика и список вопросов, которые стоит проверить
(нет evidence, порог покрытия не достигнут, низкий judge score).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODE_LABELS = {
    'hybrid': 'OK: гибридный поиск, порог покрытия достигнут',
    'hybrid_partial': 'ЧАСТИЧНО: evidence найдено, но Judge не '
                       'подтвердил полное покрытие',
    'llm_only': 'LLM-only: явный fallback без подтверждения источниками',
    'no_evidence': 'НЕТ EVIDENCE: гибридный поиск не нашёл релевантных '
                   'документов',
}


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _fmt_score(value) -> str:
    return f'{value:.2f}' if isinstance(value, (int, float)) else '—'


def render_record(rec: dict) -> str:
    mode = rec.get('generation_mode', '?')
    label = MODE_LABELS.get(mode, mode)
    lines = [
        f'## {rec.get("id")}. {rec.get("question")}',
        '',
        f'**Режим:** {label}  ',
        f'**Judge coverage:** {_fmt_score(rec.get("judge_score"))}'
        f' (модель: {rec.get("judge_model") or "—"},'
        f' итераций: {rec.get("iterations")},'
        f' evidence использовано: {rec.get("evidence_used")})  ',
    ]
    if rec.get('judge_missing'):
        lines.append(f'**Judge отметил как не покрытое:** '
                     f'{rec["judge_missing"]}  ')
    lines += ['', '**Ответ:**', '', rec.get('answer', ''), '']
    citations = rec.get('citations') or []
    if citations:
        lines.append('**Источники:** ' + ', '.join(citations))
    lines.append('')
    lines.append('---')
    return '\n'.join(lines)


def render_summary(records: list[dict]) -> str:
    total = len(records)
    by_mode = {}
    scores = []
    for rec in records:
        mode = rec.get('generation_mode', '?')
        by_mode[mode] = by_mode.get(mode, 0) + 1
        if isinstance(rec.get('judge_score'), (int, float)):
            scores.append(rec['judge_score'])
    avg_score = sum(scores) / len(scores) if scores else 0.0
    lines = [
        '# Отчёт по прогону экспертных вопросов',
        '',
        f'Всего вопросов: **{total}**',
        f'Средний judge coverage (по вопросам с evidence): '
        f'**{avg_score:.2f}** (n={len(scores)})',
        '',
        '| Режим | Кол-во | Доля |',
        '| --- | --- | --- |',
    ]
    for mode, count in sorted(by_mode.items(), key=lambda kv: -kv[1]):
        label = MODE_LABELS.get(mode, mode)
        lines.append(f'| {label} | {count} | {count / total:.0%} |')
    lines.append('')
    flagged = [
        rec for rec in records
        if rec.get('generation_mode') in ('no_evidence', 'hybrid_partial')
        or (isinstance(rec.get('judge_score'), (int, float))
            and rec['judge_score'] < 0.8)
    ]
    if flagged:
        lines.append('## Вопросы, требующие внимания')
        lines.append('')
        lines.append(
            'Либо корпус не содержит прямого ответа, либо вопрос '
            'сформулирован так, что поиск не может однозначно найти '
            'релевантный фрагмент (слишком общий/абстрактный вопрос, '
            'сформулирован не по терминологии корпуса и т.п.):')
        lines.append('')
        for rec in flagged:
            mode = rec.get('generation_mode')
            score = _fmt_score(rec.get('judge_score'))
            lines.append(
                f'- **{rec.get("id")}.** {rec.get("question")} '
                f'— _{MODE_LABELS.get(mode, mode)}_, coverage={score}')
        lines.append('')
    lines.append('---')
    lines.append('')
    return '\n'.join(lines)


def build_report(records: list[dict]) -> str:
    parts = [render_summary(records)]
    parts += [render_record(rec) for rec in records]
    return '\n\n'.join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Markdown-отчёт по JSONL из pdfscan.rag.answer batch')
    parser.add_argument('input', type=Path)
    parser.add_argument('-o', '--out', type=Path)
    args = parser.parse_args(argv)
    records = load_records(args.input)
    report = build_report(records)
    out = args.out or args.input.with_suffix('.report.md')
    out.write_text(report, encoding='utf-8')
    print(f'Отчёт: {out}')


if __name__ == '__main__':
    main()
