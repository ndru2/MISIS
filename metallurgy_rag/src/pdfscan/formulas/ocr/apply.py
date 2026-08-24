"""Запись распознанных формул обратно в blocks.jsonl.

Правка идёт в те же файлы, а не в отдельное дерево: иначе следующие этапы надо
учить выбирать между двумя источниками правды, и рано или поздно они выберут
неверный. Но правка на месте требует трёх предосторожностей.

Первая — обратимость. До первой записи рядом появляется ``blocks.jsonl.orig``,
и вернуться к исходному разбору можно всегда, не запуская его заново.

Вторая — целостность. Файл пишется во временный и переставляется одним вызовом,
поэтому обрыв посреди работы оставляет либо прежний файл, либо новый, но не
половину. Двести документов правятся по одному, и прерванный прогон не оставляет
корпус в смешанном состоянии.

Третья — прослеживаемость. Прежний текст остаётся в блоке в поле
``text_original``, а рядом с документом ложится сводка решений. Через месяц
вопрос «почему здесь такая формула» должен иметь ответ.
"""

import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from . import config, decide


def _read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def manifest_scope(path=None) -> dict:
    """Что мы сейчас считаем формулой: блоки из манифеста, по файлам.

    Манифест пересобирается вместе с правилами отбора, а ``recognized.jsonl``
    растёт только вперёд: в нём лежит всё, что когда-либо распознавалось. После
    ужесточения отбора там остаются ответы по блокам, которые формулами больше не
    считаются, — например по рамке в пять пунктов, где формуле не поместиться.
    Данные они не испортят, решение по ним всё равно будет отказом, но счётчики и
    примеры в отчёте засорят, а отчёт нужен для выбора порогов.

    Отсутствие манифеста даёт пустую область видимости, и тогда сверки нет:
    молча выбросить работу из-за пропавшего файла хуже, чем показать лишнее.
    """
    path = Path(path or config.MANIFEST)
    if not path.is_file():
        return {}
    scope = defaultdict(set)
    for record in _read_jsonl(path):
        if 'blocks' in record and 'block_id' in record:
            scope[record['blocks']].add(record['block_id'])
    return dict(scope)


def in_scope(record, scope) -> bool:
    """Считается ли этот блок формулой по текущим правилам отбора."""
    if not scope:
        return True
    return record.get('block_id') in scope.get(record.get('blocks'), ())


def _load_recognized(path, scope=None) -> tuple:
    """Группирует результаты по файлам блоков, отбрасывая вышедшие из отбора.

    Возвращает ``(результаты по файлам, число отброшенных)``.
    """
    by_file = defaultdict(dict)
    skipped = 0
    for record in _read_jsonl(path):
        if not in_scope(record, scope):
            skipped += 1
            continue
        by_file[record['blocks']][record['block_id']] = record
    return by_file, skipped


def backup_once(blocks_path: Path):
    """Сохраняет исходный разбор, если копии ещё нет."""
    backup = blocks_path.with_suffix(blocks_path.suffix + config.BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(blocks_path, backup)
    return backup


def write_atomic(blocks_path: Path, blocks: list):
    """Переписывает файл целиком через временный."""
    temp = blocks_path.with_suffix(blocks_path.suffix + '.tmp')
    with open(temp, 'w', encoding='utf-8') as handle:
        for block in blocks:
            handle.write(json.dumps(block, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, blocks_path)


def apply_document(blocks_path, recognized: dict, cfg=config.DEFAULT) -> dict:
    """Применяет решения к одному документу и возвращает счётчики."""
    blocks_path = Path(blocks_path)
    blocks, verdicts = [], []

    with open(blocks_path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                blocks.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    counts = Counter()
    changed = False
    for block in blocks:
        record = recognized.get(block.get('block_id'))
        if record is None:
            continue

        verdict = decide.decide(block, record.get('latex'), cfg)
        counts[verdict['decision']] += 1

        trace = {
            'model': record.get('model'),
            'model_score': record.get('model_score'),
            'latex': record.get('latex'),
            'crop': record.get('crop'),
            'dpi': record.get('dpi'),
            'selected_because': record.get('reason'),
            'decision': verdict['decision'],
            'reason': verdict['reason'],
        }

        if decide.applies(verdict, cfg):
            # Исходный текст сохраняется только при первой правке: повторный
            # прогон не должен затереть его уже подменённым.
            block.setdefault('text_original', block.get('text'))
            block['text'] = verdict['candidate']
            block['text_source'] = 'formula_ocr'
            trace['applied'] = True
            counts['applied'] += 1
            changed = True
        else:
            trace['applied'] = False

        block['formula_ocr'] = trace
        verdicts.append({'block_id': block['block_id'],
                         'page': block.get('page'), **verdict, **trace})

    if changed:
        backup_once(blocks_path)
        write_atomic(blocks_path, blocks)

    if verdicts:
        sidecar = blocks_path.parent / config.SIDECAR_NAME
        with open(sidecar, 'w', encoding='utf-8') as handle:
            for verdict in verdicts:
                handle.write(json.dumps(verdict, ensure_ascii=False) + '\n')

    return {'counts': counts, 'verdicts': verdicts, 'changed': changed}


def dry_run(recognized_path=None, cfg=config.DEFAULT, progress=True) -> dict:
    """Считает решения и пишет отчёт, не трогая blocks.jsonl.

    Посмотреть примеры замен до того, как они попадут в корпус, важнее, чем
    сэкономить один прогон: неверный порог здесь портит данные молча.
    """
    recognized_path = recognized_path or config.RECOGNIZED
    scope = manifest_scope()
    counts, verdicts = Counter(), []
    skipped = 0

    for record in _read_jsonl(recognized_path):
        if not in_scope(record, scope):
            skipped += 1
            continue
        block = {'text': record.get('text'),
                 'text_source': record.get('text_source')}
        verdict = decide.decide(block, record.get('latex'), cfg)
        counts[verdict['decision']] += 1
        if decide.applies(verdict, cfg):
            counts['applied'] += 1
        verdicts.append({'block_id': record['block_id'],
                         'page': record.get('page'),
                         'selected_because': record.get('reason'), **verdict})

    summary = {'documents_changed': 0, 'counts': dict(counts),
               'total': len(verdicts), 'outside_scope': skipped}
    write_report(verdicts, summary, cfg)
    if progress:
        print(f'разбор без записи: заменило бы {counts["applied"]} формул '
              f'из {len(verdicts)}')
        if skipped:
            print(f'вне текущего отбора, не рассматривалось: {skipped}')
        print(f'отчёт: {config.REPORT}')
    return summary


def run(recognized_path=None, cfg=config.DEFAULT, progress=True) -> dict:
    """Применяет решения по всему корпусу и пишет сводный отчёт."""
    recognized_path = recognized_path or config.RECOGNIZED
    by_file, skipped = _load_recognized(recognized_path, manifest_scope())

    counts = Counter()
    all_verdicts = []
    touched = 0

    for position, (blocks_file, recognized) in enumerate(sorted(by_file.items()), 1):
        path = Path(blocks_file)
        if not path.is_file():
            if progress:
                print(f'  [!] пропал файл блоков: {path}')
            continue

        report = apply_document(path, recognized, cfg)
        counts.update(report['counts'])
        all_verdicts.extend(report['verdicts'])
        touched += 1 if report['changed'] else 0
        if progress:
            applied = report['counts']['applied']
            print(f'[{position}/{len(by_file)}] {path.parent.name}: '
                  f'заменено {applied} из {len(report["verdicts"])}')

    config.DECISIONS.parent.mkdir(parents=True, exist_ok=True)
    with open(config.DECISIONS, 'w', encoding='utf-8') as handle:
        for verdict in all_verdicts:
            handle.write(json.dumps(verdict, ensure_ascii=False) + '\n')

    summary = {
        'documents_changed': touched,
        'counts': dict(counts),
        'total': len(all_verdicts),
        'outside_scope': skipped,
    }
    write_report(all_verdicts, summary, cfg)
    if progress:
        print(f'заменено формул: {counts["applied"]} из {len(all_verdicts)}; '
              f'документов изменено: {touched}')
        if skipped:
            print(f'вне текущего отбора, не рассматривалось: {skipped}')
        print(f'отчёт: {config.REPORT}')
    return summary


def write_report(verdicts, summary, cfg=config.DEFAULT):
    """Отчёт с выборками: без примеров решение о порогах принять нельзя."""
    lines = ['# Повторное распознавание формул', '',
             f'- всего рассмотрено: {summary["total"]}',
             f'- заменено: {summary["counts"].get("applied", 0)}',
             f'- документов изменено: {summary["documents_changed"]}',
             f'- спорные варианты применялись: '
             f'{"да" if cfg.accept_unsure else "нет"}']
    if summary.get('outside_scope'):
        lines.append(f'- отброшено как не формулы по текущему отбору: '
                     f'{summary["outside_scope"]}')
    lines += ['', '## Решения', '',
              '| решение | формул |', '| --- | --- |']
    for decision in ('replace', 'unsure', 'keep'):
        lines.append(f'| {decision} | {summary["counts"].get(decision, 0)} |')

    by_reason = Counter(verdict['reason'] for verdict in verdicts)
    lines += ['', '## Причины', '', '| причина | формул |', '| --- | --- |']
    for reason, count in by_reason.most_common(20):
        lines.append(f'| {reason} | {count} |')

    for decision in ('replace', 'unsure', 'keep'):
        sample = [v for v in verdicts if v['decision'] == decision][:15]
        if not sample:
            continue
        lines += ['', f'## Примеры: {decision}', '']
        for verdict in sample:
            lines += [
                f'**{verdict["block_id"]}** (стр. {verdict["page"]}) — '
                f'{verdict["reason"]}',
                '',
                f'- было: `{_short(verdict["existing"])}`',
                f'- стало: `{_short(verdict["candidate"])}`',
                '',
            ]

    config.REPORT.parent.mkdir(parents=True, exist_ok=True)
    config.REPORT.write_text('\n'.join(lines), encoding='utf-8')


def _short(text, limit=160):
    text = (text or '').replace('`', "'").replace('\n', ' ')
    return text[:limit] + ('…' if len(text) > limit else '')
