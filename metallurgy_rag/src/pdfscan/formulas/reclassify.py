"""Пересчёт подтипа формулы после того, как её текст заменили.

Повторное распознавание меняет текст блока, но не его метку, а метка решает
судьбу блока при индексации: символы элементов и их русские названия
добавляются только там, где в типе стоит ``chemistry``. Формула, от которой в
разборе осталось «Cu?*(aq)» и которая поэтому получила тип без подтипа, после
замены становится нормальным уравнением — и должна перестать быть безымянной,
иначе запрос «медь» её не найдёт.

Пересчитываются только те блоки, текст которых действительно изменился. У
остальных исходные данные ровно те же, что при разборе, и менять их разметку
значило бы расходиться с корпусом без причины.

Порядок решения повторяет разбор: сначала грамматика химического уравнения,
потом модель. Это не косметика — правило и модель расходятся на пограничных
записях, и если здесь спросить сначала модель, часть уравнений получит другой
подтип, чем такие же уравнения в блоках, которых распознавание не касалось.
"""

import json
from collections import Counter
from pathlib import Path

from pdfscan import paths
from pdfscan.formulas.features import looks_like_chemical_equation
from pdfscan.formulas.ocr import apply as ocr_apply

CHEMISTRY_TYPE = 'Formula (chemistry)'


def read_blocks(path) -> list:
    blocks = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                blocks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return blocks


def text_was_replaced(block: dict) -> bool:
    """Заменяло ли распознавание текст этого блока."""
    trace = block.get('formula_ocr')
    return bool(trace) and bool(trace.get('applied'))


def context_around(blocks: list, index: int) -> str:
    """Соседний текст, которым классификатор пользовался при разборе.

    Два блока до и до пяти после, с остановкой на первом длинном: подпись под
    формулой обычно объясняет обозначения, и дальше неё смотреть незачем. Здесь
    соседи берутся по порядку в файле — это ближайшее, что есть к исходному
    порядку элементов на странице.
    """
    before = [str(blocks[j].get('text') or '')
              for j in range(max(0, index - 2), index)]

    after = []
    for j in range(index + 1, min(len(blocks), index + 6)):
        text = str(blocks[j].get('text') or '').strip()
        if not text:
            continue
        after.append(text)
        if len(text) > 250:
            break

    return ' '.join(before + after).strip()


def collect(blocks_files, progress=True) -> tuple:
    """Первый проход: правило решает сразу, остальное собирается для модели.

    Возвращает ``(решённое правилом, задания для модели)``. Модель вызывается
    одной пачкой на весь корпус, а не по документу: загрузка BERT стоит куда
    дороже самого предсказания.
    """
    decided, jobs = {}, []

    for position, path in enumerate(blocks_files, 1):
        blocks = read_blocks(path)
        changed = 0
        for index, block in enumerate(blocks):
            if not text_was_replaced(block):
                continue
            changed += 1
            key = (str(path), block['block_id'])
            text = block.get('text') or ''
            if looks_like_chemical_equation(text):
                decided[key] = CHEMISTRY_TYPE
                continue
            jobs.append({
                'key': key,
                'text': text,
                'context': context_around(blocks, index),
                'layout': block.get('layout'),
            })
        if progress and changed:
            print(f'[{position}/{len(blocks_files)}] {path.parent.name}: '
                  f'изменённых формул {changed}')

    return decided, jobs


def predict(jobs: list, model, progress=True) -> dict:
    """Второй проход: подтип от модели для всего, что не решило правило."""
    if not jobs or model is None:
        return {}
    if progress:
        print(f'модель определяет подтип для {len(jobs)} формул')
    predictions = model.predict_batch(
        [job['text'] for job in jobs],
        [job['context'] for job in jobs],
        [job['layout'] for job in jobs],
    )
    return {job['key']: f'Formula ({name})'
            for job, name in zip(jobs, predictions)}


def write_back(blocks_files, types: dict, progress=True) -> Counter:
    """Третий проход: запись новых меток и запись о том, что метка менялась."""
    moves = Counter()

    for path in blocks_files:
        blocks = read_blocks(path)
        changed = False
        for block in blocks:
            new_type = types.get((str(path), block.get('block_id')))
            if new_type is None:
                continue
            old_type = block.get('type')
            if new_type == old_type:
                moves[f'{old_type} → без изменений'] += 1
                continue

            block['type'] = new_type
            # След остаётся там же, где вся история правки блока: через месяц
            # вопрос «почему у этой формулы такой тип» должен иметь ответ.
            trace = block.get('formula_ocr')
            if isinstance(trace, dict):
                trace['type_before'] = old_type
                trace['type_after'] = new_type
            moves[f'{old_type} → {new_type}'] += 1
            changed = True

        if changed:
            ocr_apply.backup_once(Path(path))
            ocr_apply.write_atomic(Path(path), blocks)
            if progress:
                print(f'  переписан {Path(path).parent.name}')

    return moves


def run(blocks_files=None, model=None, progress=True) -> dict:
    """Пересчитывает подтип у блоков, которым распознавание сменило текст."""
    blocks_files = list(blocks_files or paths.blocks_files())
    decided, jobs = collect(blocks_files, progress)

    if jobs and model is None:
        from pdfscan.formulas.classifier import FormulaClassifier
        model = FormulaClassifier.load()
        if model is None:
            raise SystemExit(
                'классификатор не загрузился: нет '
                f'{paths.FORMULA_CLASSIFIER} или устарел его формат')

    from_model = predict(jobs, model, progress)
    types = {**decided, **from_model}
    moves = write_back(blocks_files, types, progress)

    summary = {
        'candidates': len(types),
        'by_rule': len(decided),
        'by_model': len(from_model),
        'moves': dict(moves),
        'changed': sum(count for move, count in moves.items()
                       if 'без изменений' not in move),
    }
    if progress:
        print(f'\nпересчитано меток: {summary["candidates"]} (правилом '
              f'{summary["by_rule"]}, моделью {summary["by_model"]})')
        print(f'метка изменилась у {summary["changed"]} формул')
        for move, count in moves.most_common():
            print(f'    {move}: {count}')
    return summary
