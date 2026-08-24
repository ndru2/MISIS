"""Распознавание формул моделью PP-FormulaNet.

Модель импортируется внутри функции, а не сверху файла. Это не стилистика:
paddle несовместим с тем стеком, на котором работает разбор PDF, поэтому модуль
живёт в отдельном окружении, и остальной проект должен уметь читать результаты
его работы, не имея paddle установленным вовсе.

Результат дописывается в файл построчно и сразу сбрасывается на диск. Прогон по
корпусу идёт часами, и обрыв не должен стоить всей работы: при повторном запуске
уже распознанные картинки пропускаются.
"""

import json

from . import config


def _load_done(path) -> set:
    """Картинки, для которых результат уже есть."""
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)['crop'])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _extract_latex(result):
    """Достаёт разметку из ответа модели.

    Ответ приходит объектом paddlex, у которого структура словаря меняется между
    версиями, поэтому доступ защищён: пустой результат обрабатывается дальше как
    «кандидата нет», а исключение уронило бы весь прогон.
    """
    try:
        payload = result.json.get('res', {})
    except Exception:
        return None, None
    latex = payload.get('rec_formula')
    score = payload.get('rec_score') or payload.get('score')
    return latex, score


def run(manifest_path=None, out_path=None, cfg=config.DEFAULT, progress=True):
    """Распознаёт формулы из манифеста, продолжая прерванный прогон."""
    manifest_path = manifest_path or config.MANIFEST
    out_path = out_path or config.RECOGNIZED

    with open(manifest_path, encoding='utf-8') as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    done = _load_done(out_path)
    pending = [record for record in records if record['crop'] not in done]
    if progress:
        print(f'формул в манифесте: {len(records)}, '
              f'уже распознано: {len(done)}, к работе: {len(pending)}')
    if not pending:
        return out_path

    from paddlex import create_model

    if progress:
        print(f'загружаю модель {cfg.model_name}')
    model = create_model(model_name=cfg.model_name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'a', encoding='utf-8') as handle:
        stream = model.predict(input=[record['crop'] for record in pending],
                               batch_size=cfg.batch_size)
        for position, (record, result) in enumerate(zip(pending, stream), 1):
            latex, score = _extract_latex(result)
            handle.write(json.dumps({
                **record,
                'latex': latex,
                'model_score': score,
                'model': cfg.model_name,
            }, ensure_ascii=False) + '\n')
            handle.flush()
            if progress and position % 50 == 0:
                print(f'  распознано {position}/{len(pending)}')

    if progress:
        print(f'готово → {out_path}')
    return out_path
