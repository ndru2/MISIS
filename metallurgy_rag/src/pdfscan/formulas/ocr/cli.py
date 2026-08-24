"""Повторное распознавание формул: запуск из командной строки.

Распознавание живёт в отдельном окружении, потому что paddle не сосуществует с
тем стеком, на котором работает разбор PDF (порядок установки — в
``requirements/formula-ocr.txt``). Из четырёх шагов paddle нужен только одному:

    venv/bin/python -m pdfscan.formulas.ocr.cli crop
    venv-formula/bin/python -m pdfscan.formulas.ocr.cli recognize
    venv/bin/python -m pdfscan.formulas.ocr.cli apply --dry-run
    venv/bin/python -m pdfscan.formulas.ocr.cli apply
    venv/bin/python -m pdfscan.formulas.ocr.cli reclassify

Вырезка обходится ``pypdfium2`` и ``pillow``, а отбор, оценка и запись обратно —
стандартной библиотекой, поэтому всё, кроме ``recognize``, запускается из
основного окружения.

Последний шаг пересчитывает подтип формулы там, где текст сменился: метка
осталась от прежней, нечитаемой записи, а по ней при индексации решается, искать
ли в блоке символы элементов.

Шаги разделены не для красоты. Вырезка идёт минуты, распознавание — часы, и при
подборе порогов заново нужен только последний шаг. ``apply --dry-run`` считает
решения и пишет отчёт, ничего не меняя в файлах: посмотреть примеры замен до
того, как они попадут в корпус, здесь важнее, чем сэкономить один запуск.

    venv/bin/python -m pdfscan.formulas.ocr.cli revert

вернёт исходный разбор из копий ``blocks.jsonl.orig``.
"""

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

from pdfscan import paths

from . import apply, config, crops, recognize


def _config_from_args(args):
    overrides = {name: value for name, value in (
        ('render_dpi', getattr(args, 'dpi', None)),
        ('padding_pt', getattr(args, 'padding', None)),
        ('batch_size', getattr(args, 'batch_size', None)),
        ('model_name', getattr(args, 'model', None)),
    ) if value is not None}
    if getattr(args, 'accept_unsure', False):
        overrides['accept_unsure'] = True
    return replace(config.DEFAULT, **overrides) if overrides else config.DEFAULT


def _blocks_files(args):
    files = paths.blocks_files(args.out) if args.out else paths.blocks_files()
    if not files:
        raise SystemExit(f'не найдено ни одного {paths.BLOCKS_NAME} в '
                         f'{args.out or paths.OUT_DIR}')
    return files[:args.limit] if args.limit else files


def command_crop(args):
    crops.build_manifest(_blocks_files(args), _config_from_args(args))


def command_recognize(args):
    if not config.MANIFEST.exists():
        raise SystemExit('нет манифеста; сначала: cli crop')
    recognize.run(cfg=_config_from_args(args))


def command_apply(args):
    if not config.RECOGNIZED.exists():
        raise SystemExit('нет результатов распознавания; сначала: cli recognize')
    cfg = _config_from_args(args)
    if args.dry_run:
        apply.dry_run(cfg=cfg)
    else:
        apply.run(cfg=cfg)


def command_reclassify(args):
    """Пересчёт подтипа у блоков, которым замена сменила текст.

    Импорт внутри функции по той же причине, что и у paddle, только наоборот:
    классификатор тянет torch и transformers, которых в формульном окружении
    нет и быть не должно.
    """
    from pdfscan.formulas import reclassify
    reclassify.run(_blocks_files(args))


def command_run(args):
    command_crop(args)
    command_recognize(args)
    command_apply(args)


def command_revert(args):
    """Возвращает исходный разбор из копий, сделанных перед правкой."""
    restored = 0
    for blocks_path in _blocks_files(args):
        backup = Path(str(blocks_path) + config.BACKUP_SUFFIX)
        if backup.exists():
            shutil.copy2(backup, blocks_path)
            restored += 1
    print(f'восстановлено документов: {restored}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=None, help='корень разбора, по умолчанию out/')
    parser.add_argument('--limit', type=int, default=None,
                        help='взять только первые N документов — для пробы')
    sub = parser.add_subparsers(dest='step', required=True)

    crop = sub.add_parser('crop', help='вырезать картинки формул по рамкам')
    crop.add_argument('--dpi', type=int, default=None)
    crop.add_argument('--padding', type=float, default=None)
    crop.set_defaults(func=command_crop)

    rec = sub.add_parser('recognize', help='распознать вырезанные формулы')
    rec.add_argument('--model', default=None)
    rec.add_argument('--batch-size', type=int, default=None)
    rec.set_defaults(func=command_recognize)

    app = sub.add_parser('apply', help='записать удачные варианты в blocks.jsonl')
    app.add_argument('--dry-run', action='store_true',
                     help='посчитать решения и отчёт, файлы не менять')
    app.add_argument('--accept-unsure', action='store_true',
                     help='применять и спорные варианты')
    app.set_defaults(func=command_apply)

    again = sub.add_parser(
        'reclassify', help='пересчитать подтип у заменённых формул')
    again.set_defaults(func=command_reclassify)

    whole = sub.add_parser('run', help='все три шага подряд')
    whole.add_argument('--dpi', type=int, default=None)
    whole.add_argument('--padding', type=float, default=None)
    whole.add_argument('--model', default=None)
    whole.add_argument('--batch-size', type=int, default=None)
    whole.add_argument('--accept-unsure', action='store_true')
    whole.add_argument('--dry-run', action='store_true')
    whole.set_defaults(func=command_run)

    back = sub.add_parser('revert', help='вернуть исходный разбор из копий')
    back.set_defaults(func=command_revert)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
