"""Запуск стадий очистки из командной строки.

    python -m pdfscan.prepare.cli ingest   # out/**/blocks.jsonl → data/blocks_raw.parquet
    python -m pdfscan.prepare.cli clean    # → data/blocks_clean.parquet
    python -m pdfscan.prepare.cli audit    # → reports/clean_audit.md
    python -m pdfscan.prepare.cli all

Стадии разделены, потому что стоят они разного. Чтение двухсот JSONL повторять
при каждой правке порога незачем, а вот ``clean`` и ``audit`` после сдвига порога
пересчитываются часто.
"""

import argparse
from dataclasses import replace

from . import audit, clean, config, store


def _config_from_args(args):
    overrides = {name: value for name, value in (
        ('garbage_threshold', args.garbage_threshold),
        ('fragment_max_chars', args.fragment_max_chars),
        ('orphan_max_chars', args.orphan_max_chars),
    ) if value is not None}
    return replace(config.DEFAULT, **overrides) if overrides else config.DEFAULT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('ingest', 'clean', 'audit', 'all'))
    parser.add_argument('--pattern', default=None,
                        help='шаблон поиска blocks.jsonl; по умолчанию весь out/')
    parser.add_argument('--garbage-threshold', type=float, default=None,
                        help='порог оценки шума, по умолчанию '
                             f'{config.DEFAULT.garbage_threshold}')
    parser.add_argument('--fragment-max-chars', type=int, default=None)
    parser.add_argument('--orphan-max-chars', type=int, default=None)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    cfg = _config_from_args(args)
    progress = not args.quiet

    if args.stage in ('ingest', 'all'):
        store.ingest(args.pattern, progress=progress)

    if args.stage in ('clean', 'all'):
        clean.run(cfg=cfg, progress=progress)

    if args.stage in ('audit', 'all'):
        summary = audit.build(cfg=cfg)
        if progress:
            print(f'осталось {summary["blocks_out"]} блоков из '
                  f'{summary["blocks_in"]} → {config.AUDIT_MD}')


if __name__ == '__main__':
    main()
