"""Пути проекта в одном месте.

Раньше `formula_classifier.pkl`, `out/` и `rag_index` были вписаны строками в
десяток модулей, и любой перенос каталога означал охоту за ними по всему коду.
Здесь эти имена названы один раз, а переопределяются переменными окружения —
это то, что нужно и для Colab, где корень лежит на Google Диске, и для CI, где
данных нет вовсе.
"""

import os
from pathlib import Path

# src/pdfscan/paths.py → корень проекта на два уровня выше
ROOT = Path(os.environ.get('PDFSCAN_ROOT',
                           Path(__file__).resolve().parents[2]))

PDF_DIR = Path(os.environ.get('PDFSCAN_PDF_DIR', ROOT / 'pdf'))
OUT_DIR = Path(os.environ.get('PDFSCAN_OUT_DIR', ROOT / 'out'))
DATA_DIR = Path(os.environ.get('PDFSCAN_DATA_DIR', ROOT / 'data'))
MODELS_DIR = Path(os.environ.get('PDFSCAN_MODELS_DIR', ROOT / 'models'))
REPORTS_DIR = Path(os.environ.get('PDFSCAN_REPORTS_DIR', ROOT / 'reports'))
DATASETS_DIR = Path(os.environ.get('PDFSCAN_DATASETS_DIR', ROOT / 'datasets'))

# Разбор PDF: одна папка на документ, повторяющая дерево исходников.
BLOCKS_NAME = 'blocks.jsonl'
BLOCKS_GLOB = f'**/{BLOCKS_NAME}'

FORMULA_CLASSIFIER = MODELS_DIR / 'formula_classifier.pkl'
HARVESTED_DATASET = DATASETS_DIR / 'harvested_dataset.csv'
RAG_INDEX_DIR = Path(os.environ.get('PDFSCAN_INDEX_DIR', ROOT / 'rag_index'))


def blocks_files(root=None):
    """Все ``blocks.jsonl`` в порядке, устойчивом между запусками."""
    root = Path(root or OUT_DIR)
    return sorted(root.rglob(BLOCKS_NAME), key=str)
