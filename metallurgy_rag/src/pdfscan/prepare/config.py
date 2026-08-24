"""Пути и пороги отбора.

Пороги собраны в одном месте не для красоты: они подбираются глазами по
аудит-отчёту, и когда цифра встречается в коде дважды, вторая рано или поздно
остаётся старой.
"""

from dataclasses import dataclass, field

from pdfscan import paths

DATA_DIR = paths.DATA_DIR
REPORTS_DIR = paths.REPORTS_DIR

RAW_BLOCKS = DATA_DIR / 'blocks_raw.parquet'
CLEAN_BLOCKS = DATA_DIR / 'blocks_clean.parquet'
CHARLM_MODEL = DATA_DIR / 'charlm.npz'
EXTRAS = DATA_DIR / 'clean_extras.json'

AUDIT_MD = REPORTS_DIR / 'clean_audit.md'
AUDIT_JSON = REPORTS_DIR / 'clean_audit.json'

# Роль блока решает, какими сигналами его можно мерить. Формула на любом
# счётчике «похоже на текст» выглядит мусором, потому что она мусором и должна
# выглядеть: там мало гласных, много символов и короткие токены.
FORMULA_PREFIX = 'Formula'
TABLE_TYPES = frozenset({'Table', 'TableOfContents'})
VISUAL_TYPES = frozenset({'Image', 'Figure'})
FURNITURE_TYPES = frozenset({'Header', 'Footer', 'PageBreak'})
PROSE_TYPES = frozenset({
    'NarrativeText', 'ListItem', 'UncategorizedText', 'Title',
    'Caption', 'FigureCaption',
})

# Текст, на котором учится символьная модель: только собранный из текстового
# слоя и только связная проза. Всё остальное либо распознано, либо слишком
# короткое, чтобы отражать статистику языка.
LM_TRAIN_TYPES = frozenset({'NarrativeText', 'ListItem'})
LM_MIN_CHARS = 200
LM_HOLDOUT_SHARE = 0.1
LM_SMOOTHING = 0.1


@dataclass
class CleanConfig:
    """Пороги очистки."""

    # Мусор. Порог подбирается по выборкам из отчёта: в нём печатаются примеры
    # по полосам оценки, и видно, где начинается нечитаемое. На этом корпусе
    # нечитаемое начинается раньше 0.60 — полоса 0.45–0.60 почти целиком занята
    # распознаванием перевёрнутых и повёрнутых страниц, где кириллица выходит
    # зеркальной кашей вроде «излоонжомсов иеиненосчноной». Из выборки в
    # пятнадцать блоков связным оказался один, поэтому граница опущена до 0.45.
    garbage_threshold: float = 0.45
    # Короткому тексту символьная модель не верит: на двадцати символах
    # разброс оценки больше, чем разница между прозой и мусором.
    garbage_min_chars: int = 30

    # Колонтитул: одна и та же строка на многих страницах одного документа.
    # Три страницы — минимум, ниже которого совпадение может быть случайным.
    boilerplate_min_pages: int = 3
    boilerplate_min_share: float = 0.15
    boilerplate_max_chars: int = 120

    # Склейка фрагментов. Порог длины отделяет обрывок строки от короткого,
    # но осмысленного блока вроде подписи к рисунку.
    fragment_max_chars: int = 25
    # Зазоры считаются в долях высоты строки на странице, а не в пунктах:
    # у книги и у статьи кегль разный.
    same_line_overlap: float = 0.5
    same_line_gap_chars: float = 2.5
    paragraph_gap_lines: float = 2.0
    paragraph_x_overlap: float = 0.3

    # Осиротевший обрывок, который не удалось ни к чему приклеить.
    orphan_max_chars: int = 15

    # Точный дубль блока внутри документа. Короткие строки не берём: там
    # совпадения законны, и с ними разбирается детектор колонтитулов.
    duplicate_min_chars: int = 40

    # Похожесть документов. Ниже верхнего порога — только сообщаем: сборник
    # симпозиума и отдельная статья из него совпадают частично, и что из них
    # оставить, решает человек. Выше — это буквально один файл, разложенный по
    # двум тематическим папкам, и вторая копия отбрасывается сама.
    doc_similarity_report: float = 0.5
    doc_duplicate_drop: float = 0.99
    minhash_permutations: int = 128
    minhash_shingle_words: int = 5
    minhash_keep_every: int = 8

    # Сколько примеров показывать в отчёте на каждую причину отброса.
    audit_samples: int = 20
    audit_sample_chars: int = 140

    drop_reasons: tuple = field(default=(
        'empty', 'garbage', 'boilerplate', 'merged', 'duplicate', 'orphan',
        'duplicate_document',
    ))


DEFAULT = CleanConfig()
