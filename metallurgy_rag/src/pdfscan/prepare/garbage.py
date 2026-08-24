"""Сводная оценка «это шум распознавания, а не текст».

Один признак ошибается на любом корпусе: у химической записи мало гласных, у
таблицы короткие токены, у подписи к рисунку мало символов. Поэтому признаки
складываются с весами, а набор работающих признаков зависит от роли блока.
Формулы не оцениваются вовсе: по символьной статистике формула неотличима от
шума, и единственный надёжный признак для неё — откуда взят текст.

Оценка не удаляет ничего сама. Она пишется в таблицу вместе с исходными
признаками и с именем признака, который дал главный вклад, чтобы по отчёту было
видно не только «сколько выброшено», но и «за что».
"""

from . import config

# Границы, между которыми признак разворачивается из «нормально» в «шум».
# Нижняя взята по прозе корпуса, верхняя — по распознанным картинкам.
_BANDS = {
    'deviation': (-2.0, -6.0),
    'doubling_ratio': (0.08, 0.35),
    'consonant_run_max': (5.0, 9.0),
    'short_token_ratio': (0.35, 0.70),
    'nonalpha_ratio': (0.25, 0.60),
    'repeat_run_max': (4.0, 10.0),
}

# Правдоподобная доля гласных. Отклонение в любую сторону одинаково плохо:
# «бгдж» и «оооаа» оба не язык.
_VOWEL_BAND = (0.25, 0.55)
_VOWEL_SLOPE = 0.20

_WEIGHTS = {
    'deviation': 0.40,
    'doubling_ratio': 0.20,
    'short_token_ratio': 0.10,
    'vowel_ratio': 0.10,
    'consonant_run_max': 0.08,
    'nonalpha_ratio': 0.07,
    'repeat_run_max': 0.05,
}

_STRUCTURAL = frozenset(_WEIGHTS) - {'deviation'}


def _ramp(value, low, high) -> float:
    """Разворачивает признак в долю от нуля в ``low`` до единицы в ``high``."""
    if value is None:
        return 0.0
    if high == low:
        return 0.0
    return min(1.0, max(0.0, (value - low) / (high - low)))


def _vowel_badness(value) -> float:
    if value is None:
        return 0.0
    low, high = _VOWEL_BAND
    distance = max(low - value, value - high, 0.0)
    return min(1.0, distance / _VOWEL_SLOPE)


def enabled_signals(block_type: str, n_chars: int, cfg=config.DEFAULT) -> frozenset:
    """Какие признаки имеют смысл для блока такого типа и такой длины."""
    if block_type.startswith(config.FORMULA_PREFIX):
        return frozenset()

    if block_type in config.TABLE_TYPES:
        # Ячейки склеены в одну строку, поэтому короткие токены и знаки для
        # таблицы норма. Слова внутри ячеек модель всё ещё оценивает.
        allowed = frozenset({'deviation'})
    else:
        allowed = frozenset(_WEIGHTS)

    if n_chars < cfg.garbage_min_chars:
        # На коротком тексте символьная модель шумит сильнее, чем различает.
        allowed = allowed - {'deviation'}
    return allowed


def score(row: dict, cfg=config.DEFAULT) -> tuple:
    """Считает оценку шума и признак, давший главный вклад.

    Ожидает в ``row`` признаки из :func:`pipeline.textstats.signals` и поле
    ``deviation`` из символьной модели.
    """
    allowed = enabled_signals(row.get('type', ''), row.get('n_chars', 0), cfg)
    if not allowed:
        return 0.0, ''

    badness = {
        'deviation': _ramp(row.get('deviation'), *_BANDS['deviation']),
        'doubling_ratio': _ramp(row.get('doubling_ratio'), *_BANDS['doubling_ratio']),
        'short_token_ratio': _ramp(row.get('short_token_ratio'),
                                   *_BANDS['short_token_ratio']),
        'vowel_ratio': _vowel_badness(row.get('vowel_ratio')),
        'consonant_run_max': _ramp(row.get('consonant_run_max'),
                                   *_BANDS['consonant_run_max']),
        'nonalpha_ratio': _ramp(row.get('nonalpha_ratio'), *_BANDS['nonalpha_ratio']),
        'repeat_run_max': _ramp(row.get('repeat_run_max'), *_BANDS['repeat_run_max']),
    }

    contributions = {name: _WEIGHTS[name] * value
                     for name, value in badness.items() if name in allowed}
    total_weight = sum(_WEIGHTS[name] for name in contributions)
    if not total_weight:
        return 0.0, ''

    value = sum(contributions.values()) / total_weight
    reason = max(contributions, key=contributions.get)

    # Удвоение букв и длинные цепочки одного символа не бывают у текста ни в
    # каком языке, поэтому им разрешено перебить взвешенную сумму.
    if 'doubling_ratio' in allowed and (row.get('doubling_ratio') or 0) >= 0.30:
        if value < 0.90:
            value, reason = 0.90, 'doubling_ratio'
    if 'repeat_run_max' in allowed and (row.get('repeat_run_max') or 0) >= 12:
        if value < 0.85:
            value, reason = 0.85, 'repeat_run_max'

    return round(min(1.0, value), 4), reason


def is_garbage(row: dict, cfg=config.DEFAULT) -> bool:
    """Проверяет блок по порогу.

    Собранный из текстового слоя блок под подозрение не попадает: искажать его
    было нечему, а странная статистика там означает формулу или обозначения.
    """
    if row.get('reliable'):
        return False
    return (row.get('garbage_score') or 0.0) >= cfg.garbage_threshold
