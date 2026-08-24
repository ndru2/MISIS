"""Символьная модель языка для оценки правдоподобия текста.

Модель учится на самом корпусе: берётся только та проза, что собрана из
текстового слоя PDF, то есть заведомо не искажена распознаванием. Получается
мера «так в этих документах пишут» — без внешнего словаря, одинаково рабочая
для русского и английского и не спотыкающаяся на металлургических терминах.

Оценка одна на текст: средний логарифм вероятности символа при двух
предыдущих. Абсолютное значение ни о чём не говорит, поэтому оно приводится к
отклонению от разброса на отложенной прозе — так порог читается в единицах
«насколько это хуже обычного текста».
"""

import numpy as np

from . import config, textstats

V = textstats.ALPHABET_SIZE
_FLUSH_CHARS = 4_000_000


def _flush(buffer, counts):
    """Досчитывает триграммы накопленного куска текста."""
    joined = b''.join(buffer)
    if len(joined) < 3:
        return
    arr = np.frombuffer(joined, dtype=np.uint8).astype(np.int64)
    trigrams = arr[:-2] * V * V + arr[1:-1] * V + arr[2:]
    counts += np.bincount(trigrams, minlength=V ** 3)


def _count(texts) -> np.ndarray:
    """Считает триграммы по потоку текстов.

    Тексты разделяются граничным символом: без него конец одного блока склеился
    бы с началом другого и породил переходы, которых в языке нет.
    """
    counts = np.zeros(V ** 3, dtype=np.int64)
    buffer, size = [], 0
    boundary = bytes([textstats.BOUNDARY_INDEX])

    for text in texts:
        encoded = textstats.encode(text)
        buffer.append(boundary)
        buffer.append(encoded)
        size += len(encoded) + 1
        if size >= _FLUSH_CHARS:
            _flush(buffer, counts)
            buffer, size = [], 0

    if buffer:
        _flush(buffer, counts)
    return counts


def _log_probabilities(counts: np.ndarray, smoothing: float) -> np.ndarray:
    """Переводит счётчики в логарифмы условных вероятностей."""
    cube = counts.reshape(V, V, V).astype(np.float64)
    context = cube.sum(axis=2, keepdims=True)
    probabilities = (cube + smoothing) / (context + smoothing * V)
    return np.log(probabilities).astype(np.float32).ravel()


class CharLM:
    """Триграммная модель символов, отдельная для каждого письма."""

    def __init__(self, log_probabilities: dict, baselines: dict):
        self.log_probabilities = log_probabilities
        self.baselines = baselines

    @classmethod
    def train(cls, samples, smoothing=config.LM_SMOOTHING, progress=True):
        """Учит модель по парам ``(письмо, текст)``.

        Часть примеров откладывается, чтобы измерить разброс оценки на
        нормальной прозе: без него порог пришлось бы задавать в логарифмах,
        которые ни о чём не говорят.
        """
        train_texts, holdout_texts = {}, {}
        for text_script, text in samples:
            bucket = (holdout_texts
                      if _is_holdout(text) else train_texts)
            bucket.setdefault(text_script, []).append(text)

        log_probabilities, baselines = {}, {}
        for text_script, texts in train_texts.items():
            if progress:
                print(f'  символьная модель [{text_script}]: {len(texts)} блоков')
            log_probabilities[text_script] = _log_probabilities(
                _count(texts), smoothing)

        model = cls(log_probabilities, {})
        for text_script in log_probabilities:
            scores = [model.raw_score(text, text_script)
                      for text in holdout_texts.get(text_script, ())]
            scores = [s for s in scores if s is not None]
            if len(scores) < 20:
                # Отложенной выборки не хватило — приводить оценку не к чему,
                # и лучше честно оставить нулевое отклонение, чем выдумать
                # разброс по десятку блоков.
                baselines[text_script] = (0.0, 1.0)
                continue
            values = np.asarray(scores, dtype=np.float64)
            baselines[text_script] = (float(values.mean()),
                                      float(values.std() or 1.0))
            if progress:
                mean, std = baselines[text_script]
                print(f'  разброс на отложенной прозе [{text_script}]: '
                      f'{mean:.3f} ± {std:.3f}')

        model.baselines = baselines
        return model

    def raw_score(self, text: str, text_script=None):
        """Средний логарифм вероятности символа; ``None``, если текст короче триграммы."""
        text_script = text_script or textstats.script(text)
        table = self.log_probabilities.get(text_script)
        if table is None:
            return None
        arr = np.frombuffer(textstats.encode(text), dtype=np.uint8).astype(np.int64)
        if arr.size < 3:
            return None
        trigrams = arr[:-2] * V * V + arr[1:-1] * V + arr[2:]
        return float(table[trigrams].mean())

    def deviation(self, text: str, text_script=None):
        """Насколько текст хуже обычной прозы, в единицах её разброса.

        Ноль — как в среднем по корпусу, минус три — заметно непохоже на язык.
        """
        text_script = text_script or textstats.script(text)
        score = self.raw_score(text, text_script)
        if score is None:
            return None
        mean, std = self.baselines.get(text_script, (0.0, 1.0))
        return (score - mean) / std

    def save(self, path=None):
        path = path or config.CHARLM_MODEL
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f'logp_{k}': v for k, v in self.log_probabilities.items()}
        arrays.update({f'base_{k}': np.asarray(v, dtype=np.float64)
                       for k, v in self.baselines.items()})
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path=None):
        path = path or config.CHARLM_MODEL
        with np.load(path) as data:
            log_probabilities = {k[len('logp_'):]: data[k]
                                 for k in data.files if k.startswith('logp_')}
            baselines = {k[len('base_'):]: tuple(data[k])
                         for k in data.files if k.startswith('base_')}
        return cls(log_probabilities, baselines)


def _is_holdout(text: str) -> bool:
    """Делит выборку устойчиво: один и тот же текст всегда попадает в ту же часть."""
    digest = textstats.content_hash(text)
    return int(digest[:4], 16) / 0xFFFF < config.LM_HOLDOUT_SHARE


def training_samples(rows):
    """Отбирает из блоков связную прозу, собранную из текстового слоя."""
    for row in rows:
        if not row.get('reliable'):
            continue
        if row.get('type') not in config.LM_TRAIN_TYPES:
            continue
        text = row.get('text') or ''
        if len(text) < config.LM_MIN_CHARS:
            continue
        text_script = textstats.script(text)
        if text_script == 'none':
            continue
        yield text_script, text
