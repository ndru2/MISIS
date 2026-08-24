"""Поиск повторов: точных внутри документа и близких между документами.

Внутри документа повтор длинного блока почти всегда означает, что страница
разобрана дважды, и второй экземпляр в индексе только вытесняет из выдачи
что-то полезное.

Между документами история другая, и в ней два разных случая. Один и тот же файл,
разложенный по двум тематическим папкам, — это чистый повтор, и вторая копия в
индексе только занимает место в выдаче; такие пары отбрасываются сами. Сборник
симпозиума и отдельная статья из него совпадают лишь частично, попадают в
середину шкалы похожести и остаются: там решать человеку, потому что у сборника
и оттиска разная полнота. Порог между этими случаями стоит почти на единице
именно поэтому.

Похожесть считается по MinHash от пятисловных дробей. Точные множества дробей
для двух сотен документов пришлось бы сравнивать почти двадцать три тысячи раз,
а подписи фиксированной длины сводят это к одному матричному сравнению. Заодно
подпись — единственное, что приходится хранить от документа при потоковой
обработке: сто двадцать восемь чисел вместо всего текста.
"""

import zlib

import numpy as np

from . import config, textstats

# Модуль Мерсенна: подписи считаются в uint64, и при 32-битных дробях
# произведение на множитель до 2^31 в разрядность укладывается.
_PRIME = np.uint64((1 << 31) - 1)
_MIX = np.uint64(1000003)
_MASK = np.uint64(0xFFFFFFFF)


def exact_duplicates(rows, cfg=config.DEFAULT) -> int:
    """Помечает точные повторы блоков внутри документа.

    Короткие строки не рассматриваются: их совпадение законно, и с ними
    разбирается детектор колонтитулов.
    """
    seen = {}
    dropped = 0
    for row in rows:
        if not row['keep']:
            continue
        text = row.get('text_out') or ''
        if len(text) < cfg.duplicate_min_chars:
            continue
        key = textstats.content_hash(text)
        first = seen.get(key)
        if first is None:
            seen[key] = row['block_id']
            continue
        row['keep'] = False
        row['drop_reason'] = 'duplicate'
        row['merged_into'] = first
        dropped += 1
    return dropped


def make_coefficients(cfg=config.DEFAULT) -> tuple:
    """Коэффициенты перестановок MinHash.

    Зерно фиксировано: без него один и тот же корпус давал бы разные оценки
    похожести от запуска к запуску, и отчёт нельзя было бы сравнить с прошлым.
    """
    generator = np.random.default_rng(0)
    size = cfg.minhash_permutations
    return (generator.integers(1, int(_PRIME), size=size, dtype=np.uint64),
            generator.integers(0, int(_PRIME), size=size, dtype=np.uint64))


def _shingles(text: str, cfg) -> np.ndarray:
    """Хеши пятисловных дробей текста.

    Слова хешируются по одному, а дроби собираются свёрткой на массивах: прямое
    склеивание строк на семи миллионах слов корпуса заняло бы минуты.
    """
    words = text.split()
    size = cfg.minhash_shingle_words
    if len(words) < size:
        return np.empty(0, dtype=np.uint64)

    hashes = np.fromiter(
        (zlib.crc32(word.encode('utf-8')) for word in words),
        dtype=np.uint64, count=len(words))

    total = len(words) - size + 1
    shingles = np.zeros(total, dtype=np.uint64)
    for offset in range(size):
        shingles = (shingles * _MIX + hashes[offset:offset + total]) & _MASK

    # Прореживание оставляет ту же долю дробей у обоих документов, поэтому
    # оценка похожести не смещается, а работы становится в разы меньше.
    keep = np.uint64(cfg.minhash_keep_every)
    return np.unique(shingles[shingles % keep == np.uint64(0)])


def document_signature(texts, coefficients, cfg=config.DEFAULT) -> tuple:
    """MinHash-подпись документа и число учтённых дробей."""
    shingles = _shingles(textstats.norm_key(' '.join(texts)), cfg)
    multipliers, offsets = coefficients
    if shingles.size == 0:
        return np.full(multipliers.size, _PRIME, dtype=np.uint64), 0
    permuted = (multipliers[:, None] * shingles[None, :] + offsets[:, None]) % _PRIME
    return permuted.min(axis=1), int(shingles.size)


def similar_pairs(doc_ids, signatures, sizes, cfg=config.DEFAULT) -> list:
    """Находит пары документов с близким содержимым по готовым подписям."""
    usable = [index for index, doc_id in enumerate(doc_ids) if sizes[doc_id] > 0]
    if len(usable) < 2:
        return []

    doc_ids = [doc_ids[index] for index in usable]
    matrix = np.vstack([signatures[index] for index in usable])

    pairs = []
    for first in range(len(doc_ids)):
        matches = (matrix[first + 1:] == matrix[first]).mean(axis=1)
        for offset, similarity in enumerate(matches):
            if similarity >= cfg.doc_similarity_report:
                second = first + 1 + offset
                pairs.append({
                    'left': doc_ids[first],
                    'right': doc_ids[second],
                    'similarity': round(float(similarity), 3),
                    'left_shingles': sizes[doc_ids[first]],
                    'right_shingles': sizes[doc_ids[second]],
                })

    pairs.sort(key=lambda item: -item['similarity'])
    return pairs


def _rank(doc_id: str, kept_blocks: dict) -> tuple:
    """Чем меньше кортеж, тем больше у копии прав остаться.

    Первым идёт глубина вложенности: один и тот же файл раскладывают по темам, и
    экземпляр ближе к корню — тот, от которого делали копии. Дальше — число
    уцелевших блоков, на случай если одна из копий разобрана хуже, и наконец
    само имя, чтобы выбор не зависел от порядка обхода папок.
    """
    return (doc_id.count('/'), -kept_blocks.get(doc_id, 0), doc_id)


def duplicate_documents(pairs, kept_blocks: dict, cfg=config.DEFAULT) -> dict:
    """Какие копии документов отбросить: ``{отброшенный: оставленный}``.

    Пары связываются в группы транзитивно: один и тот же отчёт встречается в
    корпусе трижды, и попарного решения мало — иначе из трёх копий выживут две.
    """
    neighbours = {}
    for pair in pairs:
        if pair['similarity'] < cfg.doc_duplicate_drop:
            continue
        neighbours.setdefault(pair['left'], set()).add(pair['right'])
        neighbours.setdefault(pair['right'], set()).add(pair['left'])

    dropped, seen = {}, set()
    for start in sorted(neighbours):
        if start in seen:
            continue
        group, queue = set(), [start]
        while queue:
            doc_id = queue.pop()
            if doc_id in group:
                continue
            group.add(doc_id)
            queue.extend(neighbours.get(doc_id, ()))
        seen |= group

        winner = min(group, key=lambda doc_id: _rank(doc_id, kept_blocks))
        for doc_id in group - {winner}:
            dropped[doc_id] = winner

    return dropped
