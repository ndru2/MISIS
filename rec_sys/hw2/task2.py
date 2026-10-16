from dataclasses import dataclass
from itertools import combinations

import numpy as np


DEFAULT_MATRIX = np.array([
    [0, .95, .90, .85, .80],
    [.95, 0, .75, .70, .65],
    [.90, .75, 0, .60, .80],
    [.85, .70, .60, 0, .70],
    [.80, .65, .80, .70, 0],
])

Cluster = tuple[int, ...]


@dataclass(frozen=True)
class MergeStep:
    left: Cluster
    right: Cluster
    similarity: float
    clusters: tuple[Cluster, ...]
    matrix: np.ndarray


@dataclass(frozen=True)
class ClusteringResult:
    initial_matrix: np.ndarray
    threshold: float
    steps: tuple[MergeStep, ...]
    clusters: tuple[Cluster, ...]
    next_similarity: float | None


def cluster_name(cluster: Cluster) -> str:
    return "+".join(f"U{index + 1}" for index in cluster)


def validate_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Матрица сходства должна быть квадратной.")
    if len(matrix) < 2:
        raise ValueError("Нужно минимум два объекта.")
    if not np.isfinite(matrix).all():
        raise ValueError("Матрица содержит пустые или нечисловые значения.")
    if np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("Косинусное подобие должно быть в диапазоне от 0 до 1.")
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("Матрица сходства должна быть симметричной.")
    if np.any(np.diag(matrix) != 0):
        raise ValueError("На диагонали должны стоять нули: объект не объединяется сам с собой.")


def cluster_matrix(source: np.ndarray, clusters: tuple[Cluster, ...]) -> np.ndarray:
    """Сходство двух групп — максимум сходств всех пар их объектов."""
    scores = np.zeros((len(clusters), len(clusters)))
    for i, j in combinations(range(len(clusters)), 2):
        scores[i, j] = scores[j, i] = source[np.ix_(clusters[i], clusters[j])].max()
    return scores


def agglomerate(matrix, threshold: float = .85) -> ClusteringResult:
    """Объединять наиболее похожие группы, пока максимум >= R.

    При равных максимумах выбирается первая пара по порядку строк.
    При R=0 все объекты в итоге попадают в одну группу.
    Входная матрица и снимки предыдущих шагов не изменяются.
    """
    source = np.array(matrix, dtype=float, copy=True)
    validate_matrix(source)
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Порог R должен быть числом от 0 до 1.")

    clusters = tuple((i,) for i in range(len(source)))
    history = []
    current = source.copy()
    next_similarity = None

    while len(clusters) > 1:
        pairs = list(combinations(range(len(clusters)), 2))
        i, j = max(pairs, key=lambda pair: current[pair])
        score = float(current[i, j])
        if score < threshold:
            next_similarity = score
            break

        left, right = clusters[i], clusters[j]
        merged = tuple(sorted(left + right))
        clusters = tuple(
            merged if k == i else group
            for k, group in enumerate(clusters) if k != j
        )
        # Пересчёт из исходных данных исключает зависимость от изменяемых снимков.
        current = cluster_matrix(source, clusters)
        history.append(MergeStep(left, right, score, clusters, current))

    return ClusteringResult(source, threshold, tuple(history), clusters, next_similarity)


def main():
    result = agglomerate(DEFAULT_MATRIX)
    for number, step in enumerate(result.steps, 1):
        print(f"Шаг {number}: {cluster_name(step.left)} + {cluster_name(step.right)}"
              f"; сходство = {step.similarity:g}")
        print(step.matrix)
    print("Итог:", "; ".join(cluster_name(group) for group in result.clusters))


if __name__ == "__main__":
    main()
