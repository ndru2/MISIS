import numpy as np
from itertools import combinations
from pathlib import Path


PATH_DATA = Path(__file__).with_name("task1.txt")


def parse_matrix_text(text: str) -> np.ndarray:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append([float(value) for value in line.replace(",", " ").replace(";", " ").split()])
    if not rows:
        raise ValueError("Файл не содержит данных")
    if len({len(row) for row in rows}) != 1:
        raise ValueError("Во всех строках должно быть одинаковое количество значений")
    matrix = np.array(rows, dtype=float)
    if np.all(matrix == matrix.astype(int)):
        matrix = matrix.astype(int)
    return matrix


def input_matrix(path_data: str | Path) -> np.ndarray:
    with open(path_data, encoding="utf-8") as file:
        return parse_matrix_text(file.read())


def validate_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2:
        raise ValueError("Матрица должна быть двумерной")
    if matrix.shape[0] < 3 or matrix.shape[1] < 3:
        raise ValueError("Нужно минимум 3 товара и 3 пользователя")
    if not np.isfinite(matrix).all():
        raise ValueError("Матрица содержит некорректные значения")
    if np.any((matrix < 0) | (matrix > 5)):
        raise ValueError("Рейтинги должны находиться в диапазоне от 0 до 5")


def calc_cosine_sim(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def find_max_sim(vectors, labels, top_n=2):
    pairs = []
    for i, j in combinations(range(len(vectors)), 2):
        sim = calc_cosine_sim(vectors[i], vectors[j])
        pairs.append((sim, labels[i], labels[j]))
    pairs.sort(reverse=True, key=lambda x: x[0])
    return pairs[:top_n]


def main():
    m = input_matrix(PATH_DATA)
    validate_matrix(m)
    n_products, n_users = m.shape

    products = [f"P{i}" for i in range(1, n_products + 1)]
    users = [f"U{i}" for i in range(1, n_users + 1)]

    print("=== Продукты ===")
    for sim, a, b in find_max_sim(m, products):
        print(f"{a}–{b}: {sim:.4f}")

    print("\n=== Пользователи ===")
    for sim, a, b in find_max_sim(m.T, users):
        print(f"{a}–{b}: {sim:.4f}")


if __name__ == "__main__":
    main()
