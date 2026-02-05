import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from triplet_extractor import TripletExtractor
from sphere_builder import SphereBuilder
from visualizer import SphereVisualizer

import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_text(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    print("="*60)
    print("ПОСТРОЕНИЕ СФЕРИЧЕСКОГО ПРЕДСТАВЛЕНИЯ ИЗ ТЕКСТА")
    print("="*60)

    print("\nЗагрузка текста...")
    text_path = "data/sample_text.txt"

    if not os.path.exists(text_path):
        logger.error(f"Файл {text_path} не найден!")
        return

    text = load_text(text_path)
    print(f"Загружено {len(text)} символов")
    print(f"Предварительный просмотр:\n{text[:200]}...")

    print("\nИзвлечение триплетов...")

    backend = 'keybert'

    print(f"Используется backend: {backend}")

    try:
        extractor = TripletExtractor(
            backend=backend,
            top_n=50,
            diversity=0.6,
            min_keyword_score=0.25,
            ngram_range=(1, 2)
            )

        if not extractor.is_ready():
            print(f"Парсер '{backend}' не готов к работе!")
            if backend == 'spacy':
                print("Установите: python -m spacy download ru_core_news_sm")
            elif backend == 'bilstm':
                print("Установите: pip install -U supar")
            elif backend == 'stanza':
                print("Установите: pip install stanza")
            elif backend == 'keybert':
                print("Установите: pip install keybert stanza")
            return

    except Exception as e:
        print(f"Ошибка инициализации парсера: {e}")
        return

    triplets = extractor.extract_and_display(text)

    if not triplets:
        logger.warning("Не удалось извлечь триплеты из текста")
        return

    print("\nСоздание эмбеддингов и обучение Word2Vec...")
    embedding_dim = 40

    builder = SphereBuilder(embedding_dim=embedding_dim)
    builder.train_embeddings(triplets)


    print("\nПроекция на единичную сферу...")
    sphere_points = builder.project_to_sphere()


    print("\n=== Информация о сфере ===")
    info = builder.get_sphere_info()
    for key, value in info.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


    print("\nВизуализация...")

    if embedding_dim >= 3:
        points_3d = sphere_points[:, :3]
        points_3d = points_3d / np.linalg.norm(points_3d, axis=1, keepdims=True)
    else:

        padding = np.zeros((len(sphere_points), 3 - embedding_dim))
        points_3d = np.hstack([sphere_points, padding])

    labels = [f"{s}→{p}→{o}" for s, p, o in triplets]
    #short_labels = [f"T{i}" for i in range(len(triplets))]

    visualizer = SphereVisualizer()


    print("\nСоздание интерактивной 3D визуализации...")
    visualizer.plot_sphere_3d_plotly(
        points_3d,
        labels=labels,
        title=f"Сферическое представление триплетов",
        #title=f"Сферическое представление триплетов (размерность {embedding_dim}D, проекция на первые 3D)",
        save_path="outputs/sphere_3d_2.html"
    )

    # print("\nСоздание heatmap расстояний...")
    # distances = builder.compute_distances()
    # visualizer.plot_distance_heatmap(
    #     distances['angular_distance'],
    #     labels=short_labels,
    #     triplet_labels=labels,
    #     title="Матрица угловых расстояний",
    #     save_path="outputs/distance_heatmap.html"
    # )

    print("\n" + "="*60)
    print("Pipeline завершен успешно!")
    print("Результаты сохранены в папке 'outputs/'")
    print("="*60)


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    main()
