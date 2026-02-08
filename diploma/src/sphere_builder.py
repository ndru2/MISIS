import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from visualizer import SphereVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SphereBuilder:
    def __init__(self, embedding_dim: int = 50):
        self.embedding_dim = embedding_dim
        self.model = None
        self.triplet_vectors = []
        self.triplets = []
        self.visualizer = SphereVisualizer()

    def train_embeddings(self, triplets: List[Tuple[str, str, str]], 
                        use_pretrained: bool = False):

        self.triplets = triplets


        sentences = [[subj, pred, obj] for subj, pred, obj in triplets]

        all_words = set()
        for triplet in triplets:
            all_words.update(triplet)

        logger.info(f"Обучение Word2Vec на {len(sentences)} триплетах, "
                   f"словарь: {len(all_words)} слов")

        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.embedding_dim,
            window=3,
            min_count=1,
            workers=4,
            sg=1,
            epochs=100
        )

        logger.info("Word2Vec модель обучена")

    def get_triplet_embedding(self, triplet: Tuple[str, str, str]) -> np.ndarray:
        subj, pred, obj = triplet
        vectors = []

        for word in [subj, pred, obj]:
            if word in self.model.wv:
                vectors.append(self.model.wv[word])
            # else:
            #     vectors.append(np.random.randn(self.embedding_dim) * 0.1)

        return np.mean(vectors, axis=0)
    
    def cartesian_to_spherical(self, x: np.ndarray) -> np.ndarray:
        """
        Переход из декартовых координат в сферические (n-мерный случай)
        """
        n = len(x)

        r = np.linalg.norm(x)

        if r < 1e-10:
            return np.zeros(n)

        spherical = [r]

        for i in range(n - 2):
            sum_squares = np.sum(x[i+1:]**2)

            if sum_squares < 1e-10:
                theta_i = 0.0
            else:
                theta_i = np.arctan2(np.sqrt(sum_squares), x[i])

            spherical.append(theta_i)
        phi = np.arctan2(x[-1], x[-2]) if n >= 2 else 0.0
        spherical.append(phi)

        return np.array(spherical)

    def spherical_to_cartesian(self, spherical: np.ndarray) -> np.ndarray:
        """
        Обратное преобразование: из сферических координат в декартовы
        """
        n = len(spherical)
        r = spherical[0]
        angles = spherical[1:]

        if n == 1:
            return np.array([r])

        x = np.zeros(n)
        sin_product = r

        for i in range(n - 2):
            x[i] = sin_product * np.cos(angles[i])
            sin_product *= np.sin(angles[i])


        phi = angles[-1]
        x[-2] = sin_product * np.cos(phi)
        x[-1] = sin_product * np.sin(phi)

        return x

    def project_to_sphere(self) -> np.ndarray:
        """
        Преобразование эмбедингов на единичную гиперсферу S^(n-1)        
        Алгоритм:
        1. Получаем эмбединг в декартовых координатах
        2. Переводим в сферические координаты
        3. Нормализуем радиус
        4. Переводим обратно в декартовы координаты
        Результат: точки в декартовых координатах на единичной сфере
        """
        embeddings = []
        original_radii = []

        for triplet in self.triplets:
            emb = self.get_triplet_embedding(triplet)
            embeddings.append(emb)
        embeddings = np.array(embeddings)

        logger.info(f"Проекция {len(embeddings)} векторов на единичную гиперсферу S^{self.embedding_dim-1}")


        sphere_embeddings = []

        for emb in embeddings:
            spherical = self.cartesian_to_spherical(emb)
            original_radii.append(spherical[0])
            spherical[0] = 1.0
            cartesian_on_sphere = self.spherical_to_cartesian(spherical)
            sphere_embeddings.append(cartesian_on_sphere)

        sphere_embeddings = np.array(sphere_embeddings)
        self.original_radii = np.array(original_radii)

        self.triplet_vectors = sphere_embeddings

        logger.info(f"Спроецировано {len(sphere_embeddings)} точек на единичную сферу")
        logger.info(f"Размерность: {self.embedding_dim}D (гиперсфера S^{self.embedding_dim-1})")
        logger.info(f"Исходные радиусы: min={np.min(original_radii):.4f}, "
                   f"max={np.max(original_radii):.4f}, mean={np.mean(original_radii):.4f}")

        norms = np.linalg.norm(sphere_embeddings, axis=1)
        logger.info(f"Проверка нормы (должна быть ≈1.0): min={np.min(norms):.6f}, "
                   f"max={np.max(norms):.6f}, mean={np.mean(norms):.6f}")

        return sphere_embeddings

    def compute_distances(self) -> Dict[str, np.ndarray]:
        """
        Вычисление расстояний между точками на единичной сфере
        """
        if len(self.triplet_vectors) == 0:
            raise ValueError("Сначала выполните project_to_sphere()")

        n = len(self.triplet_vectors)

        cosine_similarity = np.dot(self.triplet_vectors, self.triplet_vectors.T)

        cosine_clipped = np.clip(cosine_similarity, -1.0, 1.0)
        angular_distance = np.arccos(cosine_clipped)

        euclidean_distance = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                euclidean_distance[i, j] = np.linalg.norm(
                    self.triplet_vectors[i] - self.triplet_vectors[j]
                )

        return {
            'cosine_similarity': cosine_similarity,
            'angular_distance': angular_distance,
            'euclidean_distance': euclidean_distance
        }

    def visualize_sphere_3d(self, 
                           method: str = 'plotly',
                           labels: Optional[List[str]] = None,
                           title: str = "Сфера семантических триплетов",
                           save_path: Optional[str] = None):
        if len(self.triplet_vectors) == 0:
            logger.error("Сначала нужно построить сферу с помощью project_to_sphere()")
            return

        if self.triplet_vectors.shape[1] < 3:
            logger.error("Для 3D визуализации нужно минимум 3 измерения")
            return

        points_3d = self.triplet_vectors[:, :3]

        if labels is None:
            labels = [f"{s}-{p}-{o}" for s, p, o in self.triplets]

        if method == 'plotly':
            self.visualizer.plot_sphere_3d_plotly(
                points_3d, labels, title, save_path,
                show_reference_spheres=False,
                original_radii=self.original_radii if hasattr(self, 'original_radii') else None
            )
        elif method == 'matplotlib':
            self.visualizer.plot_sphere_3d_matplotlib(
                points_3d, labels, title, save_path,
                show_reference_spheres=False,
                original_radii=self.original_radii if hasattr(self, 'original_radii') else None
            )
        else:
            logger.error(f"Неизвестный метод: {method}. Используйте 'plotly' или 'matplotlib'")
    
    def visualize_distance_heatmap(self, 
                                  distance_type: str = 'angular_distance',
                                  labels: Optional[List[str]] = None,
                                  save_path: Optional[str] = None):

        if len(self.triplet_vectors) == 0:
            logger.error("Сначала нужно построить сферу с помощью project_to_sphere()")
            return

        distances = self.compute_distances()

        if distance_type not in distances:
            logger.error(f"Неизвестный тип расстояния: {distance_type}")
            return

        distance_matrix = distances[distance_type]

        if labels is None:
            labels = [f"{s}-{p}-{o}" for s, p, o in self.triplets]


        triplet_labels = [f"{s} {p} {o}" for s, p, o in self.triplets]

        title = f"Матрица {distance_type.replace('_', ' ')}"
        self.visualizer.plot_distance_heatmap(
            distance_matrix=distance_matrix,
            labels=labels,
            triplet_labels=triplet_labels,
            title=title,
            save_path=save_path
        )

    def get_sphere_info(self) -> Dict:

        if len(self.triplet_vectors) == 0:
            return {"status": "Сфера не построена"}

        distances = self.compute_distances()

        triu_indices = np.triu_indices(len(self.triplet_vectors), k=1)
        angular_distances = distances['angular_distance'][triu_indices]

        return {
            'n_points': len(self.triplet_vectors),
            'dimension': self.embedding_dim,
            'avg_angular_distance': np.mean(angular_distances),
            'min_angular_distance': np.min(angular_distances),
            'max_angular_distance': np.max(angular_distances),
            'avg_cosine_similarity': np.mean(distances['cosine_similarity'][triu_indices])
        }


if __name__ == "__main__":
    # Тестовый запуск
    test_triplets = [
        ("программист", "создавать", "приложение"),
        ("программист", "улучшать", "приложение"),
        ("программист", "ломать", "приложение"),
        ("программист", "не создавать", "приложение"),
    ]

    print("=== Создание сферы семантических триплетов ===\n")


    builder = SphereBuilder(embedding_dim=10)
    builder.train_embeddings(test_triplets)

    sphere_points = builder.project_to_sphere()

    print("\n=== Информация о сфере ===")
    info = builder.get_sphere_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    print("\n=== Визуализация 3D сферы (Plotly) ===")
    builder.visualize_sphere_3d(
        method='plotly',
        title="Тестовая сфера триплетов",
        save_path="sphere_3d_plotly.html"
    )
    print("\n=== Визуализация матрицы угловых расстояний ===")
    builder.visualize_distance_heatmap(
        distance_type='angular_distance',
        save_path="distance_heatmap.html"
    )
    
    print("\n=== Визуализация завершена ===")
    print("Файлы сохранены:")
    print("  - sphere_3d_plotly.html")
    print("  - distance_heatmap.html")