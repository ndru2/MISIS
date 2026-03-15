import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from visualizer import SphereVisualizer
from clustering import VonMisesFisherMixture, HDBSCANClustering

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SphereBuilder:
    def __init__(self, embedding_dim: int = 50):
        self.embedding_dim = embedding_dim
        self.model = None
        self.triplet_vectors = []
        self.triplets = []
        self.tokens = []  # Для токенов с BERT
        self.mode = None  # 'triplets' или 'tokens'
        self.visualizer = SphereVisualizer()
        self.clustering_model = None  # Модель кластеризации
        self.cluster_labels = None  # Метки кластеров

    def train_embeddings(self, triplets: List[Tuple[str, str, str]], 
                        use_pretrained: bool = False):
        """Обучение Word2Vec эмбеддингов для триплетов"""
        self.triplets = triplets
        self.mode = 'triplets'

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
    
    def set_token_embeddings(self, tokens: List):
        """
        Установка токенов с готовыми BERT эмбеддингами
        
        Args:
            tokens: List[Token] с полями text, lemma, pos, embedding
            
        Если self.embedding_dim < размерность токенов, автоматически применяется PCA
        """
        self.tokens = tokens
        self.mode = 'tokens'
        
        # Проверяем что у токенов есть эмбеддинги
        if not tokens:
            raise ValueError("Список токенов пуст")
        
        if tokens[0].embedding is None:
            raise ValueError("Токены не содержат эмбеддингов")
        
        original_dim = len(tokens[0].embedding)
        n_samples = len(tokens)
        self.token_embeddings = np.array([t.embedding for t in tokens])
        
        # PCA не может создать больше компонент чем min(n_samples, n_features)
        max_components = min(n_samples, original_dim)
        
        # Автоматическое понижение размерности если требуется
        if self.embedding_dim < original_dim:
            # Ограничиваем количество компонент
            target_dim = min(self.embedding_dim, max_components)
            
            if target_dim < self.embedding_dim:
                logger.warning(f"Целевая размерность {self.embedding_dim}D слишком велика для {n_samples} образцов. Используется {target_dim}D")
                self.embedding_dim = target_dim
            
            logger.info(f"Понижение размерности: {original_dim}D → {self.embedding_dim}D с помощью PCA")
            
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.embedding_dim)
            self.token_embeddings = pca.fit_transform(self.token_embeddings)
            
            explained_variance = pca.explained_variance_ratio_.sum()
            logger.info(f"PCA сохранила {explained_variance*100:.2f}% дисперсии")
        elif self.embedding_dim > original_dim:
            logger.warning(f"Целевая размерность ({self.embedding_dim}D) больше исходной ({original_dim}D). Используется {original_dim}D")
            self.embedding_dim = original_dim
        
        logger.info(f"Загружено {len(tokens)} токенов с эмбеддингами (финальная dim={self.embedding_dim})")

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
        Работает как для триплетов (Word2Vec), так и для токенов (BERT)
        
        Алгоритм:
        1. Получаем эмбединг в декартовых координатах
        2. Переводим в сферические координаты
        3. Нормализуем радиус на 1.0
        4. Переводим обратно в декартовы координаты
        Результат: точки в декартовых координатах на единичной сфере
        """
        embeddings = []
        original_radii = []
        
        # Выбираем источник эмбеддингов в зависимости от режима
        if self.mode == 'triplets':
            logger.info("Режим: триплеты с Word2Vec")
            for triplet in self.triplets:
                emb = self.get_triplet_embedding(triplet)
                embeddings.append(emb)
        elif self.mode == 'tokens':
            logger.info("Режим: токены с BERT эмбеддингами")
            embeddings = self.token_embeddings.copy()
        else:
            raise ValueError("Сначала вызовите train_embeddings() или set_token_embeddings()")
        
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
                           save_path: Optional[str] = None,
                           color_by_clusters: bool = False):
        if len(self.triplet_vectors) == 0:
            logger.error("Сначала нужно построить сферу с помощью project_to_sphere()")
            return

        if self.triplet_vectors.shape[1] < 3:
            logger.error("Для 3D визуализации нужно минимум 3 измерения")
            return

        points_3d = self.triplet_vectors[:, :3]

        # Генерируем метки в зависимости от режима
        if labels is None:
            if self.mode == 'triplets':
                labels = [f"{s}-{p}-{o}" for s, p, o in self.triplets]
            elif self.mode == 'tokens':
                labels = [f"{t.lemma} ({t.pos})" for t in self.tokens]
            else:
                labels = [f"Point {i}" for i in range(len(points_3d))]
        
        # Выбор цветовой схемы
        colors = None
        if color_by_clusters and self.cluster_labels is not None:
            # Цвета по кластерам
            import matplotlib.cm as cm
            n_clusters = len(np.unique(self.cluster_labels))
            colormap = cm.get_cmap('tab10' if n_clusters <= 10 else 'tab20')
            colors = [colormap(label / n_clusters) for label in self.cluster_labels]
            colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b, _ in colors]
        elif self.mode == 'tokens':
            # Цвета по частям речи (старое поведение)
            colors_map = {'NOUN': 'blue', 'VERB': 'red',}
            colors = [colors_map.get(t.pos, 'gray') for t in self.tokens]

        if method == 'plotly':
            self.visualizer.plot_sphere_3d_plotly(
                points_3d, labels, title, save_path,
                show_reference_spheres=False,
                original_radii=self.original_radii if hasattr(self, 'original_radii') else None,
                colors=colors
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

        # Генерируем метки в зависимости от режима
        if labels is None:
            if self.mode == 'triplets':
                labels = [f"{s}-{p}-{o}" for s, p, o in self.triplets]
            elif self.mode == 'tokens':
                labels = [f"{t.lemma}" for t in self.tokens]
            else:
                labels = [f"Point {i}" for i in range(len(self.triplet_vectors))]

        # Для heatmap нужны более подробные метки
        if self.mode == 'triplets':
            triplet_labels = [f"{s} {p} {o}" for s, p, o in self.triplets]
        elif self.mode == 'tokens':
            triplet_labels = [f"{t.text} ({t.pos})" for t in self.tokens]
        else:
            triplet_labels = labels

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

        info = {
            'n_points': len(self.triplet_vectors),
            'dimension': self.embedding_dim,
            'avg_angular_distance': np.mean(angular_distances),
            'min_angular_distance': np.min(angular_distances),
            'max_angular_distance': np.max(angular_distances),
            'avg_cosine_similarity': np.mean(distances['cosine_similarity'][triu_indices])
        }
        
        # Добавляем информацию о кластеризации, если она была выполнена
        if self.cluster_labels is not None:
            info['n_clusters'] = len(np.unique(self.cluster_labels))
            info['cluster_sizes'] = np.bincount(self.cluster_labels).tolist()
        
        return info
    
    def fit_vmf_clustering(
        self,
        n_clusters: int = 3,
        max_iter: int = 100,
        init_method: str = 'kmeans++',
        random_state: Optional[int] = None
    ) -> 'SphereBuilder':
        """
        Кластеризация точек на сфере с помощью von Mises-Fisher mixture model
        
        Parameters:
        -----------
        n_clusters : int
            Количество кластеров
        max_iter : int
            Максимальное количество итераций EM-алгоритма
        init_method : str
            Метод инициализации: 'random', 'kmeans++', 'spherical_kmeans'
        random_state : int, optional
            Seed для воспроизводимости
        
        Returns:
        --------
        self : SphereBuilder
        """
        if len(self.triplet_vectors) == 0:
            raise ValueError("Сначала выполните project_to_sphere()")
        
        logger.info(f"Запуск vMF кластеризации: {n_clusters} кластеров")
        
        # Создаем и обучаем модель
        self.clustering_model = VonMisesFisherMixture(
            n_clusters=n_clusters,
            max_iter=max_iter,
            init_method=init_method,
            random_state=random_state
        )
        
        self.clustering_model.fit(self.triplet_vectors)
        self.cluster_labels = self.clustering_model.labels_
        
        logger.info(f"Кластеризация завершена")
        logger.info(f"Распределение по кластерам: {np.bincount(self.cluster_labels)}")
        
        return self
    
    def get_clustering_info(self) -> Dict:
        """
        Получить информацию о кластеризации
        
        Returns:
        --------
        info : dict
            Информация о кластерах
        """
        if self.clustering_model is None:
            return {"status": "Кластеризация не выполнена"}
        
        return self.clustering_model.get_cluster_info()
    
    def get_clustering_quality_metrics(self) -> dict:
        if self.clustering_model is None:
            return {"status": "Кластеризация не выполнена"}
        
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        X = self.triplet_vectors
        labels = self.cluster_labels
        
        if len(np.unique(labels)) < 2:
            return {"error": "Нужно минимум 2 кластера"}
        
        silhouette = silhouette_score(X, labels, metric='cosine')
        
        davies_bouldin = davies_bouldin_score(X, labels)
        
        calinski = calinski_harabasz_score(X, labels)
        
        # Вычисление внутрикластерных расстояний
        intra_cluster_distances = []
        for k in range(self.clustering_model.n_clusters):
            mask = (labels == k)
            if mask.sum() > 0:
                cluster_points = X[mask]
                
                # Для VMF используем mu_, для HDBSCAN вычисляем центроид
                if hasattr(self.clustering_model, 'mu_'):
                    centroid = self.clustering_model.mu_[k]
                else:
                    centroid = cluster_points.mean(axis=0)
                    centroid_norm = np.linalg.norm(centroid)
                    if centroid_norm > 1e-10:
                        centroid = centroid / centroid_norm
                
                distances = 1 - np.dot(cluster_points, centroid)
                intra_cluster_distances.append(distances.mean())
        
        avg_intra_distance = np.mean(intra_cluster_distances) if intra_cluster_distances else 0
        
        # Вычисление межкластерных расстояний
        if hasattr(self.clustering_model, 'mu_'):
            centroids = self.clustering_model.mu_
        else:
            # Вычисляем центроиды для всех кластеров
            centroids = []
            for k in range(self.clustering_model.n_clusters):
                mask = (labels == k)
                if mask.sum() > 0:
                    cluster_points = X[mask]
                    centroid = cluster_points.mean(axis=0)
                    centroid_norm = np.linalg.norm(centroid)
                    if centroid_norm > 1e-10:
                        centroid = centroid / centroid_norm
                    centroids.append(centroid)
            centroids = np.array(centroids)
        
        inter_distances = []
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                dist = 1 - np.dot(centroids[i], centroids[j])
                inter_distances.append(dist)
        
        avg_inter_distance = np.mean(inter_distances) if inter_distances else 0
        
        return {
            'silhouette_score': silhouette,
            'davies_bouldin_index': davies_bouldin,
            'calinski_harabasz_score': calinski,
            'avg_intra_cluster_distance': avg_intra_distance,
            'avg_inter_cluster_distance': avg_inter_distance,
            'separation_ratio': avg_inter_distance / (avg_intra_distance + 1e-10)
        }
    
    def get_cluster_representatives(self, top_n: int = 5) -> Dict[int, List]:
        """
        Получить наиболее представительные элементы каждого кластера
        (ближайшие к центроиду кластера)
        
        Parameters:
        -----------
        top_n : int
            Количество представителей для каждого кластера
        
        Returns:
        --------
        representatives : Dict[int, List]
            Словарь {cluster_id: [(index, similarity, label), ...]}
        """
        if self.clustering_model is None:
            raise ValueError("Сначала выполните fit_vmf_clustering() или fit_hdbscan_clustering()")
        
        representatives = {}
        
        for k in range(self.clustering_model.n_clusters):
            # Находим все точки кластера
            cluster_mask = (self.cluster_labels == k)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                representatives[k] = []
                continue
            
            cluster_points = self.triplet_vectors[cluster_mask]
            
            # Для VMF используем центроид mu_, для HDBSCAN вычисляем центроид
            if hasattr(self.clustering_model, 'mu_'):
                # VMF кластеризация
                centroid = self.clustering_model.mu_[k]
            else:
                # HDBSCAN или другая кластеризация - вычисляем центроид
                centroid = cluster_points.mean(axis=0)
                # Нормализуем для сферических данных
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-10:
                    centroid = centroid / centroid_norm
            
            # Вычисляем сходство с центроидом кластера
            similarities = np.dot(cluster_points, centroid)
            
            # Сортируем по убыванию сходства
            sorted_indices = np.argsort(-similarities)[:top_n]
            
            # Формируем результат
            result = []
            for idx in sorted_indices:
                global_idx = cluster_indices[idx]
                similarity = similarities[idx]
            
            # Сортируем по убыванию сходства
            sorted_indices = np.argsort(-similarities)[:top_n]
            
            # Формируем результат
            result = []
            for idx in sorted_indices:
                global_idx = cluster_indices[idx]
                similarity = similarities[idx]
                
                # Получаем метку (токен или триплет)
                if self.mode == 'tokens':
                    label = f"{self.tokens[global_idx].lemma} ({self.tokens[global_idx].pos})"
                elif self.mode == 'triplets':
                    label = f"{self.triplets[global_idx][0]}-{self.triplets[global_idx][1]}-{self.triplets[global_idx][2]}"
                else:
                    label = f"Point {global_idx}"
                
                result.append((global_idx, similarity, label))
            
            representatives[k] = result
        
        return representatives
    
    def fit_hdbscan_clustering(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        metric: str = 'cosine',
        cluster_selection_method: str = 'eom',
        **kwargs
    ) -> 'SphereBuilder':
        """
        HDBSCAN кластеризация точек на сфере
        
        HDBSCAN автоматически определяет количество кластеров и находит выбросы.
        Хорошо работает с кластерами разного размера и плотности.
        
        Parameters:
        -----------
        min_cluster_size : int
            Минимальное количество точек в кластере (по умолчанию 5)
        min_samples : int, optional
            Минимальное количество соседей для core point
            Если None, используется min_cluster_size
        metric : str
            Метрика расстояния: 'cosine' (рекомендуется для сферы), 'euclidean'
        cluster_selection_method : str
            Метод выбора кластеров: 'eom' (по умолчанию) или 'leaf'
        **kwargs
            Дополнительные параметры для HDBSCAN
        
        Returns:
        --------
        self : SphereBuilder
        """
        if len(self.triplet_vectors) == 0:
            raise ValueError("Сначала выполните project_to_sphere()")
        
        logger.info(f"Запуск HDBSCAN кластеризации")
        
        try:
            # Создаем и обучаем модель
            self.clustering_model = HDBSCANClustering(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                cluster_selection_method=cluster_selection_method,
                **kwargs
            )
            
            self.clustering_model.fit(self.triplet_vectors)
            self.cluster_labels = self.clustering_model.labels_
            
            logger.info(f"HDBSCAN кластеризация завершена")
            
            # Информация о результатах
            n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
            n_outliers = (self.cluster_labels == -1).sum()
            
            logger.info(f"Найдено кластеров: {n_clusters}")
            logger.info(f"Выбросов: {n_outliers}")
            
            if n_clusters > 0:
                mask = self.cluster_labels != -1
                cluster_sizes = np.bincount(self.cluster_labels[mask])
                logger.info(f"Распределение по кластерам: {cluster_sizes.tolist()}")
            
        except ImportError as e:
            logger.error(f"HDBSCAN не установлен: {e}")
            raise ImportError(
                "Для использования HDBSCAN установите библиотеку:\n"
                "  pip install hdbscan\n"
                "Или для более быстрой версии:\n"
                "  pip install hdbscan[speedup]"
            )
        
        return self
    
    def get_hdbscan_outliers(self) -> List:
        """
        Получить выбросы (шум), найденные HDBSCAN
        
        Returns:
        --------
        outliers : List[Tuple[int, str, float]]
            Список кортежей (index, label, probability)
        """
        if self.clustering_model is None or not isinstance(self.clustering_model, HDBSCANClustering):
            raise ValueError("Сначала выполните fit_hdbscan_clustering()")
        
        outlier_indices = self.clustering_model.get_outliers()
        
        outliers = []
        for idx in outlier_indices:
            # Получаем метку (токен или триплет)
            if self.mode == 'tokens':
                label = f"{self.tokens[idx].lemma} ({self.tokens[idx].pos})"
            elif self.mode == 'triplets':
                label = f"{self.triplets[idx][0]}-{self.triplets[idx][1]}-{self.triplets[idx][2]}"
            else:
                label = f"Point {idx}"
            
            # Вероятность принадлежности (для выбросов обычно низкая)
            prob = self.clustering_model.probabilities_[idx]
            
            outliers.append((idx, label, prob))
        
        return outliers


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