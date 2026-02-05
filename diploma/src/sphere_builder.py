import numpy as np
from typing import List, Tuple, Dict
import logging
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SphereBuilder:
    def __init__(self, embedding_dim: int = 50):
        self.embedding_dim = embedding_dim
        self.model = None
        self.triplet_vectors = []
        self.triplets = []

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
            else:
                vectors.append(np.random.randn(self.embedding_dim) * 0.1)

        return np.mean(vectors, axis=0)
    
    def project_to_sphere(self) -> np.ndarray:
        embeddings = []
        for triplet in self.triplets:
            emb = self.get_triplet_embedding(triplet)
            embeddings.append(emb)

        embeddings = np.array(embeddings)

        sphere_embeddings = normalize(embeddings, norm='l2', axis=1)

        self.triplet_vectors = sphere_embeddings

        logger.info(f"Спроецировано {len(sphere_embeddings)} точек на сферу")
        logger.info(f"Размерность сферы: {self.embedding_dim}D")
        logger.info(f"Проверка нормы (должна быть 1.0): "
                   f"{np.linalg.norm(sphere_embeddings[0]):.6f}")
        return sphere_embeddings

    def compute_distances(self) -> Dict[str, np.ndarray]:
        if len(self.triplet_vectors) == 0:
            raise ValueError("Сначала project_to_sphere()")

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


# if __name__ == "__main__":
#     # Тестовый запуск
#     test_triplets = [
#         ("студент", "изучать", "программирование"),
#         ("программист", "создавать", "приложение"),
#         ("алгоритм", "решать", "задача"),
#     ]
    
#     builder = SphereBuilder(embedding_dim=10)
#     builder.train_embeddings(test_triplets)
#     sphere_points = builder.project_to_sphere()
    
#     print("\n=== Информация о сфере ===")
#     info = builder.get_sphere_info()
#     for key, value in info.items():
#         print(f"{key}: {value}")
