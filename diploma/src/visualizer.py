import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SphereVisualizer:
    def __init__(self):
        pass

    def plot_sphere_3d_plotly(self, 
                             points: np.ndarray, 
                             labels: Optional[List[str]] = None,
                             title: str = "Sphere Visualization",
                             save_path: Optional[str] = None):
        if points.shape[1] < 3:
            logger.error("Для 3D визуализации нужно минимум 3 измерения")
            return
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        fig = go.Figure()


        hover_text = labels if labels else [f"Point {i}" for i in range(len(points))]

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=10,
                color=np.arange(len(points)),
                colorscale='Viridis',
                showscale=False,
                line=dict(width=1, color='white')
            ),
            hovertext=hover_text,
            hoverinfo='text',
            name='Триплеты'
        ))

        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(go.Surface(
            x=xs, y=ys, z=zs,
            opacity=0.1,
            colorscale='Greys',
            showscale=False,
            hoverinfo='skip'
        ))

        # Настройки макета
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(range=[-1.2, 1.2], title='X'),
                yaxis=dict(range=[-1.2, 1.2], title='Y'),
                zaxis=dict(range=[-1.2, 1.2], title='Z'),
                aspectmode='cube'
            ),
            width=900,
            height=700
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"График сохранен в {save_path}")

        fig.show()

    def plot_sphere_3d_matplotlib(self, 
                                 points: np.ndarray,
                                 labels: Optional[List[str]] = None,
                                 title: str = "Sphere Visualization",
                                 save_path: Optional[str] = None):

        if points.shape[1] < 3:
            logger.error("Для 3D визуализации нужно минимум 3 измерения")
            return

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(xs, ys, zs, alpha=0.1, color='gray')

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        scatter = ax.scatter(x, y, z, c=np.arange(len(points)), 
                            cmap='viridis', s=100, edgecolors='black', linewidth=1)

        if labels:
            for i, label in enumerate(labels):
                ax.text(x[i], y[i], z[i], label, fontsize=9)


        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(title, fontsize=14)


        plt.colorbar(scatter, ax=ax, label='Point Index', shrink=0.5)


        max_range = 1.2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"График сохранен в {save_path}")

        plt.show()

    def plot_distance_heatmap(self, 
                             distance_matrix: np.ndarray,
                             labels: Optional[List[str]] = None,
                             triplet_labels: Optional[List[str]] = None,
                             title: str = "Distance Matrix",
                             save_path: Optional[str] = None):
        if triplet_labels:
            hover_texts = []
            for i in range(len(distance_matrix)):
                row_texts = []
                for j in range(len(distance_matrix)):
                    text = (f"{triplet_labels[i]}<br>"
                           f"↔<br>"
                           f"{triplet_labels[j]}<br>"
                           f"Расстояние: {distance_matrix[i,j]:.4f}")
                    row_texts.append(text)
                hover_texts.append(row_texts)
        else:
            hover_texts = None

        fig = go.Figure(data=go.Heatmap(
            z=distance_matrix,
            x=labels if labels else list(range(len(distance_matrix))),
            y=labels if labels else list(range(len(distance_matrix))),
            colorscale='Viridis',
            colorbar=dict(title="Расстояние"),
            hovertext=hover_texts,
            hoverinfo='text' if hover_texts else 'z'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Триплет",
            yaxis_title="Триплет",
            width=800,
            height=700
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Heatmap сохранена в {save_path}")

        fig.show()

# if __name__ == "__main__":
#     # Тестовая визуализация
#     # Создаем случайные точки на сфере
#     n_points = 10
#     points = np.random.randn(n_points, 3)
#     points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
#     labels = [f"T{i}" for i in range(n_points)]
    
#     visualizer = SphereVisualizer()
#     visualizer.plot_sphere_3d_plotly(points, labels=labels)
