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
                             save_path: Optional[str] = None,
                             show_reference_spheres: bool = False,
                             original_radii: Optional[np.ndarray] = None,
                             colors: Optional[List[str]] = None):
        if points.shape[1] < 3:
            logger.error("Для 3D визуализации нужно минимум 3 измерения")
            return
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        fig = go.Figure()

        # Вычисляем радиусы точек для проверки (должны быть ~1.0)
        radii = np.sqrt(x**2 + y**2 + z**2)
        
        # Определяем цветовую кодировку
        if colors is not None:
            # Используем пользовательские цвета (для разных POS)
            marker_config = dict(
                size=8,
                color=colors,
                line=dict(width=1, color='white')
            )
        else:
            # Используем исходные радиусы для цветовой кодировки, если доступны
            color_values = original_radii if original_radii is not None else radii
            marker_config = dict(
                size=8,
                color=color_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Исходный r" if original_radii is not None else "Радиус r"
                ),
                line=dict(width=1, color='white')
            )
        
        hover_text = []
        text_labels = []  # Для отображения на точках
        
        if labels:
            for i, label in enumerate(labels):
                # Упрощенная метка для отображения (только лемма без POS)
                short_label = label.split('(')[0].strip() if '(' in label else label
                text_labels.append(short_label)
                
                # Полная информация в hover
                if original_radii is not None:
                    hover_text.append(f"{label}<br>r_orig={original_radii[i]:.3f}<br>r_sphere={radii[i]:.3f}")
                else:
                    hover_text.append(f"{label}<br>r={radii[i]:.3f}")
        else:
            for i in range(len(points)):
                text_labels.append(f"{i}")
                if original_radii is not None:
                    hover_text.append(f"Point {i}<br>r_orig={original_radii[i]:.3f}<br>r_sphere={radii[i]:.3f}")
                else:
                    hover_text.append(f"Point {i}<br>r={radii[i]:.3f}")

        # Точки данных
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            marker=marker_config,
            text=text_labels,  # Используем названия токенов вместо номеров
            textposition="top center",
            textfont=dict(size=8),
            hovertext=hover_text,
            hoverinfo='text',
            name='Точки'
        ))

        # Рисуем единичную сферу
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 30)
        
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=xs, y=ys, z=zs,
            opacity=0.15,
            colorscale=[[0, 'lightgray'], [1, 'lightgray']],
            showscale=False,
            hoverinfo='skip',
            name='Единичная сфера (r=1)'
        ))

        # Фиксированный диапазон осей для единичной сферы
        axis_range = 1.3
        
        # Настройки макета
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(range=[-axis_range, axis_range], title='X'),
                yaxis=dict(range=[-axis_range, axis_range], title='Y'),
                zaxis=dict(range=[-axis_range, axis_range], title='Z'),
                aspectmode='cube'
            ),
            width=1000,
            height=800
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"График сохранен в {save_path}")

        fig.show()

    def plot_sphere_3d_matplotlib(self, 
                                 points: np.ndarray,
                                 labels: Optional[List[str]] = None,
                                 title: str = "Sphere Visualization",
                                 save_path: Optional[str] = None,
                                 show_reference_spheres: bool = False,
                                 original_radii: Optional[np.ndarray] = None):
        if points.shape[1] < 3:
            logger.error("Для 3D визуализации нужно минимум 3 измерения")
            return

        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # Вычисляем радиусы точек для проверки (должны быть ~1.0)
        radii = np.sqrt(x**2 + y**2 + z**2)
        
        # Используем исходные радиусы для цветовой кодировки, если доступны
        color_values = original_radii if original_radii is not None else radii

        # Рисуем единичную сферу
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(xs, ys, zs, alpha=0.1, color='lightgray')

        # Точки данных
        scatter = ax.scatter(x, y, z, c=color_values, 
                            cmap='viridis', s=100, edgecolors='black', linewidth=1)

        # Метки
        if labels:
            for i, label in enumerate(labels):
                ax.text(x[i], y[i], z[i], f"{i}", fontsize=8)

        # Оси
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(title, fontsize=14)

        # Colorbar
        cbar_label = 'Исходный радиус r' if original_radii is not None else 'Радиус r'
        plt.colorbar(scatter, ax=ax, label=cbar_label, shrink=0.5)

        # Фиксированный диапазон для единичной сферы
        axis_range = 1.3
        ax.set_xlim([-axis_range, axis_range])
        ax.set_ylim([-axis_range, axis_range])
        ax.set_zlim([-axis_range, axis_range])

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
