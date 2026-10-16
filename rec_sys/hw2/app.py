import numpy as np
import pandas as pd
import streamlit as st

from task2 import DEFAULT_MATRIX, agglomerate, cluster_name


def invalidate_result():
    st.session_state.pop("result", None)


def create_matrix(size, example=False):
    st.session_state["matrix_size"] = size
    st.session_state["matrix_version"] = st.session_state.get("matrix_version", 0) + 1
    st.session_state["use_example"] = example
    invalidate_result()


def load_example():
    create_matrix(5, example=True)
    st.session_state["count"] = 5
    st.session_state["threshold"] = .85


def show_matrix(matrix, clusters):
    labels = [cluster_name(group) for group in clusters]
    st.dataframe(
        pd.DataFrame(matrix, index=labels, columns=labels).style.format("{:.4f}"),
        width="stretch",
    )


def show_result(result):
    st.caption(f"Порог R = {result.threshold:g} · Объединений: {len(result.steps)}")
    st.markdown("**Исходная матрица**")
    show_matrix(result.initial_matrix, tuple((i,) for i in range(len(result.initial_matrix))))
    for number, step in enumerate(result.steps, 1):
        st.markdown(
            f"**Шаг {number}: {cluster_name(step.left)} + {cluster_name(step.right)}**"
        )
        st.caption(f"Максимальное сходство: {step.similarity:g} ≥ R = {result.threshold:g}")
        show_matrix(step.matrix, step.clusters)

    if result.next_similarity is None:
        st.success("Остановка: все объекты объединены в один кластер (матрица 1 × 1).")
    else:
        st.info(
            f"Остановка: максимальное сходство {result.next_similarity:g} "
            f"меньше порога R = {result.threshold:g}."
        )
    st.markdown("**Итоговые кластеры**")
    for number, group in enumerate(result.clusters, 1):
        st.write(f"Кластер {number}: {', '.join(f'U{i + 1}' for i in group)}")


def main():
    st.set_page_config(page_title="ДЗ 2 · Кластеризация", page_icon="🔗", layout="wide")
    st.title("Кластеризация по косинусному подобию")
    st.write("Введите значения над диагональю и задайте порог объединения R.")
    st.caption("Метод ближайшего соседа: сходство кластеров равно максимальному сходству их объектов.")

    if "matrix_size" not in st.session_state:
        create_matrix(5, example=True)
    st.session_state.setdefault("count", 5)
    st.session_state.setdefault("threshold", .85)

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Исходные данные")
        controls = st.columns([1, 1, 1])
        with controls[0]:
            count_value = st.number_input("Число объектов", 2, 12, value=None, step=1, key="count")
            count = int(count_value) if count_value is not None else st.session_state["matrix_size"]
        with controls[1]:
            st.button("Создать матрицу", on_click=create_matrix, args=(count,),
                      width="stretch", disabled=count_value is None)
        with controls[2]:
            st.button("Пример задания", on_click=load_example, width="stretch")

        threshold = st.number_input(
            "Порог R", min_value=0., max_value=1., value=None, step=.01,
            format="%.4f", key="threshold", on_change=invalidate_result,
        )
        st.caption("R = 0 — объединить всех. При сходстве, равном R, объединение выполняется.")
        size = st.session_state["matrix_size"]
        version = st.session_state["matrix_version"]
        if count_value is None or threshold is None:
            st.warning("Заполните число объектов и порог R.")
        if count != size:
            st.warning("Нажмите «Создать матрицу», чтобы применить новое число объектов.")

        matrix = np.zeros((size, size))
        # Матрица сохраняет строки и столбцы даже при узком окне браузера.
        st.html(f"""
            <style>
            .st-key-similarity_grid {{ overflow-x: auto; }}
            .st-key-similarity_grid [data-testid="stHorizontalBlock"] {{
                min-width: {size * 90 + 40}px; flex-wrap: nowrap !important;
            }}
            .st-key-similarity_grid [data-testid="stColumn"] {{
                min-width: 0 !important; flex: 1 1 0 !important;
            }}
            .st-key-similarity_grid [data-testid="stColumn"]:first-child {{
                flex: .6 1 0 !important;
            }}
            </style>
        """)
        with st.container(border=True, key="similarity_grid"):
            headers = st.columns([.6] + [1] * size, gap="small")
            for j in range(size):
                headers[j + 1].markdown(f"**U{j + 1}**")
            for i in range(size):
                cells = st.columns([.6] + [1] * size, gap="small")
                cells[0].markdown(f"**U{i + 1}**")
                for j in range(size):
                    with cells[j + 1]:
                        if i < j:
                            initial = (
                                float(DEFAULT_MATRIX[i, j])
                                if st.session_state["use_example"] else 0.
                            )
                            value = st.number_input(
                                f"Сходство U{i + 1} и U{j + 1}",
                                min_value=0., max_value=1., value=initial,
                                step=.01, format="%.4f", label_visibility="collapsed",
                                key=f"cell_{version}_{i}_{j}", on_change=invalidate_result,
                            )
                            matrix[i, j] = matrix[j, i] = value
                        else:
                            st.text(f"{matrix[i, j]:.4f}")
        st.caption("Нижняя половина обновляется автоматически; диагональ исключена из поиска максимума.")
        if st.button("Выполнить кластеризацию", type="primary", width="stretch",
                     disabled=count != size or count_value is None or threshold is None):
            try:
                st.session_state["result"] = agglomerate(matrix, threshold)
            except ValueError as error:
                st.error(str(error))

    with right:
        st.subheader("Шаги объединения")
        with st.container(height=680, border=True):
            if "result" in st.session_state:
                show_result(st.session_state["result"])
            else:
                st.info("Нажмите «Выполнить кластеризацию», чтобы увидеть матрицы каждого шага и итоговые группы.")


if __name__ == "__main__":
    main()
