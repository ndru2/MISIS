import numpy as np
import pandas as pd
import streamlit as st

from task1 import find_max_sim, parse_matrix_text, validate_matrix


DEFAULT_MATRIX = np.array([
    [5, 4, 5, 3, 5],
    [5, 5, 5, 3, 5],
    [5, 4, 4, 2, 5],
    [5, 3, 5, 0, 3],
    [5, 0, 5, 0, 0],
    [4, 5, 5, 3, 1],
])


def decode_file(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Не удалось определить кодировку файла")


def matrix_for_size(rows: int, columns: int) -> np.ndarray:
    matrix = np.zeros((rows, columns), dtype=int)
    row_limit = min(rows, DEFAULT_MATRIX.shape[0])
    column_limit = min(columns, DEFAULT_MATRIX.shape[1])
    matrix[:row_limit, :column_limit] = DEFAULT_MATRIX[:row_limit, :column_limit]
    return matrix


def result_frame(pairs):
    return pd.DataFrame(
        {
            "Пара": [f"{first}–{second}" for _, first, second in pairs],
            "Косинусное подобие": [round(similarity, 4) for similarity, _, _ in pairs],
        }
    )


st.set_page_config(page_title="Косинусное подобие", page_icon="📊", layout="wide")
st.title("Поиск похожих товаров и пользователей")
st.write("Введите матрицу предпочтений или загрузите файл. Строки соответствуют товарам, столбцы — пользователям.")

source = st.radio("Способ ввода", ["Ввести вручную", "Загрузить файл"], horizontal=True)
matrix = None

if source == "Ввести вручную":
    left, right = st.columns(2)
    with left:
        product_count = st.number_input("Количество товаров", min_value=3, max_value=50, value=6, step=1)
    with right:
        user_count = st.number_input("Количество пользователей", min_value=3, max_value=50, value=5, step=1)
    product_count = int(product_count)
    user_count = int(user_count)
    products = [f"P{i}" for i in range(1, product_count + 1)]
    users = [f"U{i}" for i in range(1, user_count + 1)]
    initial = pd.DataFrame(
        matrix_for_size(product_count, user_count),
        index=products,
        columns=users,
    )
    edited = st.data_editor(
        initial,
        width="stretch",
        num_rows="fixed",
        key=f"matrix_{product_count}_{user_count}",
    )
    matrix = edited.to_numpy(dtype=float)
else:
    uploaded = st.file_uploader("Файл с матрицей", type=["txt", "csv"])
    st.caption("Разделители: пробел, табуляция, запятая или точка с запятой.")
    if uploaded is not None:
        try:
            matrix = parse_matrix_text(decode_file(uploaded.getvalue()))
            products = [f"P{i}" for i in range(1, matrix.shape[0] + 1)]
            users = [f"U{i}" for i in range(1, matrix.shape[1] + 1)]
            st.dataframe(
                pd.DataFrame(matrix, index=products, columns=users),
                width="stretch",
            )
        except ValueError as error:
            st.error(str(error))

if st.button("Рассчитать", type="primary", width="stretch"):
    if matrix is None:
        st.error("Введите матрицу или загрузите файл")
    else:
        try:
            validate_matrix(matrix)
            products = [f"P{i}" for i in range(1, matrix.shape[0] + 1)]
            users = [f"U{i}" for i in range(1, matrix.shape[1] + 1)]
            product_pairs = find_max_sim(matrix, products, top_n=2)
            user_pairs = find_max_sim(matrix.T, users, top_n=2)
            st.success("Расчёт выполнен")
            product_column, user_column = st.columns(2)
            with product_column:
                st.subheader("Самые близкие товары")
                st.dataframe(result_frame(product_pairs), hide_index=True, width="stretch")
            with user_column:
                st.subheader("Самые близкие пользователи")
                st.dataframe(result_frame(user_pairs), hide_index=True, width="stretch")
        except ValueError as error:
            st.error(str(error))
