"""Чтение вопросов из предоставленной экспертной Excel-таблицы."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def load_questions(workbook: Union[str, Path], *, sheet='Лист1') -> list[dict]:
    frame = pd.read_excel(workbook, sheet_name=sheet)
    if 'Вопрос' not in frame.columns:
        raise ValueError(f'В листе {sheet!r} не найдена колонка «Вопрос»')
    items = []
    for index, row in frame.iterrows():
        question = str(row.get('Вопрос') or '').strip()
        if not question or question.lower() == 'nan':
            continue
        number = row.get('№')
        items.append({'id': str(number) if pd.notna(number) else str(index + 1),
                      'question': question})
    return items
