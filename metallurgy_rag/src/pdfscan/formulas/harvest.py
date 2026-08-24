"""Собирает обучающие примеры формул из реальных PDF.

Синтетический набор записан в LaTeX, а парсер выдаёт юникод с ``_{}``/``^{}``
и признаками начертания. Примеры из настоящих документов дают модели ту самую
запись, с которой она работает, вместе с разметкой шрифтов.

Метка проставляется двумя источниками. Химию определяет грамматика символов
элементов — она разбирает уравнение без остатка и не ошибается. Для остального
берётся предсказание текущей модели, а строки, где она не уверена или спорит с
грамматикой, помечаются как требующие проверки.
"""

import json
import sys

import pandas as pd
from unstructured.partition.pdf import partition_pdf

from pdfscan.formulas.classifier import FormulaClassifier
from pdfscan.formulas.features import looks_like_chemical_equation
from pdfscan.parse.extractor import SmartPDFExtractor
from pdfscan.parse.text_layer import rebind_elements
# Импорт по именам, а не модулем: у функций ниже свой параметр ``paths``, и
# модуль с тем же именем он бы перекрыл.
from pdfscan.paths import FORMULA_CLASSIFIER, HARVESTED_DATASET, PDF_DIR

CONFIDENT = 0.60


def context_around(elements, index, before=2, after=5):
    head = ' '.join(str(elements[j].text).strip()
                    for j in range(max(0, index - before), index))
    tail = []
    for j in range(index + 1, min(len(elements), index + after + 1)):
        text = str(elements[j].text).strip()
        if text:
            tail.append(text)
            if len(text) > 250:
                break
    return f"{head} {' '.join(tail)}".strip()


def main(paths, output=HARVESTED_DATASET):
    model = FormulaClassifier.load(FORMULA_CLASSIFIER)
    if model is None:
        print('⚠️ Модель не найдена, метки будут только от грамматики химии.')

    rows = []
    for pdf_path in paths:
        print(f'📄 {pdf_path}')
        extractor = SmartPDFExtractor(pdf_path)
        extractor.elements = partition_pdf(
            filename=pdf_path, strategy='hi_res', infer_table_structure=True,
            include_page_breaks=True, languages=['rus'],
        )
        rebind_elements(extractor.elements, pdf_path)
        extractor.apply_graph_clustering()

        for index, element in enumerate(extractor.elements):
            text = str(element.text).strip()
            category = str(element.category)
            candidate = 'Formula' in category or (
                category in ('NarrativeText', 'UncategorizedText')
                and extractor.is_likely_formula(text))
            # Проверка нужна и для блоков, которые unstructured сам назвал
            # формулой: одинокий номер «(3.7)» тоже приходит с этим классом.
            if not candidate or len(text) < 3:
                continue
            if not extractor.formula_text_is_usable(element, text):
                continue

            layout = getattr(element, 'text_layer', None)
            # Блоки с неразобранной кодировкой шрифта учить нельзя: модель
            # запомнит артефакты вместо математики.
            if (layout or {}).get('cid_artifacts'):
                continue

            context = context_around(extractor.elements, index)

            if looks_like_chemical_equation(text):
                label, source, confidence, review = 'chemistry', 'грамматика', 1.0, 0
            elif model is not None:
                probabilities = model.predict_proba(text, context, layout)
                label = max(probabilities, key=probabilities.get)
                confidence = probabilities[label]
                source = 'модель'
                # Метка от самой модели не может подтверждать саму себя,
                # поэтому в обучение такая строка попадёт только после того,
                # как человек проставит needs_review = 0.
                review = 1
            else:
                label, source, confidence, review = '', 'нет', 0.0, 1

            rows.append({
                'pdf': pdf_path,
                'page': getattr(element.metadata, 'page_number', 0),
                'text': text,
                'context': context,
                'layout': json.dumps(layout, ensure_ascii=False) if layout else '',
                'text_source': getattr(element, 'text_source', 'ocr'),
                'class': label,
                'label_source': source,
                'confidence': round(float(confidence), 3),
                'needs_review': review,
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    print(f'\n💾 Сохранено: {output} — примеров {len(frame)}')
    if not frame.empty:
        print(frame['class'].value_counts().to_string())
        print(f"Требуют проверки: {int(frame['needs_review'].sum())}")
        unreliable = int((frame['text_source'] != 'text_layer').sum())
        if unreliable:
            print(f"⚠️ С ненадёжным текстом от OCR: {unreliable} — их лучше исключить")


if __name__ == '__main__':
    # Без аргументов берём весь корпус: пилотные PDF, на которых это писалось,
    # в репозитории больше не лежат.
    main(sys.argv[1:] or [str(PDF_DIR)])
