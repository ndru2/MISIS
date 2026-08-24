import argparse
import pdfplumber
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from unstructured.partition.pdf import partition_pdf
import re
from collections import Counter, defaultdict
import json
import PIL.Image
import os
import time
import sys
import difflib

import httpx

from dotenv import load_dotenv

from pdfscan import paths
from pdfscan.formulas.classifier import FormulaClassifier
from pdfscan.formulas.features import (STANDARD_REFERENCE_RE,
                                       is_technical_text_not_formula,
                                       looks_like_chemical_equation, prose_words)
from pdfscan.parse.export import export_blocks
from pdfscan.parse.text_layer import (detect_languages, element_bbox_in_pdf,
                                      rebind_elements)

def setup_apis():
    """Инициализация API ключей из файла .env"""
    config = {}
    try:
        # Явная загрузка .env файла
        load_dotenv()
            
        or_key = os.getenv("OPENROUTER_API_KEY")
        if or_key:
            config['openrouter_key'] = or_key
            print("✅ OpenRouter API ключ найден!")
        else:
            print("⚠️ OpenRouter API ключ не найден в .env файле.")
            
        return config
    except Exception as e:
        print(f"⚠️ Ошибка при настройке API: {e}")
        return config

def call_openrouter_visual_inspector(image_path, openrouter_key):
    """Вызов OpenRouter (модель poolside/laguna-m.1:free) для проверки разметки."""
    print(f"👁️ Запуск ИИ-инспектора OpenRouter для: {Path(image_path).name}...")
    
    # Модель poolside/laguna-m.1:free на данный момент может не поддерживать vision напрямую через API в бесплатном режиме или иметь специфичный формат.
    # Если модель текстовая, мы передадим описание того, что видим (на основе текущего df), 
    # НО пользователь просил именно для "генерации исправленных изображений" и "поправления ошибок".
    # Для бесплатной модели без Vision мы будем передавать текстовое описание извлеченных блоков.
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5173",
        "X-Title": "Smart PDF Extractor Inspector"
    }
    
    prompt = (
        "Ты — эксперт-лингвист и технический инспектор. Я пришлю тебе список извлеченных блоков текста из PDF. "
        "Твоя задача — найти ошибки классификации. "
        "Типы: NarrativeText (обычный текст), Formula (math), Formula (physics), Formula (chemistry).\n"
        "Ошибки для поиска:\n"
        "1. Формула (содержит знаки =, +, -, /, греческие буквы) помечена как NarrativeText.\n"
        "2. Обычный текст (просто слова) помечен как Formula.\n"
        "3. Перепутаны типы формул (например, химическая реакция Fe+O2 помечена как math).\n\n"
        "Верни ответ ТОЛЬКО в формате строгого JSON списка:\n"
        "[{\"element_id\": \"точная цитата начала блока\", \"correct_type\": \"правильный тип\", \"reason\": \"почему\"}]\n"
        "Если ошибок нет, верни []."
    )

    # Здесь мы будем передавать не картинку (так как laguna-m.1 текстовая), а данные из DF
    # Но так как метод вызывается внутри цикла по страницам, нам нужно получить данные текущей страницы.
    # Для этого передадим данные блоков в контексте.
    return [] # Заглушка, реальный вызов будет реализован в методе класса


# === НАСТРОЙКИ ПО ГОСТУ ===
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14


class SmartPDFExtractor:
    # ИИ-модель передается один раз при создании объекта
    def __init__(self, pdf_path, model=None, out_dir=None):
        self.pdf_path = Path(pdf_path)
        self.elements = []
        self.model = model
        self.text_layer_stats = {}
        self.languages = []
        # Все файлы одного документа складываются в его собственную папку:
        # при разборе сотен PDF общий каталог превращается в свалку.
        self.out_dir = Path(out_dir) if out_dir else Path('.')
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def extract(self):
        print(f"🚀 Запускаю unstructured для файла: {self.pdf_path.name}...")
        self.languages = detect_languages(self.pdf_path)
        print(f"🌐 Языки документа: {', '.join(self.languages)}")
        self.elements = partition_pdf(
            filename=str(self.pdf_path),
            strategy="hi_res",
            infer_table_structure=True,
            include_page_breaks=True,
            languages=self.languages,
        )
        print(f"📄 До графового слияния найдено элементов: {len(self.elements)}")
        self.rebind_text_layer()
        self.apply_graph_clustering()
        return self.elements

    def rebind_text_layer(self):
        """Этап 1.5: содержимое блоков берётся из текстового слоя документа.

        Макет, классы и рамки остаются от unstructured, но текст внутри рамок
        восстанавливается по символам PDF вместе с индексами и степенями.
        Для сканов ветка остаётся прежней — там результат OCR единственный
        доступный источник.
        """
        print("🔤 Ресвязывание блоков с текстовым слоем PDF...")
        try:
            stats = rebind_elements(self.elements, self.pdf_path)
        except Exception as e:
            print(f"⚠️ Ресвязывание пропущено, остаётся текст OCR: {e}")
            return

        self.text_layer_stats = stats
        digital, scanned = len(stats['digital_pages']), len(stats['scanned_pages'])
        print(f"   Страниц с текстовым слоем: {digital}, распознаваемых через OCR: {scanned}")
        print(f"   Восстановлено блоков: {stats['rebound']}, оставлено как есть: {stats['unchanged']}")
        if stats['image_blocks']:
            print(f"   ⚠️ Блоков-картинок на страницах с текстовым слоем: {stats['image_blocks']}"
                  f" (текст в них — ненадёжный OCR)")
        if stats['orphan_lines']:
            print(f"   ⚠️ Строк вне рамок unstructured: {stats['orphan_lines']}")
        if stats['cid_artifacts']:
            print(f"   ⚠️ Символов без таблицы юникода в шрифте: {stats['cid_artifacts']}")

    def build_nodes(self, page_els, page_size):
        """Узлы графа с координатами в пунктах PDF и настоящим кеглем шрифта.

        Пороги правил задаются в долях кегля, поэтому подставлять вместо него
        высоту рамки нельзя: у объединённого абзаца она равна высоте всего
        абзаца, и допуски раздуваются в разы.
        """
        page_width, page_height = page_size if page_size else (None, None)

        nodes = []
        for el in page_els:
            coords = getattr(el.metadata, 'coordinates', None)
            if coords and coords.points:
                pts = coords.points
                x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
                x1, y1 = max(p[0] for p in pts), max(p[1] for p in pts)
            else:
                x0 = y0 = x1 = y1 = 0

            bbox = element_bbox_in_pdf(el, page_width, page_height) if page_width else None
            px0, ptop, px1, pbottom = bbox if bbox else (x0, y0, x1, y1)

            info = getattr(el, 'text_layer', None) or {}
            nodes.append({
                # Исходная система координат нужна при пересчёте общей рамки.
                'el': el, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                'px0': px0, 'ptop': ptop, 'px1': px1, 'pbottom': pbottom,
                'font_size': info.get('font_size') or 0.0,
            })

        # Блокам без текстового слоя (картинки, сканы) даём медианный кегль
        # страницы — он ближе к истине, чем высота их рамки.
        known = sorted(n['font_size'] for n in nodes if n['font_size'] > 0)
        median = known[len(known) // 2] if known else 0.0
        for node in nodes:
            if node['font_size'] <= 0:
                node['font_size'] = median or max(node['pbottom'] - node['ptop'], 1.0)
        return nodes

    def column_right_edge(self, nodes):
        """Правый край наборной полосы по широким блокам страницы."""
        widths = [n['px1'] - n['px0'] for n in nodes]
        if not widths:
            return None
        page_span = max(widths)
        wide = sorted(n['px1'] for n in nodes if n['px1'] - n['px0'] > 0.6 * page_span)
        return wide[len(wide) // 2] if wide else None

    def blocks_between(self, nodes, n1, n2, font_size):
        """Стоит ли между двумя блоками третий, через который пойдёт слияние.

        Склейка через голову соседа рвёт порядок чтения: вводная фраза к формуле
        выпадает из потока, а общая рамка накрывает чужой текст.
        """
        top = n1['pbottom'] - 0.2 * font_size
        bottom = n2['ptop'] + 0.2 * font_size
        left, right = min(n1['px0'], n2['px0']), max(n1['px1'], n2['px1'])
        for other in nodes:
            if other is n1 or other is n2:
                continue
            if other['ptop'] < top or other['pbottom'] > bottom:
                continue
            if min(other['px1'], right) > max(other['px0'], left):
                return True
        return False

    def formula_text_is_usable(self, element, text):
        """Можно ли по тексту блока судить о разделе формулы.

        На скане результат OCR — единственный доступный источник, и он идёт в
        дело. А вот на странице с текстовым слоем OCR означает, что под рамкой
        картинка: там вместо формулы окажется мусор вроде ``а =В_. УОВ``.
        """
        if getattr(element, 'text_source', 'ocr') == 'ocr_fallback':
            return False
        # Одинокий номер формулы не несёт ни одного признака раздела.
        return not is_technical_text_not_formula(text)

    def is_formula_node(self, node):
        """Формульность блока: сперва по шрифтам, при их отсутствии — по тексту.

        Доля символов в математических гарнитурах — куда более надёжный
        признак, чем плотность спецсимволов: у формульных блоков она 0.09–0.80,
        у прозаических не превышает 0.01.
        """
        if 'Formula' in str(node['el'].category):
            return True
        info = getattr(node['el'], 'text_layer', None) or {}
        if info.get('math_font_ratio', 0.0) >= 0.08:
            return True
        return self.is_likely_formula(str(node['el'].text))

    def merge_text_layer(self, base_el, elements):
        """Сводит разбор текстового слоя объединяемых блоков в один."""
        sources = {getattr(el, 'text_source', 'ocr') for el in elements}
        base_el.text_source = sources.pop() if len(sources) == 1 else 'mixed'

        parts = [getattr(el, 'text_layer', None) for el in elements]
        parts = [p for p in parts if p]
        if not parts:
            return

        visible = sum(len(''.join(p['lines_plain'])) for p in parts) or 1
        sizes = sorted(p['font_size'] for p in parts if p['font_size'])
        base_el.text_layer = {
            'lines_latex': [line for p in parts for line in p['lines_latex']],
            'lines_plain': [line for p in parts for line in p['lines_plain']],
            'text_plain': ' '.join(p['text_plain'] for p in parts),
            'font_size': sizes[len(sizes) // 2] if sizes else 0.0,
            'n_subscript': sum(p['n_subscript'] for p in parts),
            'n_superscript': sum(p['n_superscript'] for p in parts),
            'math_font_ratio': sum(
                p['math_font_ratio'] * len(''.join(p['lines_plain'])) for p in parts) / visible,
            'italic_ratio': sum(
                p['italic_ratio'] * len(''.join(p['lines_plain'])) for p in parts) / visible,
            'fonts': dict(sum((Counter(p['fonts']) for p in parts), Counter())),
            'cid_artifacts': sum(p['cid_artifacts'] for p in parts),
        }

    def apply_graph_clustering(self):
        """
        Графовый метод: выстраивает взаимосвязь между пространственно близкими
        блоками и объединяет разрозненные слова в единые абзацы.
        Это радикально усиливает фактор контекста для классификатора.
        """
        print(" Запуск графового алгоритма связывания смежных блоков...")

        # УДАЛЯЕМ МУСОР: отсекаем блоки, где нет реального текста
        self.elements = [el for el in self.elements if str(el.text).strip()]

        page_sizes = {}
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for number, page in enumerate(pdf.pages, start=1):
                    page_sizes[number] = (page.width, page.height)
        except Exception as e:
            print(f"⚠️ Размеры страниц недоступны, пороги считаются по рамкам: {e}")

        pages = defaultdict(list)
        for el in self.elements:
            page = getattr(el.metadata, 'page_number', 1)
            pages[page].append(el)

        final_elements = []

        for page_num, page_els in pages.items():
            nodes = self.build_nodes(page_els, page_sizes.get(page_num))
            column_right = self.column_right_edge(nodes)

            # Сортировка узлов сверху-вниз, затем слева-направо. Шаг корзины
            # взят в половину строки, чтобы соседи по одной строке не меняли
            # порядок из-за разницы в пару пунктов.
            nodes.sort(key=lambda n: (round(n['ptop'] / 5), n['px0']))

            n_count = len(nodes)
            adj = {i: [] for i in range(n_count)}

            for i in range(n_count):
                # Ищем соседей в локальном окне (до 20 следующих элементов)
                for j in range(i + 1, min(i + 20, n_count)):
                    n1, n2 = nodes[i], nodes[j]

                    text1 = str(n1['el'].text).strip()
                    text2 = str(n2['el'].text).strip()

                    vertical_gap = n2['ptop'] - n1['pbottom']
                    font_size = (n1['font_size'] + n2['font_size']) / 2

                    is_f1 = self.is_formula_node(n1)
                    is_f2 = self.is_formula_node(n2)
                    is_fraction_bar = lambda t: re.fullmatch(r'[-—_]+', str(t).strip()) is not None

                    if vertical_gap > font_size * 0.3 and self.blocks_between(nodes, n1, n2, font_size):
                        continue

                    # Многоэтажные формулы (дроби, системы уравнений) склеиваем
                    # раньше прозаических правил: их строки часто короткие и
                    # заканчиваются точкой, из-за чего те правила их разрывают.
                    if (is_f1 or is_fraction_bar(text1)) and (is_f2 or is_fraction_bar(text2)):
                        # Ярусы одной формулы стоят вплотную, а соседние выключные
                        # формулы разнесены сильнее. Прежний допуск в три кегля
                        # был подогнан под высоту рамки вместо кегля и склеивал
                        # заведомо разные формулы.
                        if -font_size * 1.6 < vertical_gap < font_size * 1.6:
                            x_overlap = min(n1['px1'], n2['px1']) - max(n1['px0'], n2['px0'])
                            if x_overlap > -font_size * 2:  # Есть перекрытие или они очень близко по X
                                adj[i].append(j)
                                adj[j].append(i)
                                continue

                    if text1.endswith(('.', '!', '?', ';', ':')):
                        # Проверяем, что следующий начинается с заглавной
                        if text2 and text2[0].isupper():
                            continue  # Не склеиваем

                    # 3. Нумерованный список
                    if re.match(r'^\d+[\.\)]\s', text2):
                        continue

                    if vertical_gap > font_size * 1.2:  # Отступ больше высоты строки
                        continue

                    left_indent_diff = abs(n1['px0'] - n2['px0'])
                    if left_indent_diff > font_size * 4.0:
                        continue

                    if re.match(r'^\d{2,4}$', text2) or re.match(r'^P\s*\d+', text2):
                        continue

                    if n1['el'].category == 'Title' or n2['el'].category == 'Title':
                        if vertical_gap > font_size * 0.8:
                            continue

                    # Пропускаем элементы без валидных координат
                    if n1['px1'] == 0 or n2['px1'] == 0: continue

                    # Связываем только текстовые элементы
                    txt_cats = ['NarrativeText', 'UncategorizedText', 'Title']
                    if n1['el'].category not in txt_cats or n2['el'].category not in txt_cats:
                        continue

                    y_overlap = min(n1['pbottom'], n2['pbottom']) - max(n1['ptop'], n2['ptop'])
                    is_same_line = (y_overlap > -font_size * 0.3) and (n2['px0'] > n1['px0']) and (
                                n2['px0'] - n1['px1'] < font_size * 2.5)

                    is_next_line = (0 <= vertical_gap < font_size * 0.6) and (
                                abs(n1['px0'] - n2['px0']) < font_size * 4.0)

                    if is_next_line and not is_same_line:
                        # Защита от слипания разных абзацев: если следующая строка
                        # имеет отступ вправо (красная строка), это начало НОВОГО
                        # абзаца. Мы их не объединяем.
                        if n2['px0'] - n1['px0'] > font_size * 1.2:
                            is_next_line = False
                        # Строка, не дотянувшаяся до правого края полосы, завершает
                        # абзац. Проверка геометрическая, поэтому одинаково работает
                        # на русском и английском наборе.
                        elif column_right and column_right - n1['px1'] > font_size * 2.0:
                            is_next_line = False

                    # Если один блок - формула, а второй - текст, разрываем связь по вертикали
                    if (is_f1 and not is_f2) or (not is_f1 and is_f2):
                        if vertical_gap > font_size * 0.2: # Даже минимальный отступ означает, что это разные блоки
                            continue

                    if (is_same_line or is_next_line):
                        adj[i].append(j)
                        adj[j].append(i)

            # Поиск компонент связности (DFS)
            visited = set()
            components = []

            for i in range(n_count):
                if i not in visited:
                    comp = []
                    stack = [i]
                    while stack:
                        curr = stack.pop()
                        if curr not in visited:
                            visited.add(curr)
                            comp.append(curr)
                            stack.extend(adj[curr])
                    
                    # Восстанавливаем порядок чтения внутри объединенной компоненты
                    comp.sort(key=lambda idx: (round(nodes[idx]['ptop'] / 5), nodes[idx]['px0']))
                    components.append(comp)

            # Объединение элементов внутри каждой компоненты
            for comp in components:
                if len(comp) == 1:
                    final_elements.append(nodes[comp[0]]['el'])
                else:
                    base_el = nodes[comp[0]]['el']
                    merged_text = " ".join([str(nodes[idx]['el'].text).strip() for idx in comp])
                    merged_text = re.sub(r'\s+', ' ', merged_text)

                    base_el.text = merged_text
                    # Слияние строк одной многоэтажной формулы не должно
                    # превращать её в обычный текст.
                    categories = {str(nodes[idx]['el'].category) for idx in comp}
                    if all('Formula' in category for category in categories):
                        base_el.category = categories.pop() if len(categories) == 1 else 'Formula'
                    else:
                        base_el.category = 'NarrativeText'
                    self.merge_text_layer(base_el, [nodes[idx]['el'] for idx in comp])

                    # Пересчет общей рамки (Bounding Box) для визуализации
                    new_x0 = min(nodes[idx]['x0'] for idx in comp)
                    new_y0 = min(nodes[idx]['y0'] for idx in comp)
                    new_x1 = max(nodes[idx]['x1'] for idx in comp)
                    new_y1 = max(nodes[idx]['y1'] for idx in comp)

                    if hasattr(base_el.metadata, 'coordinates') and base_el.metadata.coordinates:
                        sys = base_el.metadata.coordinates.system
                        CoordClass = type(base_el.metadata.coordinates)
                        base_el.metadata.coordinates = CoordClass(
                            points=((new_x0, new_y0), (new_x1, new_y0), (new_x1, new_y1), (new_x0, new_y1)),
                            system=sys
                        )
                    final_elements.append(base_el)

        # Итоговая сортировка: по страницам, затем сверху вниз (по Y)
        self.elements = sorted(final_elements, key=lambda el: (
            getattr(el.metadata, 'page_number', 0) or 0,
            getattr(el.metadata, 'coordinates').points[0][1] if getattr(el.metadata, 'coordinates', None) and getattr(el.metadata, 'coordinates').points else 0
        ))
        print(f"✅ После графовой кластеризации объединено элементов: {len(self.elements)}")

    # Знак отношения или надстрочная разметка. Выключная формула почти всегда
    # содержит что-то из этого, а заголовок раздела — никогда.
    OPERATOR_RE = re.compile(r'[=≈→⟶↔<>≤≥∫∑√]|\\frac|[_^]\{')

    def is_formula_candidate(self, category_name, text):
        """Стоит ли разбирать блок как формулу.

        Макетная модель ставит выключным формулам класс по внешнему виду: они
        короткие и стоят по центру, поэтому регулярно уезжают в заголовки и
        подписи. Чтобы вернуть их, но не потащить за собой настоящие заголовки,
        от таких блоков дополнительно требуется знак отношения.
        """
        if category_name == 'Formula':
            return True
        if category_name in ('NarrativeText', 'UncategorizedText'):
            return self.is_likely_formula(text)
        if category_name in ('Title', 'FigureCaption'):
            return bool(self.OPERATOR_RE.search(text)) and self.is_likely_formula(text)
        return False

    def is_likely_formula(self, text):
        """Универсальный структурный анализатор. Ищет высокую концентрацию цифр и спецсимволов."""
        if not text or len(text) > 400:
            return False

        text = text.strip()
        clean_text = text.replace(" ", "")
        if not clean_text:
            return False

        if is_technical_text_not_formula(text):
            return False

        if re.match(r'^\d{2,4}$', text):
            return False
        if re.match(r'^[PpСс]\.?\s*\d+$', text):
            return False
        if re.match(r'^[-\d]+\s*[PpСс]\.?\s*\d*$', text):
            return False

        if re.search(r'[А-Я]{2,}\s*[\d.-]+', text):
            return False

        if re.search(r'[А-Я]{2,}\s*\d+', text):
            return False

        # Латинские прописные с числом трогать нельзя: так же выглядит «2CO -172»
        # в химическом уравнении, поэтому отсекаются только известные ссылки.
        if STANDARD_REFERENCE_RE.match(text):
            return False

        words = prose_words(text)
        if len(words) >= 3:
            letters = sum(1 for c in text if c.isalpha())
            if letters / len(text) > 0.4:
                return False

        tech_symbols = set('=+-*/^\\_→Δ∫∑°≈·|[]{}()<>!')
        tech_count = sum(1 for c in clean_text if c in tech_symbols)
        digit_count = sum(1 for c in clean_text if c.isdigit())
        alpha_count = sum(1 for c in clean_text if c.isalpha())
        cyrillic_count = sum(1 for c in text if 'А' <= c <= 'я')

        has_equation = any(c in clean_text for c in ['=', '→', '≈', '>', '<'])
        has_latex = any(c in clean_text for c in ['\\', '^', '_', '{', '}'])

        # Заголовок, набранный прописными, формулой не является. Но в записи
        # «S + O_{2} = SO_{2}» строчных букв тоже нет, поэтому выражение со
        # знаком отношения или разметкой под это правило не подпадает.
        if text.isupper() and len(text) > 10 and not (has_equation or has_latex):
            return False

        if alpha_count == 0:
            return (tech_count + digit_count) > 0

        density = (tech_count + digit_count) / alpha_count

        if has_latex and (tech_count + digit_count) >= 2:
            return True

        if re.fullmatch(r'\d+', clean_text) or re.fullmatch(r'[\(\[]\d+[\)\]]', clean_text):
            return False

        if cyrillic_count / len(text) > 0.5 and len(text) > 20:
            return False

        if density > 0.45:
            return True
        if has_equation and (tech_count + digit_count) > 0 and cyrillic_count < 10:
            return True
        if has_equation and density >= 0.2:
            return True
        if has_equation and len(text) < 80 and cyrillic_count < 15:
            return True
        if has_latex and (tech_count + digit_count) >= 1:
            return True
        return False

    def is_table_of_contents(self, elements):
        """Проверяет, является ли группа элементов оглавлением"""
        text = " ".join([str(el.text) for el in elements])

        # Признаки оглавления
        has_page_numbers = len(re.findall(r'\s+(\d+)\s*$', text, re.MULTILINE)) > 3
        has_dots = '...' in text or '…' in text
        has_chapter_pattern = re.search(r'(Глава|Chapter|Раздел|§)\s*\d+', text)

        # Не должно быть формул
        formula_count = sum(1 for el in elements if 'Formula' in str(el.category))
        table_count = sum(1 for el in elements if el.category == 'Table')

        # Оглавление, если есть страницы, точки/многоточие и НЕТ формул/таблиц
        if (has_page_numbers or has_dots or has_chapter_pattern) and formula_count == 0:
            return True

        return False

    def is_table_with_formulas(self, element):
        """Проверяет, содержит ли таблица формулы"""
        if element.category != 'Table':
            return False

        text = str(element.text)
        formula_indicators = sum([
            1 for c in text if c in '=+-*/^_→∫∑'
        ])

        if formula_indicators > 5:
            # Это не обычная таблица, а таблица с формулами
            element.category = 'FormulaTable'
            return True
        return False

    def to_dataframe(self):

        page_elements = defaultdict(list)
        for el in self.elements:
            page = getattr(el.metadata, 'page_number', 1)
            page_elements[page].append(el)

        for page, els in page_elements.items():
            # Поиск групп таблиц, которые могут быть оглавлением
            table_groups = []
            current_group = []

            for i, el in enumerate(els):
                if el.category == 'Table':
                    current_group.append(el)
                else:
                    if current_group and self.is_table_of_contents(current_group):
                        for table_el in current_group:
                            table_el.category = 'TableOfContents'
                    current_group = []

            if current_group and self.is_table_of_contents(current_group):
                for table_el in current_group:
                    table_el.category = 'TableOfContents'

        data = []
        formula_jobs = []

        for i, el in enumerate(self.elements):
            text = str(el.text).strip()
            if not text:
                continue

            category_name = el.category

            # Эвристика: Крупные графики (по размеру рамки)
            if hasattr(el.metadata, 'coordinates') and el.metadata.coordinates:
                pts = el.metadata.coordinates.points
                h = max(p[1] for p in pts) - min(p[1] for p in pts)
                if h > 200 and len(text) < 150:
                    category_name = 'Figure'
                    el.category = category_name
                    el.is_manually_corrected = True

            # Эвристика: Подписи к рисункам и таблицам
            if re.match(r'^(Рис\.|Рисунок|Таблица|Fig\.|Table)\s*\d+', text, re.IGNORECASE):
                category_name = 'Caption'
                el.category = category_name
                el.is_manually_corrected = True

            is_formula_candidate = self.is_formula_candidate(category_name, text)

            if getattr(el, 'is_manually_corrected', False):
                category_name = el.category
            elif is_formula_candidate and not self.formula_text_is_usable(el, text):
                # Содержимое формулы не извлеклось: под рамкой картинка, либо от
                # неё остался один номер. Раздел определять не по чему, поэтому
                # блок остаётся формулой без подтипа, а не получает выдуманный.
                category_name = 'Formula'
                el.category = category_name
            elif is_formula_candidate and looks_like_chemical_equation(text):
                # Запись, которая целиком раскладывается на вещества вокруг знака
                # реакции, — химия наверняка. Грамматика символов элементов здесь
                # надёжнее вероятностной оценки, поэтому решаем до модели.
                category_name = 'Formula (chemistry)'
                el.category = category_name
            elif is_formula_candidate and self.model:
                before_blocks = [str(self.elements[j].text) for j in range(max(0, i - 2), i)]
                context_before = " ".join(before_blocks)

                after_blocks = []
                for j in range(i + 1, min(len(self.elements), i + 6)):
                    after_text = str(self.elements[j].text).strip()
                    if not after_text:
                        continue
                    after_blocks.append(after_text)
                    if len(after_text) > 250:
                        break
                context_after = " ".join(after_blocks)
                context_combined = f"{context_before} {context_after}".strip()

                formula_jobs.append({
                    'index': i,
                    'text': text,
                    'context': context_combined,
                    'layout': getattr(el, 'text_layer', None),
                })
                category_name = '__pending_formula__'

            data.append({
                'page': getattr(el.metadata, 'page_number', 0),
                'type': category_name,
                'text_preview': text[:150] + '...' if len(text) > 150 else text,
                'text': text,
                'element_id': i,
            })

        if formula_jobs and self.model:
            try:
                predictions = self.model.predict_batch(
                    [job['text'] for job in formula_jobs],
                    [job['context'] for job in formula_jobs],
                    [job['layout'] for job in formula_jobs],
                )
                pred_map = {
                    job['index']: f"Formula ({pred})"
                    for job, pred in zip(formula_jobs, predictions)
                }
                for row in data:
                    idx = row['element_id']
                    if idx in pred_map:
                        row['type'] = pred_map[idx]
                        self.elements[idx].category = pred_map[idx]
            except Exception as e:
                print(f"Ошибка классификации: {e}")
                for row in data:
                    if row['type'] == '__pending_formula__':
                        row['type'] = 'Formula'

        df = pd.DataFrame(data)
        excel_filename = self.out_dir / 'blocks.xlsx'
        df.to_excel(excel_filename, index=False)
        print(f"💾 Сохранено: {excel_filename}")
        return df

    def visualize(self, page_number, output_image=None):
        page_blocks = [el for el in self.elements
                       if getattr(el.metadata, 'page_number', None) == page_number]

        if not page_blocks: return

        with pdfplumber.open(self.pdf_path) as pdf:
            if page_number > len(pdf.pages): return
            page = pdf.pages[page_number - 1]
            img = page.to_image(resolution=200)
            page_img = np.array(img.original)

        width_px, height_px = img.original.width, img.original.height
        fig, ax = plt.subplots(figsize=(10, 14.14), dpi=200)
        ax.imshow(page_img, origin='upper')
        ax.set_xlim(0, width_px); ax.set_ylim(height_px, 0)
        ax.set_autoscale_on(False)

        type_map = {
            'NarrativeText': 'text', 'UncategorizedText': 'text', 'Title': 'text',
            'Formula': 'formula', 'Formula (physics)': 'phys',
            'Formula (chemistry)': 'chem', 'Formula (math)': 'math',
            'Table': 'table', 'Caption': 'caption'
        }
        colors = {
            'text': '#4285F4', 'formula': '#34A853',
            'phys': '#0F9D58', 'chem': '#00E676', 'math': '#B9F6CA',
            'table': '#EA4335', 'caption': '#FBBC05'
        }

        for el in page_blocks:
            if not hasattr(el.metadata, 'coordinates') or not el.metadata.coordinates: continue
            coords = el.metadata.coordinates.points
            xs, ys = [p[0] for p in coords], [p[1] for p in coords]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            sys = el.metadata.coordinates.system
            u_w, u_h = (sys.width, sys.height) if sys and hasattr(sys, 'width') else (page.width, page.height)
            rx, rw = (x0 / u_w) * width_px, ((x1 - x0) / u_w) * width_px
            if sys and "Bottom" in sys.__class__.__name__:
                ry, rh = (1.0 - (y1 / u_h)) * height_px, ((y1 - y0) / u_h) * height_px
            else:
                ry, rh = (y0 / u_h) * height_px, ((y1 - y0) / u_h) * height_px

            el_type = type_map.get(el.category, 'text')
            color = colors.get(el_type, '#4285F4')
            ax.add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=2, edgecolor=color, facecolor=color, alpha=0.15))
            if rw > 40 and rh > 15:
                # Подписи выравниваем по левому краю страницы (x = 10)
                ax.text(10, ry, f" {el_type} ", color='black' if 'math' in el_type else 'white',
                        fontsize=10, weight='bold', bbox=dict(boxstyle='round', facecolor=color, alpha=0.9), ha='left', va='top')

        # Добавляем легенду цветов
        legend_elements = [
            patches.Patch(facecolor='#4285F4', edgecolor='#4285F4', alpha=0.5, label='Текст'),
            patches.Patch(facecolor='#34A853', edgecolor='#34A853', alpha=0.5, label='Формула (Общая)'),
            patches.Patch(facecolor='#0F9D58', edgecolor='#0F9D58', alpha=0.5, label='Формула (phys)'),
            patches.Patch(facecolor='#00E676', edgecolor='#00E676', alpha=0.5, label='Формула (chem)'),
            patches.Patch(facecolor='#B9F6CA', edgecolor='#B9F6CA', alpha=0.5, label='Формула (math)'),
            patches.Patch(facecolor='#EA4335', edgecolor='#EA4335', alpha=0.5, label='Таблица'),
            patches.Patch(facecolor='#FBBC05', edgecolor='#FBBC05', alpha=0.5, label='Подпись')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1), title="Типы блоков", frameon=True)

        ax.axis('off')
        if output_image:
            plt.savefig(output_image, dpi=200, bbox_inches='tight', pad_inches=0.1)
            print(f"🖼️ Изображение успешно обновлено: {Path(output_image).name}")
        plt.close()

    def inspect_with_openrouter(self, page_number, openrouter_key):
        """Инспекция разметки страницы с помощью текстовой модели OpenRouter."""
        if not openrouter_key: return []
        
        print(f"👁️ Запуск ИИ-инспектора OpenRouter для страницы {page_number}...")
        
        page_blocks = [el for el in self.elements 
                       if getattr(el.metadata, 'page_number', 1) == page_number]
        
        if not page_blocks: return []

        # Формируем компактное текстовое представление блоков для анализа
        blocks_data = []
        # Разрешенные категории для проверки ИИ (остальные он не трогает)
        allowed_cats = ['NarrativeText', 'UncategorizedText', 'Formula', 
                        'Formula (math)', 'Formula (physics)', 'Formula (chemistry)']
        
        for el in page_blocks:
            if el.category not in allowed_cats:
                continue # ПРОПУСКАЕМ Figure, Caption, Table, чтобы ИИ их не сломал
                
            text = str(el.text).strip()
            if not text: continue
            blocks_data.append({
                "id": self.elements.index(el),
                "current_type": el.category,
                "text": text[:200] # Даем достаточно контекста
            })

        prompt = (
            "Ты — эксперт-лингвист и технический инспектор. Ниже приведен список блоков, извлеченных из технического PDF-документа. "
            "Твоя задача — проверить правильность их классификации по типам: NarrativeText (текст), Formula (math), Formula (physics), Formula (chemistry).\n"
            "Признаки формул:\n"
            "- math: чистая математика, функции, суммы, интегралы.\n"
            "- physics: формулы с греческими буквами (sigma, tau), физическими константами, переменными (F, v, P, E).\n"
            "- chemistry: химические элементы (Fe, H2O, C, O2), знаки реакций (стрелки, плюс).\n"
            "ОШИБКОЙ считается, если сложная формула помечена как NarrativeText или если обычные слова помечены как Formula.\n\n"
            "Верни ответ ТОЛЬКО в формате строгого JSON списка, без лишнего текста:\n"
            "[{\"id\": 123, \"correct_type\": \"правильный тип\", \"reason\": \"почему\"}]\n"
            "Если всё верно, верни [].\n\n"
            f"ДАННЫЕ СТРАНИЦЫ {page_number}:\n"
            f"{json.dumps(blocks_data, ensure_ascii=False)}"
        )

        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "poolside/laguna-m.1:free",
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = httpx.post(url, headers=headers, json=data, timeout=60.0)
            res_json = response.json()
            
            if 'choices' in res_json:
                content = res_json['choices'][0]['message']['content']
                # Извлекаем JSON из ответа
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].strip()
                
                content = content.replace('\\', '\\\\')
                return json.loads(content)
        except Exception as e:
            print(f"⚠️ Ошибка при обращении к OpenRouter: {e}")
        
        return []

    def apply_corrections(self, corrections, df):
        if not corrections: return df
        print(f"🔧 Применение {len(corrections)} ИИ-исправлений...")
        dataset_file = "retrain_dataset.csv"

        for corr in corrections:
            idx = corr.get('id')
            correct_type = corr.get('correct_type', '')
            reason = corr.get('reason', '')

            if isinstance(idx, int) and 0 <= idx < len(self.elements):
                el = self.elements[idx]
                if el.category != correct_type:
                    print(f"   -> Исправлен блок[{str(el.text)[:30]}...]: {el.category} -> {correct_type} ({reason})")
                    el.category = correct_type
                    el.is_manually_corrected = True

                    data_row = pd.DataFrame([{'text': str(el.text), 'correct_class': correct_type, 'reason': reason}])
                    data_row.to_csv(dataset_file, mode='a' if os.path.exists(dataset_file) else 'w', header=not os.path.exists(dataset_file), index=False)

        return self.to_dataframe()

def collect_pdfs(targets):
    """Собирает документы из файлов и папок, обходя вложенные каталоги.

    Возвращает пары «путь к PDF, путь относительно корня поиска»: по второму
    строится папка результатов, поэтому исходное дерево папок сохраняется и
    одноимённые файлы из разных разделов не затирают друг друга.
    """
    found = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            for pdf in sorted(path.rglob('*.pdf')):
                found.append((pdf, pdf.relative_to(path)))
        elif path.is_file() and path.suffix.lower() == '.pdf':
            found.append((path, Path(path.name)))
        else:
            print(f'⚠️ Пропущено, не найдено: {target}')
    return found


def process_pdf(pdf_file, out_dir, model, or_key, render_pages=True, doc_id=None):
    """Полный разбор одного документа со всеми выходными файлами."""
    extractor = SmartPDFExtractor(pdf_file, model=model, out_dir=out_dir)
    extractor.extract()

    df = extractor.to_dataframe()
    pages = {int(getattr(el.metadata, 'page_number', 1)) for el in extractor.elements
             if getattr(el.metadata, 'page_number', None)}

    pages_dir = Path(out_dir) / 'pages'
    if render_pages:
        pages_dir.mkdir(parents=True, exist_ok=True)

    has_corrections = False
    all_corrections = []

    for p in sorted(pages):
        image_path = pages_dir / f'page{p}_initial.png'
        final_image_path = pages_dir / f'page{p}.png'
        if render_pages:
            extractor.visualize(p, str(image_path))

        if or_key:
            corrections = extractor.inspect_with_openrouter(p, or_key)
            if corrections:
                print(f'🔍 Найдено потенциальных исправлений: {len(corrections)}')
                all_corrections.extend(corrections)
                has_corrections = True

        if render_pages and not has_corrections and image_path.exists():
            image_path.replace(final_image_path)

    if has_corrections:
        print('🔄 Применение ИИ-исправлений и перерисовка...')
        df = extractor.apply_corrections(all_corrections, df)
        for p in sorted(pages):
            if render_pages:
                extractor.visualize(p, str(pages_dir / f'page{p}.png'))
            temp_img = pages_dir / f'page{p}_initial.png'
            if temp_img.exists():
                temp_img.unlink()

    # Выгрузка идёт последней: к этому моменту классы блоков окончательные,
    # включая правки ИИ-инспектора.
    export_blocks(extractor.elements, pdf_file, extractor.languages,
                  out_path=Path(out_dir) / 'blocks.jsonl', doc_id=doc_id)
    return extractor


def main():
    parser = argparse.ArgumentParser(
        description='Разбор PDF: блоки, классификация формул, выгрузка для RAG')
    parser.add_argument('targets', nargs='*', default=['.'],
                        help='файлы PDF или папки с ними (папки обходятся рекурсивно)')
    parser.add_argument('--out', default=str(paths.OUT_DIR),
                        help='корень для результатов; у каждого документа своя папка')
    parser.add_argument('--skip-existing', action='store_true',
                        help='пропускать документы, у которых уже есть blocks.jsonl')
    parser.add_argument('--no-images', action='store_true',
                        help='не рисовать страницы с рамками — заметно быстрее')
    parser.add_argument('--no-inspect', action='store_true',
                        help='пропустить ИИ-инспекцию: на пакетном прогоне это '
                             'обращение к сети на каждую страницу')
    args = parser.parse_args()

    api_config = setup_apis()
    or_key = None if args.no_inspect else api_config.get('openrouter_key')
    if args.no_inspect:
        print('⏭️ ИИ-инспекция отключена ключом --no-inspect')

    global_model = FormulaClassifier.load(paths.FORMULA_CLASSIFIER)
    if global_model:
        print('🧠 ИИ-классификатор формул (BERT + Boosting) успешно загружен!')
    else:
        print('⚠️ Модель не найдена. Запустите: python -m pdfscan.formulas.train')

    documents = collect_pdfs(args.targets)
    if not documents:
        print('❌ PDF не найдены.')
        sys.exit(1)
    print(f'📚 К разбору документов: {len(documents)}')

    root = Path(args.out)
    for position, (pdf_file, relative) in enumerate(documents, 1):
        out_dir = root / relative.parent / relative.stem
        if args.skip_existing and (out_dir / 'blocks.jsonl').exists():
            print(f'[{position}/{len(documents)}] уже разобран, пропуск: {relative}')
            continue

        print(f'\n📄 [{position}/{len(documents)}] Обработка: {relative}')
        try:
            process_pdf(pdf_file, out_dir, global_model, or_key,
                        render_pages=not args.no_images,
                        doc_id=str(relative.with_suffix('')))
        except Exception as error:
            # Один битый файл не должен останавливать разбор целой библиотеки.
            print(f'❌ Ошибка на {relative}: {type(error).__name__}: {error}')


if __name__ == '__main__':
    main()
