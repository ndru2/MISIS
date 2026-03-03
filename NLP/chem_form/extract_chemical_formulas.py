import pymupdf as fitz
import json
import re
import os
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
from pdf2image import convert_from_path
import pytesseract


CHEMICAL_ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

# Химические элементы 
ELEMENT = r'(?:' + '|'.join(CHEMICAL_ELEMENTS) + r')'

# Индексы: цифры или в Unicode подстрочные символы
SUBSCRIPT = r'[\u2080-\u2089\d]+'

# Группа в скобках с возможным индексом
GROUP = r'\(' + ELEMENT + r'(?:' + SUBSCRIPT + r')?' + r'(?:' + ELEMENT + r'(?:' + SUBSCRIPT + r')?)*\)(?:' + SUBSCRIPT + r')?'

# Заряды ионов ^0-9, ^+- unicode
CHARGE = r'[\u2070-\u2079]+[\u207a\u207b]|[\d]+[+\-]'

# Коэффициенты 
COEFFICIENT = r'\d+\s*'

# хим формулы
CHEMICAL_FORMULA = re.compile(
    r'(?:' + COEFFICIENT + r')?'  # опциональный коэффициент
    + r'(?:' + GROUP + r'|' + ELEMENT + r'(?:' + SUBSCRIPT + r')?'  + r')+'  # элементы/группы/скобки
    + r'(?:' + CHARGE + r')?',  # опциональный заряд
    re.UNICODE
)

# стрелки между формулами (БЕЗ обычного знака равенства)
REACTION_ARROW = re.compile(r'[\u2192\u27f6\u21cc\u27f7\u2194\u21c4\u21c6=]')


STOPWORDS = {
    'А', 'В', 'С', 'Т', 'Р', 'N', 'M', 'К', 'E', 'I', 'V', 'U', 'R',
    'II', 'III', 'IV', 'VI', 'VII', 'VIII', 'IX', 'XI', 'XII',
    'SP', 'XP', 'DVD', 'ISBN', 'PDF', 'DOC', 'OS',
    # Не формулы но часто встретились
    'SS', 'OC', 'NI', 'CI', 'OF', 'IN', 'NC', 'ON', 'NO', 'OR',
    'AN', 'AS', 'AT', 'BE', 'BY', 'IF', 'IS', 'IT', 'OF', 'ON',
    'TO', 'UP', 'US', 'WE', 'HO', 'SO', 'DO', 'GO', 'HE', 'ME',
}


def normalize_subscript(text: str) -> str:
    """Преобразует обычные цифры в индексах в Unicode подстрочные."""
    subscript_map = str.maketrans(
        '0123456789',
        '\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089'
    )

    def replace_index(match):
        element = match.group(1)
        number = match.group(2)
        return element + number.translate(subscript_map)

    element_pattern = r'(' + '|'.join(CHEMICAL_ELEMENTS) + r')(\d+)'
    result = re.sub(element_pattern, replace_index, text)

    result = re.sub(r'(\))(\d+)', lambda m: m.group(1) + m.group(2).translate(subscript_map), result)

    return result


def is_valid_formula(formula: str) -> bool:
    formula = formula.strip()

    if len(formula) < 2 or len(formula) > 50:
        return False

    if formula in STOPWORDS:
        return False

    clean = re.sub(r'[(\)\[\]\d\u2080-\u2089]', '', formula)
    if clean in STOPWORDS:
        return False

    # Отфильтровываем формулы с слишком большими коэффициентами
    if re.search(r'\d{3,}', formula):
        return False

    # Отфильтровываем формулы начинающиеся с большого коэффициента без элемента
    if re.match(r'^\d+[A-Z]$', formula) and not re.match(r'^\d+[A-Z][a-z]', formula):
        return False

    has_index = bool(re.search(r'[\u2080-\u2089\d]', formula))
    has_brackets = '(' in formula or '[' in formula
    element_pattern = '|'.join(CHEMICAL_ELEMENTS)
    num_elements = len(re.findall(element_pattern, formula))
    has_multiple_elements = num_elements >= 2

    if not (has_index or has_brackets):
        if num_elements > 2:
            return False

    if not (has_index or has_brackets or has_multiple_elements):
        return False
    
    if formula[0].isdigit() and not re.match(r'\d+[A-Z]', formula):
        return False

    return True


def classify_formula(formula: str) -> str:
    """    
    Классы:
    1. simple_substance - простые вещества 
    2. oxide - оксиды 
    3. acid - кислоты 
    4. salt - соли 
    5. base_other - основания и прочие 
    """

    clean = re.sub(r'^\d+\s*', '', formula).strip()

    # Простые вещества: один элемент (с индексом или без)
    element_pattern = '(?:' + '|'.join(CHEMICAL_ELEMENTS) + ')'
    if re.fullmatch(element_pattern + r'[\u2080-\u2089\d]*', clean):
        return 'simple_substance'

    # Кислоты (неорганика)
    if clean.startswith('H') and clean not in ['H\u2082O', 'H\u2082O\u2082', 'H\u2082']:
        if 'O' in clean or 'Cl' in clean or 'Br' in clean or 'I' in clean:
            return 'acid'

    # Основания
    if '(OH)' in clean or re.search(r'OH[\u2080-\u2089\d]*(?:\)|$)', clean):
        return 'base_other'

    if clean in ['NH\u2083', 'NH\u2084', 'NH3', 'NH4']:
        return 'base_other'

    # Оксиды
    element_pattern = '|'.join(CHEMICAL_ELEMENTS)
    if re.search(r'O[\u2080-\u2089\d]*$', clean):

        num_elements = len(re.findall(element_pattern, clean))
        if num_elements == 2:

            return 'oxide'
        elif not re.search(element_pattern + r'(SO|CO|NO|PO)[\u2080-\u2089\d]+$', clean):

            return 'oxide'

    # Соли
    num_elements = len(re.findall(element_pattern, clean))
    if num_elements >= 2:
        return 'salt'

    return 'base_other'


def extract_chemical_formulas_from_text(text: str) -> list[tuple[str, str]]:
    results = []
    seen = set()

    for match in CHEMICAL_FORMULA.finditer(text):
        formula = match.group(0).strip()

        if not is_valid_formula(formula):
            continue
        formula = normalize_subscript(formula)

        if formula in seen:
            continue
        seen.add(formula)

        category = classify_formula(formula)
        results.append((formula, category))

    return results


def extract_reactions_from_text(text: str) -> list[str]:
    results = []
    seen = set()

    for line in text.split('\n'):
        line = line.strip()
        if not line or len(line) > 200:
            continue

        if not REACTION_ARROW.search(line):
            continue

        formulas = CHEMICAL_FORMULA.findall(line)
        if len(formulas) < 2:
            continue
        
        # хотя бы 2 валидных формулы
        valid_formulas = [f for f in formulas if is_valid_formula(f)]
        if len(valid_formulas) < 2:
            continue
        
        line_without_formulas = line
        for f in formulas:
            line_without_formulas = line_without_formulas.replace(f, '')
        # Если осталось больше 100 символов текста - это не реакция
        if len(line_without_formulas.strip()) > 100:
            continue

        reaction = normalize_subscript(line)

        # Очистка реакции от артефактов
        reaction = re.sub(r'\s*\(\d+\.?\d*\)\s*$', '', reaction)
        reaction = re.sub(r'[+\-]?\s*\d+[,\.]?\d*\s*(кДж|ккал|кДж/моль|ккал/моль|кВт|МДж|Дж|kJ|kcal|cal|J)\b.*$', '', reaction, flags=re.IGNORECASE)
        reaction = re.sub(r'\s*[;,]\s*[\u2206\u0394\u03b4]?[HGS]\u00b0?\s*[=<>].*$', '', reaction)
        reaction = re.sub(r'\s{2,}', ' ', reaction)
        reaction = re.sub(r'[.,;]\s*$', '', reaction)
        reaction = reaction.strip()
        
        
        if len(reaction) < 5:
            continue
        
        if not REACTION_ARROW.search(reaction):
            continue

        if reaction not in seen:
            seen.add(reaction)
            results.append(reaction)

    return results


def has_text_layer(pdf_path: str) -> bool:
    doc = fitz.open(pdf_path)

    for page_num in range(min(3, len(doc))):
        page = doc[page_num]
        text = page.get_text().strip()
        # Если на странице больше 100 символов текста - есть текстовый слой
        if len(text) > 100:
            doc.close()
            return True

    doc.close()
    return False


def extract_text_with_ocr(pdf_path: str) -> str:
    print(f"  [OCR] Конвертация PDF в изображения...")
    try:
        images = convert_from_path(pdf_path, dpi=300)
        print(f"  [OCR] Получено {len(images)} страниц")
    except Exception as e:
        print(f"  [ОШИБКА] Не удалось конвертировать PDF: {e}")
        return ""

    full_text = ""

    for i, image in enumerate(images, start=1):
        print(f"  [OCR] Распознавание страницы {i}/{len(images)}...", end=" ")

        try:
            page_text = pytesseract.image_to_string(
                image,
                lang='rus+eng',
                config='--psm 6'
            )

            full_text += f"\n{'='*60}\n"
            full_text += f"СТРАНИЦА {i}\n"
            full_text += f"{'='*60}\n\n"
            full_text += page_text

            print(f"({len(page_text)} символов)")

        except Exception as e:
            print(f"Ошибка: {e}")
            full_text += f"\n[ОШИБКА НА СТРАНИЦЕ {i}]\n"

    print(f"  [OCR] Итого распознано: {len(full_text)} символов")
    return full_text


def extract_from_text_pdf(pdf_path: str) -> tuple[list[dict], list[dict]]:
    # тип PDF
    print(f"  [INFO] Проверка типа PDF...", end=" ")
    has_text = has_text_layer(pdf_path)

    if has_text:
        doc = fitz.open(pdf_path)
        formulas = []
        reactions = []
        seen_formulas = set()
        seen_reactions = set()

        for page_num, page in enumerate(doc):
            for block in page.get_text("blocks"):
                block_text = block[4].strip()
                if not block_text:
                    continue

                for formula, category in extract_chemical_formulas_from_text(block_text):
                    if formula not in seen_formulas:
                        seen_formulas.add(formula)
                        formulas.append({
                            "source": os.path.basename(pdf_path),
                            "page": page_num + 1,
                            "formula": formula,
                            "category": category,
                            "method": "text_layer",
                        })

                for reaction in extract_reactions_from_text(block_text):
                    if reaction not in seen_reactions:
                        seen_reactions.add(reaction)
                        reactions.append({
                            "source": os.path.basename(pdf_path),
                            "page": page_num + 1,
                            "reaction": reaction,
                            "method": "text_layer",
                        })

        doc.close()
        return formulas, reactions

    else:
        # OCR
        full_text = extract_text_with_ocr(pdf_path)

        if not full_text:
            print(" OCR не дал результатов")
            return [], []

        formulas = []
        reactions = []
        seen_formulas = set()
        seen_reactions = set()

        for formula, category in extract_chemical_formulas_from_text(full_text):
            if formula not in seen_formulas:
                seen_formulas.add(formula)
                formulas.append({
                    "source": os.path.basename(pdf_path),
                    "page": None,
                    "formula": formula,
                    "category": category,
                    "method": "ocr",
                })

        for reaction in extract_reactions_from_text(full_text):
            if reaction not in seen_reactions:
                seen_reactions.add(reaction)
                reactions.append({
                    "source": os.path.basename(pdf_path),
                    "page": None,
                    "reaction": reaction,
                    "method": "ocr",
                })

        return formulas, reactions


def _omml_to_text(elem) -> str:
    """Извлекает текст из OMML-элемента (Office Math)."""
    parts = [node.text for node in elem.iter()
             if node.tag == qn('m:t') and node.text]
    return re.sub(r'\s{2,}', ' ', ' '.join(parts)).strip()


def extract_from_docx(docx_path: str) -> tuple[list[dict], list[dict]]:
    doc = Document(docx_path)
    formulas = []
    reactions = []
    seen_formulas = set()
    seen_reactions = set()

    for para_num, para in enumerate(doc.paragraphs):
        # OMML формулы
        for elem in para._element.findall('.//' + qn('m:oMath')):
            text = _omml_to_text(elem)

            for formula, category in extract_chemical_formulas_from_text(text):
                if formula not in seen_formulas:
                    seen_formulas.add(formula)
                    formulas.append({
                        "source": os.path.basename(docx_path),
                        "page": para_num,
                        "formula": formula,
                        "category": category,
                        "type": "omml",
                    })

            for reaction in extract_reactions_from_text(text):
                if reaction not in seen_reactions:
                    seen_reactions.add(reaction)
                    reactions.append({
                        "source": os.path.basename(docx_path),
                        "page": para_num,
                        "reaction": reaction,
                        "type": "omml",
                    })

        text = para.text.strip()
        for formula, category in extract_chemical_formulas_from_text(text):
            if formula not in seen_formulas:
                seen_formulas.add(formula)
                formulas.append({
                    "source": os.path.basename(docx_path),
                    "page": para_num,
                    "formula": formula,
                    "category": category,
                    "type": "text",
                })

        for reaction in extract_reactions_from_text(text):
            if reaction not in seen_reactions:
                seen_reactions.add(reaction)
                reactions.append({
                    "source": os.path.basename(docx_path),
                    "page": para_num,
                    "reaction": reaction,
                    "type": "text",
                })

    return formulas, reactions


def save_results(formulas: list[dict], reactions: list[dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    formulas_path = os.path.join(output_dir, "chemical_formulas.json")
    with open(formulas_path, 'w', encoding='utf-8') as f:
        json.dump(formulas, f, ensure_ascii=False, indent=2)
    print(f"Сохранено {len(formulas)} формул -> {formulas_path}")

    reactions_path = os.path.join(output_dir, "chemical_reactions.json")
    with open(reactions_path, 'w', encoding='utf-8') as f:
        json.dump(reactions, f, ensure_ascii=False, indent=2)
    print(f"Сохранено {len(reactions)} реакций -> {reactions_path}")


def print_stats(formulas: list[dict], reactions: list[dict]):
    print(f"Всего формул: {len(formulas)}")
    print(f"Всего реакций: {len(reactions)}")

    categories = {}
    for f in formulas:
        cat = f['category']
        categories[cat] = categories.get(cat, 0) + 1

    print("\nПо категориям:")
    cat_names = {
        'simple_substance': 'Простые вещества',
        'oxide': 'Оксиды',
        'acid': 'Кислоты',
        'salt': 'Соли',
        'base_other': 'Основания и другие',
    }
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat_names.get(cat, cat)}: {count}")
    sources = {}
    for f in formulas:
        src = f['source']
        sources[src] = sources.get(src, 0) + 1

    print("\nПо файлам (формулы):")
    for src, count in sources.items():
        print(f"  {src}: {count}")


if __name__ == "__main__":
    import sys

    DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "chemical_results")

    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "all":
        
            data_folder = Path(DATA_DIR)
            if data_folder.exists():
                all_files = sorted(list(data_folder.glob("*.pdf")) + list(data_folder.glob("*.docx")))

                if all_files:
                    print(f" Найдено файлов в папке {data_folder.absolute()}: {len(all_files)}")

                    max_files = int(os.getenv("MAX_FILES", "0"))
                    if max_files > 0:
                        print(f" Ограничение: обработка первых {max_files} файлов")
                        all_files = all_files[:max_files]

                    files_to_process = [f.name for f in all_files]
                    print(f"К обработке: {len(files_to_process)} файлов\n")
                else:
                    print(f" В папке {data_folder.absolute()} не найдено PDF или DOCX файлов!")
                    print("Поместите файлы в папку data/ и запустите снова.")
                    exit(1)
            else:
                print(f"Папка {data_folder.absolute()} не найдена!")
                print("Создайте папку data/ и поместите в неё PDF или DOCX файлы.")
                exit(1)
        else:
            files_to_process = sys.argv[1:]
            print(f"Обработка указанных файлов: {len(files_to_process)}")
            for f in files_to_process:
                print(f"  - {f}")
            print()
    else:
        print("=" * 50)
        print("\nИспользование:")
        print("  python extract_chemical_formulas.py all")
        print("    → Обработать все PDF и DOCX файлы из папки data/")
        print()
        print("  python extract_chemical_formulas.py file1.pdf file2.docx ...")
        print("    → Обработать конкретные файлы")
        print()
        print("Примеры:")
        print("  python extract_chemical_formulas.py all")
        print("  python extract_chemical_formulas.py document.pdf")
        print("  MAX_FILES=5 python extract_chemical_formulas.py all")
        print()
        print("Docker:")
        print("  docker-compose up --build")
        print("  MAX_FILES=3 docker-compose up --build")
        exit(0)
    
    if not files_to_process:
        print("Не найдено файлов для обработки!")
        exit(1)

    all_formulas = []
    all_reactions = []

    for filename in files_to_process:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"Файл не найден: {filename}")
            continue

        print(f"Обрабатываю: {filename}")
        ext = filepath.suffix.lower()

        if ext == '.pdf':
            formulas, reactions = extract_from_text_pdf(str(filepath))
        elif ext == '.docx':
            formulas, reactions = extract_from_docx(str(filepath))
        elif ext == '.doc':
            print(f"  Формат .doc не поддерживается. Конвертировать:")
            print(f"  libreoffice --convert-to docx \"{filepath}\"")
            continue
        else:
            print(f"  Неподдерживаемый формат: {ext}")
            continue

        print(f"  Найдено формул: {len(formulas)}, реакций: {len(reactions)}")
        all_formulas.extend(formulas)
        all_reactions.extend(reactions)

    save_results(all_formulas, all_reactions, OUTPUT_DIR)
    print_stats(all_formulas, all_reactions)
