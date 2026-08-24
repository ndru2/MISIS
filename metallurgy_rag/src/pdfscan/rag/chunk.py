"""Семантический чанкинг: сборка блоков в куски для поиска.

Резать текст по числу символов нельзя: формула, оторванная от фразы, которая её
вводит, бесполезна — по ней не понять, что за величины в ней стоят. Поэтому
границы кусков проходят по структуре документа, а формула всегда остаётся вместе
с окружающим объяснением.
"""

import re

from pdfscan.rag.normalize import clean_text, normalize_record
from pdfscan.rag.tables import (parse_table_html, table_rows_as_text,
                                table_to_markdown)

# Колонтитулы и подписи под рисунками не несут содержания, а номер страницы в
# середине куска только мешает поиску.
FURNITURE_TYPES = {'Header', 'Footer', 'PageBreak'}
NOISE_TYPES = {'Figure'}

# Заголовок раздела: короткая строка с номером или набранная прописными. У
# блоков класса Title это выполняется далеко не всегда — там же оказываются
# пояснения к формулам вроде «где E_в – модуль упругости материала винта».
_NUMBERED_RE = re.compile(r'^\d+(?:\.\d+)*\.?\s+\S')
_NAMED_RE = re.compile(
    r'^(?:глава|раздел|часть|лабораторная\s+работа|приложение|введение|заключение'
    r'|chapter|section|part|appendix|introduction|conclusion)\b', re.IGNORECASE)


def is_section_heading(record) -> bool:
    text = (record.get('text_clean') or record.get('text') or '').strip()
    if record.get('type') != 'Title' or not text or len(text) > 120:
        return False
    # Пояснение к формуле кончается двоеточием или точкой с запятой и почти
    # всегда содержит знак тире между обозначением и его расшифровкой.
    if text.endswith((':', ';', ',')) or ' – ' in text or ' - ' in text:
        return False
    # Формулу, которую макетная модель приняла за заголовок, разделом считать
    # нельзя: она станет подписью ко всем последующим кускам.
    if re.search(r'[=→≈∫∑]|\\frac|[_^]\{', text):
        return False
    if _NUMBERED_RE.match(text) or _NAMED_RE.match(text):
        return True
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.7


def _is_furniture(record) -> bool:
    if record['type'] in NOISE_TYPES:
        return True
    if record['type'] not in FURNITURE_TYPES:
        return False
    # Колонтитул с названием главы стоит сохранить как подсказку о разделе, а
    # голый номер страницы — нет.
    text = (record.get('text_clean') or record['text']).strip()
    return not re.search(r'[A-Za-zА-Яа-яЁё]{4,}', text)


class _Tokenizer:
    """Счётчик длины в токенах модели, с запасным вариантом по словам."""

    def __init__(self, name=None):
        self.tokenizer = None
        if name:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(name)
            except Exception:
                self.tokenizer = None

    def count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer(text, add_special_tokens=False)['input_ids'])
        # Оценка сверху: для кириллицы модель обычно даёт около трети символов.
        return max(1, len(text) // 3)


_SENTENCE_RE = re.compile(r'(?<=[.!?;])\s+|\n+')


def _sentences(text, max_tokens, counter):
    """Предложения текста, а слишком длинное — по словам.

    Предложение длиннее бюджета встречается там, где распознавание не поставило
    ни одной точки: страница приходит одной строкой на несколько тысяч символов.
    Доля от целого считается один раз на всё предложение — звать токенизатор на
    каждое слово вышло бы дороже самой резки.
    """
    for sentence in _SENTENCE_RE.split(text or ''):
        sentence = sentence.strip()
        if not sentence:
            continue
        tokens = counter.count(sentence)
        if tokens <= max_tokens:
            yield sentence
            continue
        words = sentence.split()
        # Целимся в четыре пятых бюджета: плотность токенов по тексту неровная,
        # и без запаса отдельные части вылезли бы за предел снова.
        step = max(1, len(words) * max_tokens * 4 // (tokens * 5))
        for start in range(0, len(words), step):
            yield ' '.join(words[start:start + step])


def split_long(text, max_tokens, counter) -> list:
    """Режет текст, который сам по себе не влезает в бюджет.

    Такой блок появляется, когда на странице нет ни заголовков, ни абзацных
    отбивок, по которым макетная модель могла бы её разделить, и вся страница
    приходит одним блоком. Оставить его целым нельзя: векторизатор обрезает вход
    по своему пределу молча, и от страницы в индекс попала бы первая четверть.
    """
    parts, current, current_tokens = [], [], 0
    for piece in _sentences(text, max_tokens, counter):
        tokens = counter.count(piece)
        if current and current_tokens + tokens > max_tokens:
            parts.append(' '.join(current))
            current, current_tokens = [], 0
        current.append(piece)
        current_tokens += tokens
    if current:
        parts.append(' '.join(current))
    return parts or ['']


def _retext(record, text) -> dict:
    """Копия записи с подменённым текстом — для частей разрезанного блока."""
    piece = dict(record)
    piece['text_clean'] = text
    piece['text_search'] = text
    return piece


def _lead_in(record, budget, counter):
    """Хвост предыдущего блока — фраза, которая вводит формулу.

    Переносить абзац целиком нельзя: вместе с формулой он даёт кусок вдвое
    больше бюджета, и векторизатор отрезает от него ровно то, ради чего перенос
    и делался. Берутся последние предложения, сколько влезает в остаток.
    """
    if budget <= 0:
        return None

    kept, size = [], 0
    for sentence in reversed(list(_sentences(
            record['text_clean'], budget, counter))):
        tokens = counter.count(sentence)
        if kept and size + tokens > budget:
            break
        if not kept and tokens > budget:
            return None
        kept.insert(0, sentence)
        size += tokens

    if not kept:
        return None
    return _retext(record, ' '.join(kept)), size


def _table_chunks(record, max_tokens, counter):
    """Таблица целиком, а если не помещается — по строкам с общей шапкой."""
    table = parse_table_html(record.get('table_html'))
    if not table:
        # Разметки нет — резать по строкам не по чему, остаётся текст как есть,
        # и его тоже надо уложить в бюджет.
        return split_long(record.get('text_clean') or record['text'],
                          max_tokens, counter)

    markdown = table_to_markdown(table)
    if counter.count(markdown) <= max_tokens:
        return [markdown]

    header_line = '| ' + ' | '.join(table['header']) + ' |'
    parts, current = [], []
    for line in table_rows_as_text(table):
        candidate = '\n'.join(current + [line])
        if current and counter.count(header_line + '\n' + candidate) > max_tokens:
            parts.append(header_line + '\n' + '\n'.join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append(header_line + '\n' + '\n'.join(current))

    # Одна строка таблицы тоже бывает длиннее бюджета — в широких таблицах с
    # текстовыми примечаниями. Резать её по словам некрасиво, но лучше, чем
    # отдать векторизатору хвост на отсечение.
    sized = []
    for part in parts:
        if counter.count(part) <= max_tokens:
            sized.append(part)
        else:
            sized.extend(split_long(part, max_tokens, counter))
    return sized


def build_chunks(records, model_name=None, max_tokens=400, lead_in=True):
    """Собирает блоки документа в куски, не разрывая формулы и таблицы.

    ``max_tokens`` считается по токенизатору модели, которой куски пойдут на
    векторизацию: обрезка на её стороне молча выбросила бы хвост.
    """
    counter = _Tokenizer(model_name)
    records = [normalize_record(dict(r)) for r in records]

    chunks = []
    section = ''
    current = []          # список записей текущего куска
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if current:
            chunks.append(_assemble(current, section, counter))
        current, current_tokens = [], 0

    for index, record in enumerate(records):
        if _is_furniture(record):
            continue

        if is_section_heading(record):
            flush()
            section = record['text_clean']
            continue

        if 'Table' in record['type']:
            flush()
            for part in _table_chunks(record, max_tokens, counter):
                chunks.append(_assemble(
                    [_retext(record, part)], section, counter))
            continue

        text = record['text_clean']
        tokens = counter.count(text)
        is_formula = 'Formula' in record['type']

        # Блок, который сам не влезает в бюджет, режется на части. Формулу не
        # трогаем: разорванная запись теряет смысл целиком, а не наполовину, и
        # формул такой длины в корпусе не бывает.
        if tokens > max_tokens and not is_formula:
            flush()
            parts = split_long(text, max_tokens, counter)
            for part in parts[:-1]:
                chunks.append(_assemble(
                    [_retext(record, part)], section, counter))
            # Последняя часть кусок не закрывает: если дальше идёт формула,
            # вводит её именно она, и отрывать их друг от друга незачем.
            current = [_retext(record, parts[-1])]
            current_tokens = counter.count(parts[-1])
            continue

        if current and current_tokens + tokens > max_tokens:
            # Формула без вводной фразы бесполезна: по ней не понять, что за
            # величины в ней стоят. Поэтому граница проходит не перед формулой,
            # а перед блоком, который её вводит, и этот блок уходит в новый
            # кусок вместе с ней. Раньше здесь граница просто не ставилась, и
            # на списке реакций кусок разрастался без предела.
            next_is_formula = (index + 1 < len(records)
                               and 'Formula' in records[index + 1]['type'])
            keeps_lead = lead_in and (is_formula or next_is_formula)
            tail = (current[-1] if keeps_lead
                    and 'Formula' not in current[-1]['type'] else None)
            flush()
            lead = _lead_in(tail, max_tokens - tokens, counter) if tail else None
            if lead is not None:
                current, current_tokens = [lead[0]], lead[1]

        current.append(record)
        current_tokens += tokens

    flush()

    for order, chunk in enumerate(chunks):
        chunk['chunk_id'] = f"{chunk['doc_id']}#c{order}"
        chunk['order'] = order
    return chunks


def _join(parts):
    """Склеивает блоки: оборванную строку — пробелом, законченную — переводом.

    Кластеризация сшивает не все строки абзаца, и без этого правила предложение
    разрывалось бы посреди слова там, где кончилась строка исходной вёрстки.
    Выключная формула всегда стоит отдельной строкой — так её видно и человеку,
    и модели.
    """
    joined = ''
    previous_formula = False
    for text, is_formula in parts:
        if not text:
            continue
        if not joined:
            joined = text
        elif is_formula or previous_formula or joined.rstrip().endswith(
                ('.', ':', ';', '!', '?', '»')):
            joined += '\n' + text
        else:
            joined += ' ' + text
        previous_formula = is_formula
    return joined


def _assemble(records, section, counter):
    """Складывает записи в один кусок и сводит их метаданные."""
    display = _join([(r['text_clean'], 'Formula' in r['type']) for r in records])
    body = _join([(r['text_search'], 'Formula' in r['type']) for r in records])

    # Название раздела приписывается к тексту для поиска: по нему кусок
    # находится вместе со своим контекстом, даже если внутри раздел не назван.
    search = f'{section}\n{body}' if section else body

    units, elements, names = [], [], []
    for record in records:
        units.extend(record.get('units', []))
        elements.extend(record.get('elements', []))
        names.extend(record.get('element_names', []))

    pages = sorted({r['page'] for r in records})
    return {
        'doc_id': records[0]['doc_id'],
        'section': section,
        'pages': pages,
        'page': pages[0] if pages else 0,
        'text': display,
        'text_search': clean_text(search),
        'block_ids': [r['block_id'] for r in records],
        'types': sorted({r['type'] for r in records}),
        'has_formula': any('Formula' in r['type'] for r in records),
        'has_table': any('Table' in r['type'] for r in records),
        'has_chemistry': any('chemistry' in r['type'] for r in records),
        'reliable': all(r['reliable'] for r in records),
        'units': units,
        'elements': sorted(set(elements)),
        'element_names': sorted(set(names)),
        'n_tokens': counter.count(search),
    }
