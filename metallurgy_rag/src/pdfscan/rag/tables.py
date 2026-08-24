"""Разбор таблиц в вид, пригодный для поиска.

Таблица целиком плохо ложится в поиск: строка «4,7 5,0 6,4» сама по себе
бессмысленна, а по запросу «высота наплыва по расчёту МКЭ» нужна именно она.
Поэтому каждая строка разворачивается в самостоятельное предложение с
заголовками столбцов, а таблица целиком остаётся для показа человеку.
"""

import re
from html.parser import HTMLParser


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.header = []
        self._row = None
        self._cell = None
        self._in_head = False

    def handle_starttag(self, tag, attrs):
        if tag == 'thead':
            self._in_head = True
        elif tag == 'tr':
            self._row = []
        elif tag in ('td', 'th'):
            self._cell = []

    def handle_endtag(self, tag):
        if tag == 'thead':
            self._in_head = False
        elif tag in ('td', 'th') and self._cell is not None:
            if self._row is not None:
                self._row.append(_clean_cell(''.join(self._cell)))
            self._cell = None
        elif tag == 'tr' and self._row is not None:
            if self._in_head and not self.header:
                self.header = self._row
            elif any(self._row):
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _clean_cell(text: str) -> str:
    """Убирает следы распознавания: рамка колонки читается как вертикальная черта."""
    return re.sub(r'\s+', ' ', text).strip().strip('|').strip()


def parse_table_html(html: str) -> dict | None:
    if not html:
        return None
    parser = _TableParser()
    parser.feed(html)
    if not parser.rows and not parser.header:
        return None

    header = parser.header
    rows = parser.rows
    # Заголовок бывает не размечен и приходит первой строкой тела.
    if not header and rows:
        header, rows = rows[0], rows[1:]

    width = max([len(header)] + [len(r) for r in rows], default=0)
    header = (header + [''] * width)[:width]
    rows = [(row + [''] * width)[:width] for row in rows]
    return {'header': header, 'rows': rows}


def table_to_markdown(table: dict) -> str:
    """Таблица целиком — для показа человеку и для передачи модели."""
    if not table:
        return ''
    header, rows = table['header'], table['rows']
    lines = ['| ' + ' | '.join(header) + ' |',
             '| ' + ' | '.join('---' for _ in header) + ' |']
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows]
    return '\n'.join(lines)


def table_rows_as_text(table: dict) -> list[str]:
    """Разворачивает строки в самостоятельные предложения.

    Первый столбец обычно называет объект строки, поэтому он становится
    подлежащим, а остальные ячейки — парами «столбец: значение». Так строка
    находится по запросу целиком, без опоры на соседние.
    """
    if not table or not table['rows']:
        return []

    header = table['header']
    lines = []
    for row in table['rows']:
        subject = row[0] if row and row[0] else ''
        pairs = [f'{header[i]}: {row[i]}' if header[i] else row[i]
                 for i in range(1, len(row)) if row[i]]
        if not pairs:
            lines.append(subject)
        elif subject:
            lines.append(f'{subject} — ' + '; '.join(pairs))
        else:
            lines.append('; '.join(pairs))
    return lines
