"""Раскладка разбора MinerU по категориям: текст, формулы, таблицы.

RAG-индекс строится по разнородному материалу: связный текст ищется эмбеддингом
по смыслу, уравнение — по составу веществ, таблица — по заголовкам колонок. В
``*_content_list.json`` всё это лежит вперемешку одним списком, и разделение
здесь нужно, чтобы каждую категорию потом разбивать на чанки своим способом.

Формулы MinerU не делит на химию и математику, поэтому уравнения проходят через
обученный классификатор (``models/formula_classifier.pkl``). У него три класса,
физика с математикой сведены в один каталог: в металлургии почти всякая
неорганическая формула физическая по смыслу, и граница между ними проходит не
там, где полезно для поиска.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from pdfscan import paths
from pdfscan.formulas.features import (
    AMBIGUOUS_ELEMENTS,
    chemical_parse,
    is_technical_text_not_formula,
    looks_like_chemical_equation,
    normalize_notation,
    prose_words,
)
from pdfscan.parse.mineru import repair_mineru_latex
from pdfscan.rag.normalize import clean_text, flatten_formula, latinize_chemistry

CATEGORIES = ('text', 'formulas_chem', 'formulas_math', 'tables')

# Колонтитулы и номера страниц режутся здесь, а не на этапе чанкинга: они
# попадают между абзацами и рвут связный текст пополам.
FURNITURE_TYPES = frozenset({'header', 'footer', 'page_number'})

# Всё, что читается как проза. Подпись к рисунку и сноска отделены от основного
# текста пометкой ``subtype``: их стоит индексировать, но не склеивать с абзацем.
TEXT_TYPES = frozenset({'text', 'aside_text', 'page_footnote', 'ref_text', 'code'})

# Картинки и графики остаются в разборе MinerU: без распознавания содержимого
# индексировать нечего, а подписи к ним приходят отдельными текстовыми блоками.
SKIP_TYPES = frozenset({'image', 'chart', 'equation_image'})

# Уравнение почти всегда объясняется соседними абзацами, и классификатор обучен
# читать их вместе с формулой. Двух блоков хватает, чтобы поймать «где ...» с
# расшифровкой обозначений.
CONTEXT_WINDOW = 2
CONTEXT_MAX_CHARS = 400

# Класс модели → каталог. Физика и математика сведены намеренно, см. docstring.
CLASS_TO_CATEGORY = {
    'chemistry': 'formulas_chem',
    'math': 'formulas_math',
    'physics': 'formulas_math',
}


_TAG_RE = re.compile(r'\\tag\{([^{}]*)\}')
_WHITESPACE_RE = re.compile(r'\s+')
# Знак отношения делает запись формулой, даже если обе части набраны словами:
# «Aspect Ratio = Major Axis / Minor Axis» — это формула, а не абзац.
_RELATION_RE = re.compile(r'[=<>≈≤≥]|→|⟶|↔|⇄|⇌|⇒')
_OPERATOR_RE = re.compile(r'\\frac|\\sqrt|\\sum|\\int|[+×÷/^]|(?<![,.\d])\d')
# Расшифровка обозначения: слева одиночный символ, справа его словесное имя.
_DEFINITION_RE = re.compile(r'^(.{1,14}?)\s*[=—–-]\s*(.+)$', re.DOTALL)


def _prose_share(text: str) -> float:
    """Доля записи, занятая самостоятельными словами."""
    if not text:
        return 0.0
    return sum(len(word) for word in prose_words(text)) / len(text)


def canonical_latex(raw: str) -> tuple[str, str]:
    """Запись формулы без обрамления и номера плюс сам номер.

    ``repair_mineru_latex`` снимает ``$$`` и сжимает разрядку MinerU: парсер
    выдаёт ``\\mathrm{O} _ {2}``, и в таком виде вещество не читается ни разбором
    по символам элементов, ни поиском по запросу «O2».
    """
    tex = repair_mineru_latex(raw)
    match = _TAG_RE.search(tex)
    number = match.group(1).strip('[]() ') if match else ''
    return _TAG_RE.sub('', tex).strip(), number


def is_chemical(latex: str) -> bool:
    """Разбор записи по символам элементов на записи без пробелов.

    Пробелы между веществом и его индексом (``Cu_{2} O``) разрывают вещество
    пополам, и разбор не проходит; без них ``Cu_{2}O`` читается целиком.
    """
    return looks_like_chemical_equation(_WHITESPACE_RE.sub('', latex))


def has_chemical_species(latex: str) -> bool:
    """Есть ли в записи хоть одно вещество, разобранное по символам элементов.

    Проверка нужна как условие поверх модели. Классификатор обучен на
    синтетических примерах из учебников и на металлургическом корпусе охотно
    зовёт химией всё подряд: на размеченной выборке из его «уверенной химии»
    веществами оказались 39%, остальное — теплофизика, кинетика и балансы. Ни
    одно вещество не разбирается — значит, реакции нет, чем бы модель это ни
    считала. Условие мягче ``is_chemical``: одного вещества хватает, знак
    реакции и вторая часть уравнения не требуются.

    Одиночная буква вещество не образует: в ``R = H = 2·13,2/3,14`` латинская
    «H» совпадает с водородом, и без этой оговорки арифметика прошла бы за
    химию. Нужен либо индекс в составе, либо символ, который ни с чем не
    спутать.
    """
    parsed = chemical_parse(_WHITESPACE_RE.sub('', latex))
    if not parsed['species']:
        return False
    return parsed['subscripted'] or any(
        element not in AMBIGUOUS_ELEMENTS for element in parsed['elements'])


def prose_subtype(latex: str) -> str | None:
    """Отличает прозу, размеченную MinerU как формулу, от настоящей формулы.

    На вход идёт запись без обрамления ``$$``: разбор по символам элементов
    принимает голую формулу, а ``$$`` прилипло бы к первому веществу.

    Распознаватель формул срабатывает и на обычных строках — «Приход тепла»,
    «aqueous pregnant leach solution» — и на расшифровках обозначений вида
    ``g = ускорение свободного падения``. В каталоге формул они мешают: поиск по
    составу веществ их никогда не найдёт, а как текст они читаются нормально.

    Возвращает подтип для каталога ``text`` либо ``None``, если запись остаётся
    формулой. Словесные формулы (``Aspect Ratio = Major Axis / Minor Axis``)
    остаются формулами: у них есть и знак отношения, и оператор.
    """
    if is_chemical(latex):
        return None

    normalized = normalize_notation(latex)
    if not normalized or _prose_share(normalized) <= 0.5:
        # Ниже порога проза встречается только в расшифровках обозначений,
        # где короткий символ слева сбивает долю слов по всей записи.
        match = _DEFINITION_RE.match(normalized)
        if (match and not _OPERATOR_RE.search(match.group(2))
                and _prose_share(match.group(2)) > 0.45
                and len(prose_words(match.group(2))) >= 2
                and not prose_words(match.group(1))):
            return 'nomenclature'
        return None

    if _RELATION_RE.search(normalized) and _OPERATOR_RE.search(normalized):
        return None
    return 'nomenclature' if _RELATION_RE.search(normalized) else 'caption'


def _document_dirs(root: Path) -> list[Path]:
    """Каталоги ``hybrid_auto`` с разбором, по одному на документ."""
    return sorted((path.parent for path in root.glob('*/hybrid_auto/*_content_list.json')),
                  key=str)


def _content_list(doc_dir: Path) -> tuple[Path, list[dict]]:
    """Плоский список блоков. Версия v2 сгруппирована по страницам и не подходит."""
    candidates = [path for path in sorted(doc_dir.glob('*_content_list.json'))
                  if not path.name.endswith('_content_list_v2.json')]
    if not candidates:
        raise FileNotFoundError(f'нет content_list.json в {doc_dir}')
    path = candidates[0]
    with path.open(encoding='utf-8') as handle:
        return path, json.load(handle)


def _topic(doc_id: str) -> str:
    """Тематический раздел из имени папки: «Cu-Co сырье__Нкана смелтер»."""
    return doc_id.split('__', 1)[0].strip() if '__' in doc_id else ''


def _join_captions(item: dict, *keys: str) -> str:
    parts = []
    for key in keys:
        value = item.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(part) for part in value)
    return clean_text(' '.join(part for part in parts if part))


def _context_for(items: list[dict], position: int) -> str:
    """Соседняя проза вокруг блока — вход классификатора и подсказка для поиска."""
    start = max(0, position - CONTEXT_WINDOW)
    around = items[start:position] + items[position + 1:position + 1 + CONTEXT_WINDOW]
    parts = [item.get('text', '') for item in around if item.get('type') in TEXT_TYPES]
    return clean_text(' '.join(part for part in parts if part))[:CONTEXT_MAX_CHARS]


def _base_record(doc_id: str, item: dict, order: int) -> dict:
    return {
        'doc_id': doc_id,
        'topic': _topic(doc_id),
        'order': order,
        'page': item.get('page_idx'),
        'bbox': item.get('bbox'),
        'mineru_type': item.get('type'),
    }


def _iter_documents(doc_dirs, classify, progress):
    """Разбор документов по одному; наружу отдаётся готовая раскладка."""
    total = len(doc_dirs)
    for position, doc_dir in enumerate(doc_dirs, 1):
        doc_id = doc_dir.parent.name
        try:
            _, items = _content_list(doc_dir)
        except (FileNotFoundError, json.JSONDecodeError) as error:
            print(f'  пропускаю {doc_id}: {error}')
            continue

        buckets, stats = _split_document(doc_id, items, classify)
        if progress and (position % 25 == 0 or position == total):
            print(f'  обработано {position}/{total} документов')
        yield doc_id, buckets, stats


def _split_document(doc_id: str, items: list[dict], classify) -> tuple[dict, Counter]:
    """Раскладывает блоки одного документа по категориям."""
    buckets = {name: [] for name in CATEGORIES}
    stats = Counter()
    # Заголовки идут отдельными блоками с ``text_level``; запоминаем последний,
    # чтобы у формулы и таблицы был раздел, из которого их достали.
    section = ''
    pending_formulas = []

    for order, item in enumerate(items):
        kind = item.get('type')
        stats[f'in:{kind}'] += 1

        if kind in FURNITURE_TYPES or kind in SKIP_TYPES:
            stats['skipped'] += 1
            continue

        if kind in TEXT_TYPES:
            text = clean_text(item.get('text') or '')
            if not text:
                stats['empty'] += 1
                continue
            level = item.get('text_level')
            if level:
                section = text
            record = _base_record(doc_id, item, order) | {
                'subtype': 'heading' if level else kind,
                'section': section,
                'text': text,
            }
            buckets['text'].append(record)
            stats['out:text'] += 1

        elif kind == 'table':
            body = (item.get('table_body') or '').strip()
            caption = _join_captions(item, 'table_caption')
            if not body and not caption:
                stats['empty'] += 1
                continue
            record = _base_record(doc_id, item, order) | {
                'section': section,
                'caption': caption,
                'footnote': _join_captions(item, 'table_footnote'),
                'table_html': body,
                'img_path': item.get('img_path') or '',
                'context': _context_for(items, order),
            }
            buckets['tables'].append(record)
            stats['out:tables'] += 1

        elif kind == 'equation':
            latex, number = canonical_latex(item.get('text') or '')
            # Номер формулы или обозначение стандарта, оставшиеся от разметки
            # отдельным блоком: индексировать в них нечего.
            if not latex or is_technical_text_not_formula(latex):
                stats['empty'] += 1
                continue

            subtype = prose_subtype(latex)
            if subtype:
                text = clean_text(flatten_formula(latex))
                if not text:
                    stats['empty'] += 1
                    continue
                buckets['text'].append(_base_record(doc_id, item, order) | {
                    'subtype': subtype,
                    'section': section,
                    'text': text,
                })
                stats['out:text'] += 1
                stats[f'prose:{subtype}'] += 1
                continue

            record = _base_record(doc_id, item, order) | {
                'section': section,
                # Номер, которым на формулу ссылается текст: «по уравнению (5.14)».
                'number': number,
                'latex': latex,
                'context': _context_for(items, order),
            }
            pending_formulas.append(record)

        else:
            stats['skipped'] += 1

    for record, formula_class in zip(pending_formulas, classify(pending_formulas)):
        category = CLASS_TO_CATEGORY[formula_class]
        # Плоская запись — то, по чему формулу реально находят: запрос «H2SO4»
        # не совпадёт с «\mathrm{H} _ {2} \mathrm{SO} _ {4}». У химии вдобавок
        # выправляются кириллические двойники, которыми набраны русские формулы.
        flat = flatten_formula(record['latex'])
        record['flat'] = latinize_chemistry(flat) if formula_class == 'chemistry' else flat
        # Исходный класс остаётся в записи: физику свели с математикой в один
        # каталог, но по этому полю её можно отделить обратно без перезапуска.
        record['formula_class'] = formula_class
        buckets[category].append(record)
        stats[f'out:{category}'] += 1
        stats[f'class:{formula_class}'] += 1

    return buckets, stats


# --------------------------------------------------------------- классификация
def _rule_based(records: list[dict]) -> list[str | None]:
    """Разбор записи по символам элементов: где он проходит, там химия.

    Грамматика веществ надёжнее модели на однозначных случаях, поэтому её ответ
    идёт первым: реакцию вида ``As2O3 + O2 = As2O5 + 136,2 ккал`` модель по
    хвосту с килокалориями относит к физике, хотя вещества разбираются целиком.
    """
    return ['chemistry' if is_chemical(record['latex']) else None for record in records]


def make_classifier(model_path=None, use_model: bool = True, require_species: bool = True,
                    model=None):
    """Классификатор формул: правило поверх обученной модели.

    Порядок такой. Разбор формулы по символам элементов отвечает первым — он
    точен. Дальше решает модель, но её «химию» принимают, только если в записи
    нашлось вещество: без этого условия точность каталога химии 0.54, с ним
    0.90 при падении полноты с 0.97 до 0.77 (замер на размеченной выборке,
    ``pdfscan.prepare.split_audit``). Отклонённая химия уходит не куда попало, а
    в следующий по вероятности класс.

    Модель подгружается один раз и лениво — загрузка BERT занимает секунды, а на
    корпусе без формул она не нужна вовсе.
    """
    # Готовую модель принимают снаружи: так её подменяют в тестах и переиспользуют
    # между прогонами, не читая четыре мегабайта с диска заново.
    state = {'model': model, 'loaded': model is not None or not use_model}

    def classify(records: list[dict]) -> list[str]:
        if not records:
            return []
        verdicts = _rule_based(records)
        unresolved = [position for position, verdict in enumerate(verdicts) if verdict is None]
        if not unresolved:
            return verdicts

        if not state['loaded']:
            from pdfscan.formulas.classifier import FormulaClassifier
            state['model'] = FormulaClassifier.load(model_path or paths.FORMULA_CLASSIFIER)
            state['loaded'] = True
            if state['model'] is None:
                print(f'⚠️ нет модели {model_path or paths.FORMULA_CLASSIFIER}: '
                      'формулы без явных признаков химии уйдут в formulas_math')

        model = state['model']
        if model is None:
            for position in unresolved:
                verdicts[position] = 'math'
            return verdicts

        probabilities = model.predict_proba_batch(
            [records[position]['latex'] for position in unresolved],
            [records[position]['context'] for position in unresolved])

        for position, proba in zip(unresolved, probabilities):
            ranked = sorted(proba, key=proba.get, reverse=True)
            prediction = ranked[0]
            if (prediction == 'chemistry' and require_species
                    and not has_chemical_species(records[position]['latex'])):
                prediction = next(name for name in ranked[1:] if name != 'chemistry')
            verdicts[position] = prediction
        return verdicts

    return classify


# ----------------------------------------------------------------------- вывод
def _markdown_lines(category: str, doc_id: str, records: list[dict]):
    """Человекочитаемый срез категории: метаданные во фронтматтере, блоки — по порядку."""
    yield '---'
    yield f'doc_id: {json.dumps(doc_id, ensure_ascii=False)}'
    yield f'category: {category}'
    yield f'items: {len(records)}'
    yield '---'
    yield ''

    section = object()
    for record in records:
        # Заголовок сам задаёт раздел, поэтому баннер над ним не нужен — иначе
        # одна и та же строка печаталась бы дважды подряд.
        is_heading = category == 'text' and record['subtype'] == 'heading'
        if record.get('section') != section:
            section = record.get('section')
            if section and not is_heading:
                yield f'## {section}'
                yield ''

        page = record.get('page')
        anchor = f'<!-- page {page} -->' if page is not None else ''

        if category == 'text':
            if is_heading:
                yield f"## {record['text']} {anchor}".rstrip()
            else:
                prefix = '> ' if record['subtype'] in ('ref_text', 'page_footnote') else ''
                yield f"{prefix}{record['text']} {anchor}".rstrip()
        elif category == 'tables':
            if record['caption']:
                yield f"**{record['caption']}** {anchor}".rstrip()
            elif anchor:
                yield anchor
            if record['table_html']:
                yield record['table_html']
            if record['footnote']:
                yield f"_{record['footnote']}_"
        else:
            number = f" ({record['number']})" if record['number'] else ''
            if anchor or number:
                yield f'{anchor}{number}'.strip()
            yield f"$$\n{record['latex']}\n$$"
        yield ''


def _write_document(out_dir: Path, doc_id: str, buckets: dict, handles: dict, write_md: bool):
    for category, records in buckets.items():
        if not records:
            continue
        category_dir = out_dir / category
        with (category_dir / f'{doc_id}.jsonl').open('w', encoding='utf-8') as handle:
            for record in records:
                line = json.dumps(record, ensure_ascii=False)
                handle.write(line + '\n')
                handles[category].write(line + '\n')
        if write_md:
            (category_dir / f'{doc_id}.md').write_text(
                '\n'.join(_markdown_lines(category, doc_id, records)), encoding='utf-8')


def run(source=None, out_dir=None, *, limit=None, write_md=True, use_model=True,
        model_path=None, require_species=True, progress=True) -> dict:
    """Раскладывает корпус MinerU по каталогам категорий."""
    source = Path(source or paths.ROOT / 'parsed_literature')
    out_dir = Path(out_dir or paths.ROOT / 'corpus_split')
    if not source.exists():
        raise SystemExit(f'нет каталога с разбором: {source}')

    doc_dirs = _document_dirs(source)
    if not doc_dirs:
        raise SystemExit(f'в {source} нет */hybrid_auto/*_content_list.json')
    if limit:
        doc_dirs = doc_dirs[:limit]
    if progress:
        print(f'{len(doc_dirs)} документов в {source}')

    for category in CATEGORIES:
        (out_dir / category).mkdir(parents=True, exist_ok=True)
    combined_dir = out_dir / 'all'
    combined_dir.mkdir(parents=True, exist_ok=True)

    classify = make_classifier(model_path, use_model, require_species)
    stats = Counter()
    per_document = []
    handles = {}

    try:
        for category in CATEGORIES:
            handles[category] = (combined_dir / f'{category}.jsonl').open('w', encoding='utf-8')

        for doc_id, buckets, doc_stats in _iter_documents(doc_dirs, classify, progress):
            _write_document(out_dir, doc_id, buckets, handles, write_md)
            stats.update(doc_stats)
            stats['documents'] += 1
            per_document.append({
                'doc_id': doc_id,
                'topic': _topic(doc_id),
                **{category: len(records) for category, records in buckets.items()},
            })
    finally:
        for handle in handles.values():
            handle.close()

    manifest = {
        'source': str(source),
        'categories': {
            category: {
                'dir': category,
                'combined': f'all/{category}.jsonl',
                'items': stats[f'out:{category}'],
                'documents': sum(1 for row in per_document if row[category]),
            }
            for category in CATEGORIES
        },
        'documents': stats['documents'],
        'skipped_blocks': stats['skipped'],
        'empty_blocks': stats['empty'],
        'input_types': {key.removeprefix('in:'): value for key, value in sorted(stats.items())
                        if key.startswith('in:')},
        'formula_classes': {key.removeprefix('class:'): value
                            for key, value in sorted(stats.items())
                            if key.startswith('class:')},
        'equations_as_prose': {key.removeprefix('prose:'): value
                               for key, value in sorted(stats.items())
                               if key.startswith('prose:')},
        'per_document': per_document,
    }
    (out_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    if progress:
        print()
        for category in CATEGORIES:
            print(f'  {category:15s} {stats[f"out:{category}"]:7d}')
        print(f'  {"пропущено":15s} {stats["skipped"]:7d}')
        print(f'\nготово → {out_dir}')
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Разложить разбор MinerU по каталогам: текст, формулы, таблицы')
    parser.add_argument('source', nargs='?', default=None,
                        help='каталог с разбором (по умолчанию parsed_literature)')
    parser.add_argument('--out', default=None,
                        help='куда писать (по умолчанию corpus_split)')
    parser.add_argument('--limit', type=int, default=None,
                        help='взять только первые N документов')
    parser.add_argument('--no-markdown', action='store_true',
                        help='только JSONL, без человекочитаемых срезов')
    parser.add_argument('--no-model', action='store_true',
                        help='без классификатора: химию определяет только разбор формулы')
    parser.add_argument('--model', default=None, help='путь к formula_classifier.pkl')
    parser.add_argument('--trust-model', action='store_true',
                        help='принимать химию от модели без разбора веществ '
                             '(точность каталога химии падает с 0.90 до 0.54)')
    args = parser.parse_args(argv)

    run(args.source, args.out,
        limit=args.limit,
        write_md=not args.no_markdown,
        use_model=not args.no_model,
        model_path=args.model,
        require_species=not args.trust_model)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
