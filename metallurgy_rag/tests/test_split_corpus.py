"""Раскладка разбора MinerU по категориям: маршрутизация блоков и проза в формулах."""

import json

from pdfscan.prepare import split_corpus


def _classify_all(label):
    return lambda records: [label] * len(records)


def _split(items, classify=None):
    return split_corpus._split_document('doc', items, classify or _classify_all('chemistry'))


def test_furniture_and_pictures_do_not_reach_any_category():
    items = [
        {'type': 'header', 'text': 'ЖУРНАЛ', 'page_idx': 1},
        {'type': 'footer', 'text': 'стр.', 'page_idx': 1},
        {'type': 'page_number', 'text': '7', 'page_idx': 1},
        {'type': 'image', 'img_path': 'images/a.jpg', 'page_idx': 1},
        {'type': 'chart', 'img_path': 'images/b.jpg', 'page_idx': 1},
    ]
    buckets, stats = _split(items)

    assert all(not records for records in buckets.values())
    assert stats['skipped'] == len(items)


def test_heading_becomes_section_for_the_blocks_below_it():
    items = [
        {'type': 'text', 'text': 'Плавка', 'text_level': 1, 'page_idx': 0},
        {'type': 'text', 'text': 'Шлак сливают через лётку.', 'page_idx': 0},
        {'type': 'table', 'table_body': '<table><tr><td>Cu</td></tr></table>',
         'table_caption': ['Таблица 1'], 'page_idx': 0},
    ]
    buckets, _ = _split(items)

    assert [record['subtype'] for record in buckets['text']] == ['heading', 'text']
    assert buckets['text'][1]['section'] == 'Плавка'
    assert buckets['tables'][0]['section'] == 'Плавка'
    assert buckets['tables'][0]['caption'] == 'Таблица 1'


def test_chemical_reaction_wins_over_the_model():
    """Разбор по символам элементов отвечает раньше модели.

    Хвост с килокалориями уводит модель в физику, хотя вещества по обе стороны
    знака реакции раскладываются целиком.
    """
    reaction, _ = split_corpus.canonical_latex(
        r'$$\mathrm{As} _ {2} \mathrm{O} _ {3} + \mathrm{O} _ {2} = '
        r'\mathrm{As} _ {2} \mathrm{O} _ {5} + 1 3 6, 2 \text {ккал}$$')
    assert split_corpus.is_chemical(reaction)

    # Модель не загружается, и всё же реакция уходит в химию, а не в математику.
    classify = split_corpus.make_classifier(use_model=False)
    assert classify([{'latex': reaction, 'context': ''}]) == ['chemistry']


def test_spacing_from_mineru_does_not_hide_a_reaction():
    # MinerU печатает «\mathrm{O} _ {2}»; с пробелом вещество рвётся пополам.
    spaced, _ = split_corpus.canonical_latex(
        '$$\n\\mathrm{Cu} _ {2} \\mathrm{O} + \\mathrm{C} = 2 \\mathrm{Cu} + \\mathrm{CO}\n$$')
    assert split_corpus.is_chemical(spaced)


class _StubModel:
    """Модель, всегда уверенно отвечающая «химия»."""

    def predict_proba_batch(self, formulas, contexts, layouts=None):
        return [{'chemistry': 0.99, 'physics': 0.008, 'math': 0.002} for _ in formulas]


def _classify_with_stub(latex, **kwargs):
    classify = split_corpus.make_classifier(model=_StubModel(), **kwargs)
    return classify([{'latex': latex, 'context': ''}])[0]


def test_model_chemistry_without_any_species_is_rejected():
    """Без вещества в записи реакции нет, что бы ни говорила модель."""
    # Одинокая «H» совпадает с водородом, но вещества не образует.
    assert _classify_with_stub(r'R = H = 2 \cdot 13,2 / 3,14 = 8,4') == 'physics'
    assert _classify_with_stub(r'T_{\Pi} = T_{\mathrm{C}} - A v^{m}') == 'physics'
    # А запись с настоящим веществом модель проводит в химию.
    assert _classify_with_stub(r'\mathrm{Fe}_{3} \mathrm{O}_{4}') == 'chemistry'


def test_species_filter_can_be_switched_off():
    assert _classify_with_stub(r'T_{\Pi} = T_{\mathrm{C}} - A v^{m}',
                               require_species=False) == 'chemistry'


def test_has_chemical_species_is_softer_than_the_reaction_rule():
    single = r'\mathrm{Fe}_{2} \mathrm{O}_{3}'
    assert split_corpus.has_chemical_species(single)
    # Одного вещества мало для уравнения: знака реакции и второй части нет.
    assert not split_corpus.is_chemical(single)
    assert not split_corpus.has_chemical_species(r'\frac{dy}{dx} = 2 t + 1')


def test_physical_formula_is_not_read_as_a_reaction():
    for latex in (r'P = I V', r'F r = \frac{u^{2}}{gL}', r'\ln k = \ln A - E_{a} / R T'):
        assert not split_corpus.is_chemical(latex)


def test_physics_and_math_share_one_directory_but_keep_their_class():
    items = [{'type': 'equation', 'text': r'$$\ln k = \ln A - E _ {a} / R T$$', 'page_idx': 2}]
    buckets, _ = _split(items, classify=_classify_all('physics'))

    assert len(buckets['formulas_math']) == 1
    assert buckets['formulas_math'][0]['formula_class'] == 'physics'


def test_flat_field_matches_plain_query_spelling_and_number_is_kept_apart():
    items = [{'type': 'equation', 'page_idx': 1,
              'text': '$$\n\\mathrm{H} _ {2} \\mathrm{SO} _ {4}\\tag{5.14}\n$$'}]
    buckets, _ = _split(items)
    record = buckets['formulas_chem'][0]

    assert 'H2SO4' in record['flat'].replace(' ', '')
    assert '\\tag' not in record['flat'] and '\\tag' not in record['latex']
    assert record['number'] == '5.14'


def test_prose_tagged_as_equation_lands_in_text():
    items = [{'type': 'equation', 'text': r'$$\text {Приход тепла}$$', 'page_idx': 4}]
    buckets, _ = _split(items)

    assert not buckets['formulas_chem'] and not buckets['formulas_math']
    assert buckets['text'][0]['text'] == 'Приход тепла'
    assert buckets['text'][0]['subtype'] == 'caption'


def _subtype(raw):
    return split_corpus.prose_subtype(split_corpus.canonical_latex(raw)[0])


def test_symbol_legend_lands_in_text_but_word_formula_stays_a_formula():
    assert _subtype(r'$$\eta_ {s l a g} = \text {viscosity of slag, Pa s}$$') == 'nomenclature'

    # У словесной формулы есть и знак отношения, и оператор — она формула.
    assert _subtype(
        r'$$\text {Aspect Ratio} = \frac {\text {Major Axis}}{\text {Minor Axis}}$$') is None
    assert _subtype(r'$$\mathrm{CO} _ {2} + \mathrm{C} = 2 \mathrm{CO}$$') is None


def test_formula_number_alone_is_dropped():
    items = [{'type': 'equation', 'text': '$$(3.4)$$', 'page_idx': 1}]
    buckets, stats = _split(items)

    assert all(not records for records in buckets.values())
    assert stats['empty'] == 1


def test_markdown_prints_a_heading_once():
    items = [
        {'type': 'text', 'text': 'Плавка', 'text_level': 1, 'page_idx': 0},
        {'type': 'text', 'text': 'Абзац.', 'page_idx': 0},
    ]
    buckets, _ = _split(items)
    body = '\n'.join(split_corpus._markdown_lines('text', 'doc', buckets['text']))

    assert body.count('Плавка') == 1


def test_run_writes_categories_and_manifest(tmp_path):
    doc_dir = tmp_path / 'src' / 'Медь__Отчёт' / 'hybrid_auto'
    doc_dir.mkdir(parents=True)
    (doc_dir / 'Медь__Отчёт_content_list.json').write_text(json.dumps([
        {'type': 'text', 'text': 'Введение', 'text_level': 1, 'page_idx': 0},
        {'type': 'text', 'text': 'Медь извлекают из шлака.', 'page_idx': 0},
        {'type': 'equation', 'text': r'$$\mathrm{Cu} _ {2} \mathrm{O} + \mathrm{C} = '
                                     r'2 \mathrm{Cu} + \mathrm{CO}$$', 'page_idx': 1},
        {'type': 'table', 'table_body': '<table><tr><td>Cu</td></tr></table>',
         'table_caption': ['Таблица 1'], 'page_idx': 1},
        {'type': 'page_number', 'text': '2', 'page_idx': 1},
    ]), encoding='utf-8')

    out = tmp_path / 'out'
    manifest = split_corpus.run(tmp_path / 'src', out, use_model=False, progress=False)

    assert manifest['documents'] == 1
    assert manifest['categories']['formulas_chem']['items'] == 1
    assert manifest['categories']['tables']['items'] == 1
    assert manifest['categories']['text']['items'] == 2

    records = [json.loads(line)
               for line in (out / 'all' / 'formulas_chem.jsonl').read_text().splitlines()]
    assert records[0]['topic'] == 'Медь'
    assert records[0]['doc_id'] == 'Медь__Отчёт'
    assert (out / 'text' / 'Медь__Отчёт.md').exists()
    assert (out / 'manifest.json').exists()


def test_run_without_model_keeps_unrecognized_formulas_in_math(tmp_path):
    doc_dir = tmp_path / 'src' / 'Док' / 'hybrid_auto'
    doc_dir.mkdir(parents=True)
    (doc_dir / 'Док_content_list.json').write_text(json.dumps([
        {'type': 'equation', 'text': r'$$\ln k = \ln A - E _ {a} / R T$$', 'page_idx': 0},
    ]), encoding='utf-8')

    manifest = split_corpus.run(tmp_path / 'src', tmp_path / 'out',
                                use_model=False, write_md=False, progress=False)

    assert manifest['categories']['formulas_math']['items'] == 1
    assert manifest['categories']['formulas_chem']['items'] == 0
