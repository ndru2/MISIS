"""Решение о замене формулы распознанным вариантом.

Проверяется главное свойство этапа: отказ дешевле ошибки. Текстовый слой не
подменяется, срывы модели не проходят, а спорные случаи не применяются молча.
"""

import json

from pdfscan.formulas.ocr import apply, config, decide, select


def block(text, source='ocr'):
    return {'text': text, 'text_source': source, 'type': 'Formula:math'}


def test_text_layer_wins_over_recognition():
    """Символьный слой PDF точнее любой модели — его не трогаем."""
    verdict = decide.decide(block('Fe_2O_3 + 3CO = 2Fe + 3CO_2', 'text_layer'),
                            r'\frac{a}{b} = c')
    assert verdict['decision'] == 'keep'


def test_bare_number_is_replaced_and_number_kept():
    """Формула-картинка поверх текстового слоя: остался только номер."""
    verdict = decide.decide(block('(3.2)', 'text_layer'),
                            r'\Delta G = \Delta H - T\Delta S')
    assert verdict['decision'] == 'replace'
    assert '(3.2)' in verdict['candidate']
    assert r'\Delta G' in verdict['candidate']


def test_cid_artifacts_are_replaced():
    verdict = decide.decide(block('(cid:11)(cid:3) = (cid:87)'),
                            r'\sigma_{т} = P/F')
    assert verdict['decision'] == 'replace'


def test_flat_ocr_text_loses_to_structured_latex():
    """Распознавание страницы теряет дробь, модель формул её видит."""
    verdict = decide.decide(block('lg K 298 5 2 T'),
                            r'\lg K = -\frac{\Delta G^{\circ}}{2,3RT}')
    assert verdict['decision'] == 'replace'


def test_runaway_candidate_rejected():
    """Срыв модели в повтор одного множителя — не замена."""
    verdict = decide.decide(block('x + y = z'), 'x + x + x + x + x + x + x + x')
    assert verdict['decision'] == 'keep'
    assert 'повтор' in verdict['reason']


def test_alien_characters_rejected():
    """Иероглифы из обучающих данных модели и тег картинки — признак выдумки."""
    hieroglyph = decide.decide(block('Cu, g/L Ni, g/L'), r'\mathrm{表}\;3.236')
    assert hieroglyph['decision'] == 'keep'
    assert 'посторонние знаки' in hieroglyph['reason']

    gave_up = decide.decide(block('CaO = Liquidus 1340'),
                            r'\text{<img>image.png</img>}')
    assert gave_up['decision'] == 'keep'

    # Нормальная разметка через проверку проходит.
    assert decide.validate(r'\Delta G^{\circ} = -RT\ln K')[0]


def test_column_of_numbers_is_not_a_formula():
    """Подписи осей графика модель складывает в столбик дробей."""
    chart = decide.decide(
        block('22 20 S 18+ Concentrates SX-EW 1900 1910 1920 1930'),
        r'\begin{array}{l}\frac{22}{10}\\\frac{22}{18}\\\frac{11}{10}\end{array}')
    assert chart['decision'] == 'keep'
    assert 'ни обозначений' in chart['reason']

    # Обозначение «array» из разметки макета обозначением не считается,
    # а настоящая переменная или знак отношения — считается.
    assert not decide.asserts_something(r'\begin{array}{l}\frac{1}{2}\end{array}')
    assert decide.asserts_something(r'\frac{a}{b}')
    assert decide.asserts_something(r'2 + 2 = 4')
    assert decide.asserts_something(r'\mathrm{CaO}\rightarrow\mathrm{Ca}')


def test_unbalanced_braces_rejected():
    verdict = decide.decide(block('a = b'), r'\frac{a}{b')
    assert verdict['decision'] == 'keep'
    assert 'скобки' in verdict['reason']


def test_reaction_arrow_is_not_an_unpaired_bracket():
    """«\\right» находится внутри «\\rightarrow» — реакции нельзя терять."""
    arrow = (r'\mathrm{Cu_{2}S(s)+O_{2}(g)}\rightarrow'
             r'\mathrm{2Cu^{\circ}(l)+SO_{2}(g)}')
    assert decide.validate(arrow)[0]
    assert decide.validate(r'A\xrightarrow{t}B')[0]
    assert decide.validate(r'A\longrightarrow B')[0]

    # Настоящая непарность по-прежнему ловится.
    assert not decide.validate(r'\left( \frac{a}{b}')[0]
    assert decide.validate(r'\left(\frac{a}{b}\right)')[0]


def test_empty_candidate_rejected():
    assert decide.decide(block('a = b'), None)['decision'] == 'keep'
    assert decide.decide(block('a = b'), '   ')['decision'] == 'keep'


def test_equal_structure_stays_unsure_and_is_not_applied():
    """Спорный случай по умолчанию не пишется в файл."""
    verdict = decide.decide(block('C + O_2 = CO_2'), r'C + O_{2} \to CO_{2}')
    assert verdict['decision'] == 'unsure'
    assert not decide.applies(verdict, config.DEFAULT)

    permissive = config.FormulaOcrConfig(accept_unsure=True)
    assert decide.applies(verdict, permissive)


def test_more_structure_still_wins():
    """Настоящая находка модели проходит: плоской строки была дробь."""
    verdict = decide.decide(block('a = b'), r'\frac{a}{b} = c')
    assert verdict['decision'] == 'replace'


def test_layout_wrappers_are_not_structure():
    """Обёртку выравнивания модель ставит всегда, даже вокруг пустого ответа."""
    assert decide.structure_score(r'\begin{aligned}&[1]\\ &[2]\end{aligned}') == 0
    assert decide.structure_score(r'\begin{array}{r l}{{a}}&{{b}}\end{array}') == 0
    assert decide.structure_score(r'\mathrm{C u(s)}') == 0
    # Окружения, которые действительно говорят о структуре, остаются признаком.
    assert decide.structure_score(r'\begin{cases}x=1\\ y=2\end{cases}') > 0


def test_reformatted_same_text_is_not_a_replacement():
    """Тот же текст в разметке — не улучшение, а потеря исходника."""
    verdict = decide.decide(block('Cu(s)'), r'\mathrm{C u(s)}')
    assert verdict['decision'] == 'keep'
    assert 'оформлением' in verdict['reason']

    wrapped = decide.decide(block('[3]'), r'\begin{aligned}&[3]\end{aligned}')
    assert wrapped['decision'] == 'keep'


def test_number_not_duplicated_when_model_already_found_it():
    verdict = decide.decide(block('(cid:5) (4.7)'), r'\eta = \frac{Q_1}{Q_2} (4.7)')
    assert verdict['candidate'].count('4.7') == 1


def test_selection_recognises_bare_numbers():
    assert select.is_bare_number('(3.2)')
    assert select.is_bare_number(' [14] ')
    assert not select.is_bare_number('E = mc^2')
    assert select.trailing_number('формула (12.3)') == '(12.3)'


# Рамка обычной выключной формулы: её должно хватать на любой порог.
DISPLAY_BOX = {'x0': 90, 'top': 300, 'x1': 320, 'bottom': 322}


def test_selection_skips_reliable_and_already_processed():
    cfg = config.DEFAULT
    good = {'type': 'Formula:math', 'text': 'a = b', 'text_source': 'text_layer',
            'bbox': dict(DISPLAY_BOX)}
    assert select.reason(good, cfg) is None

    scanned = {**good, 'text_source': 'ocr'}
    assert select.reason(scanned, cfg) == 'текст от распознавания страницы'

    done = {**scanned, 'formula_ocr': {'decision': 'keep'}}
    assert select.reason(done, cfg) is None

    no_box = {'type': 'Formula:math', 'text': 'a', 'text_source': 'ocr'}
    assert select.reason(no_box, cfg) is None


def test_citation_markers_are_not_sent_to_recognition():
    """Разметка помечает ссылку на литературу как формулу — это не формула."""
    cfg = config.DEFAULT
    for text in ('[1]', '[1] [2]', '[3, 4]', '[5-7]'):
        marker = {'type': 'Formula:math', 'text': text, 'text_source': 'ocr',
                  'bbox': dict(DISPLAY_BOX)}
        assert select.reason(marker, cfg) is None, text


def test_region_too_small_to_hold_a_formula_is_skipped():
    """Распознаётся картинка: в рамку маркера списка формула не поместится."""
    cfg = config.DEFAULT
    bullet = {'type': 'Formula:math', 'text': '*', 'text_source': 'ocr',
              'bbox': {'x0': 100, 'top': 200, 'x1': 105, 'bottom': 208}}
    assert select.reason(bullet, cfg) is None

    fragment = {**bullet, 'text': '+ –',
                'bbox': {'x0': 100, 'top': 200, 'x1': 120, 'bottom': 209}}
    assert select.reason(fragment, cfg) is None

    # Номер формулы при широкой рамке — наоборот, самый нужный случай: сама
    # формула лежит там картинкой.
    numbered = {**bullet, 'text': '(3.2)', 'bbox': dict(DISPLAY_BOX)}
    assert select.reason(numbered, cfg) == 'от формулы остался только номер'


def test_tables_and_diagrams_are_too_tall_for_a_formula():
    """Формула тянется в ширину; высокая рамка — таблица или диаграмма."""
    cfg = config.DEFAULT
    diagram = {'type': 'Formula:math', 'text': 'Pb-S-O, 1200 C PbS(l) PbO(l)',
               'text_source': 'ocr',
               'bbox': {'x0': 90, 'top': 200, 'x1': 424, 'bottom': 457}}
    assert select.reason(diagram, cfg) is None

    # Многострочная система реакций такой высоты не набирает и остаётся.
    reactions = {**diagram, 'text': 'CaO = Ca^2+ + O^2-',
                 'bbox': {'x0': 90, 'top': 200, 'x1': 462, 'bottom': 287}}
    assert select.reason(reactions, cfg) == 'текст от распознавания страницы'


def test_results_outside_current_selection_are_dropped(tmp_path):
    """Отбор ужесточили — старые ответы по этим блокам в отчёт не идут."""
    manifest = tmp_path / 'manifest.jsonl'
    with open(manifest, 'w', encoding='utf-8') as handle:
        for block_id in ('doc#12', 'doc#13'):
            handle.write(json.dumps({'blocks': 'out/doc/blocks.jsonl',
                                     'block_id': block_id}) + '\n')

    scope = apply.manifest_scope(manifest)
    kept = {'blocks': 'out/doc/blocks.jsonl', 'block_id': 'doc#12'}
    gone = {'blocks': 'out/doc/blocks.jsonl', 'block_id': 'doc#99'}
    other = {'blocks': 'out/second/blocks.jsonl', 'block_id': 'doc#12'}

    assert apply.in_scope(kept, scope)
    assert not apply.in_scope(gone, scope)
    assert not apply.in_scope(other, scope)


def test_missing_manifest_disables_the_check(tmp_path):
    """Пропавший манифест не должен молча выбрасывать всю работу."""
    scope = apply.manifest_scope(tmp_path / 'нет-такого.jsonl')
    assert scope == {}
    assert apply.in_scope({'blocks': 'любой', 'block_id': 'любой'}, scope)
