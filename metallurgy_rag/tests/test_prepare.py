"""Стадия очистки: хранение по документам, шум, обрамление, склейка."""

import pyarrow as pa
import pyarrow.parquet as pq

from pdfscan.prepare import (boilerplate, config, dedup, garbage, merge, store,
                             textstats)


def make_row(order, text, **extra):
    row = {
        'doc_id': 'док', 'source': 'pdf/док.pdf', 'block_id': f'док#{order}',
        'order': order, 'page': 1, 'type': 'NarrativeText', 'text': text,
        'text_source': 'text_layer', 'reliable': True,
        'bbox_x0': 70.0, 'bbox_top': 100.0 + order * 20,
        'bbox_x1': 500.0, 'bbox_bottom': 114.0 + order * 20,
        'page_width': 595.0, 'page_height': 842.0, 'languages': 'rus',
        'math_font_ratio': 0.0, 'italic_ratio': 0.0, 'n_subscript': 0,
        'n_superscript': 0, 'n_fonts': 1, 'table_html': None,
        'prev_id': None, 'next_id': None,
    }
    row.update(extra)
    return row


def prepared(rows):
    """Заводит рабочие поля так, как это делает стадия очистки."""
    for row in rows:
        row['text_out'] = row['text']
        row['keep'] = True
        row['drop_reason'] = ''
        row['merged_into'] = None
        row['merged_count'] = 0
        row['merged_ids'] = []
        row['boilerplate_key'] = None
        row['page_min'] = row['page']
        row['page_max'] = row['page']
        row.update(textstats.signals(row['text_out']))
    return rows


# --------------------------------------------------------------- хранение
def test_documents_are_read_one_at_a_time(tmp_path):
    """Потоковое чтение отдаёт документ целиком и по одному.

    На этом держится вся стадия: без него корпус снова оказался бы в памяти
    словарями и ушёл в своп.
    """
    path = tmp_path / 'blocks.parquet'
    with store.DocumentWriter(path, store.RAW_SCHEMA) as writer:
        writer.write([make_row(1, 'первый документ')])
        writer.write([make_row(2, 'второй', doc_id='второй'),
                      make_row(3, 'документ', doc_id='второй')])

    assert store.count_documents(path) == 2
    documents = list(store.iter_documents(path))
    assert [len(rows) for rows in documents] == [1, 2]
    assert documents[1][0]['text'] == 'второй'


def test_documents_split_by_doc_id_not_by_row_group(tmp_path):
    """Раскладка файла на группы строк не должна влиять на границы документов.

    Таблицу мог записать другой инструмент или прежняя версия модуля, сложив в
    одну группу несколько документов. Если бы границы брались от групп, шаги
    очистки увидели бы половину документа и склеили бы чужие абзацы.
    """
    path = tmp_path / 'blocks.parquet'
    rows = [make_row(1, 'первый', doc_id='а'),
            make_row(2, 'второй', doc_id='б'),
            make_row(3, 'третий', doc_id='б')]
    pq.write_table(pa.Table.from_pylist(rows, schema=store.RAW_SCHEMA), path)

    documents = list(store.iter_documents(path))
    assert [[row['doc_id'] for row in doc] for doc in documents] == [['а'], ['б', 'б']]


def test_iter_documents_adds_doc_id_to_projection(tmp_path):
    """Группировка невозможна без doc_id, поэтому он добавляется сам."""
    path = tmp_path / 'blocks.parquet'
    with store.DocumentWriter(path, store.RAW_SCHEMA) as writer:
        writer.write([make_row(1, 'текст')])

    rows = next(store.iter_documents(path, columns=['text', 'type']))
    assert set(rows[0]) == {'text', 'type', 'doc_id'}


def test_flatten_unpacks_nested_fields():
    flat = store._flatten({
        'doc_id': 'док', 'block_id': 'док#1', 'page': 2, 'type': 'Formula:math',
        'text': 'a = b', 'bbox': {'x0': 1, 'top': 2, 'x1': 3, 'bottom': 4},
        'page_size': {'width': 595, 'height': 842},
        'layout': {'math_font_ratio': 0.5, 'n_fonts': 2},
        'languages': ['rus', 'eng'],
    })
    assert flat['bbox_x0'] == 1 and flat['bbox_bottom'] == 4
    assert flat['page_width'] == 595
    assert flat['languages'] == 'rus,eng'
    assert flat['math_font_ratio'] == 0.5
    assert flat['reliable'] is False


# ------------------------------------------------------------------- шум
class FakeModel:
    """Символьная модель с заданным отклонением: тест не должен её учить."""

    def __init__(self, deviation=0.0):
        self._deviation = deviation

    def deviation(self, text, script):
        return self._deviation


def test_garbage_catches_doubled_letters():
    row = prepared([make_row(1, 'ммееддннааяя ррууддаа ппллааввккаа ммееддии')])[0]
    row['deviation'] = 6.0
    row['reliable'] = False
    score, reason = garbage.score(row, config.DEFAULT)
    assert score >= config.DEFAULT.garbage_threshold
    assert garbage.is_garbage(row | {'garbage_score': score}, config.DEFAULT)


def test_clean_prose_survives():
    text = ('Восстановление оксидов железа в доменной печи идёт ступенчато, '
            'и основным восстановителем служит оксид углерода.')
    row = prepared([make_row(1, text)])[0]
    row['deviation'] = 0.4
    score, _ = garbage.score(row, config.DEFAULT)
    row['garbage_score'] = score
    assert not garbage.is_garbage(row, config.DEFAULT)


def test_formula_is_not_judged_by_prose_signals():
    """У формулы законно мало гласных и много знаков — это не повод её выбросить."""
    row = prepared([make_row(1, 'Fe_2O_3 + 3CO = 2Fe + 3CO_2',
                             type='Formula:chemistry')])[0]
    row['deviation'] = 3.0
    assert garbage.enabled_signals('Formula:chemistry', row['n_chars']) == frozenset()
    score, _ = garbage.score(row, config.DEFAULT)
    row['garbage_score'] = score
    assert not garbage.is_garbage(row, config.DEFAULT)


# ---------------------------------------------------------- колонтитулы
def test_boilerplate_needs_several_pages():
    rows = prepared([
        make_row(order, 'Труды симпозиума, с. %d' % page, page=page,
                 type='Header')
        for order, page in enumerate((1, 2, 3, 4, 5), start=1)
    ])
    flags, summary = boilerplate.detect(rows, config.DEFAULT)
    assert len(flags) == 5
    assert summary[0]['pages'] == 5


def test_section_heading_repeated_twice_is_not_boilerplate():
    rows = prepared([make_row(1, 'Введение', page=1, type='Title'),
                     make_row(2, 'Введение', page=2, type='Title')])
    flags, _ = boilerplate.detect(rows, config.DEFAULT)
    assert flags == {}


# ---------------------------------------------------------------- склейка
def test_hyphenated_word_is_stitched_across_pages():
    first = make_row(1, 'Медный концентрат поступает на обжиг, где проис-',
                     page=1, bbox_top=700.0, bbox_bottom=714.0)
    second = make_row(2, 'ходит удаление серы.', page=2,
                      bbox_top=100.0, bbox_bottom=114.0)
    rows = prepared([first, second])
    merge.merge_document(rows, config.DEFAULT)

    assert 'происходит' in rows[0]['text_out']
    assert rows[1]['keep'] is False
    assert rows[0]['page_max'] == 2


def test_fragment_continuing_a_paragraph_is_merged():
    """Склеиваются обрывки, а не готовые абзацы."""
    first = make_row(1, 'Скорость восстановления растёт с температурой')
    second = make_row(2, 'и достигает предела', type='UncategorizedText')
    rows = prepared([first, second])
    merge.merge_document(rows, config.DEFAULT)

    assert rows[1]['keep'] is False
    assert 'предела' in rows[0]['text_out']
    assert rows[0]['merged_count'] == 1


def test_two_full_paragraphs_are_left_alone():
    """Два самостоятельных абзаца не обрывки, и склеивать их нельзя.

    Осторожность здесь важнее полноты: ошибочная склейка ломает порядок чтения,
    и в чанк попадает текст из другого места страницы.
    """
    first = make_row(1, 'Скорость восстановления растёт с температурой')
    second = make_row(2, 'и достигает максимума около 1000 градусов Цельсия')
    rows = prepared([first, second])
    merge.merge_document(rows, config.DEFAULT)
    assert rows[1]['keep'] is True


def test_finished_sentences_are_not_merged():
    first = make_row(1, 'Обжиг ведут при 900 °C.')
    second = make_row(2, 'Затем подают', type='UncategorizedText')
    rows = prepared([first, second])
    merge.merge_document(rows, config.DEFAULT)
    assert rows[1]['keep'] is True


# ------------------------------------------------------------- повторы
def test_exact_duplicates_keep_the_first():
    long_text = 'Плавка медного концентрата в печи Ванюкова. ' * 4
    rows = prepared([make_row(1, long_text), make_row(2, long_text)])
    dropped = dedup.exact_duplicates(rows, config.DEFAULT)
    assert dropped == 1
    assert rows[0]['keep'] and not rows[1]['keep']
    assert rows[1]['drop_reason'] == 'duplicate'


def test_short_repeats_are_left_to_boilerplate():
    rows = prepared([make_row(1, 'Таблица 1'), make_row(2, 'Таблица 1')])
    assert dedup.exact_duplicates(rows, config.DEFAULT) == 0


def article(seed, sentences=200):
    """Текст статьи: слов должно быть много, иначе прореживание дробей съест всё."""
    return ' '.join(
        f'извлечение кобальта из шлака составило {seed}{number} процента '
        f'при расходе извести {number} килограмма на тонну.'
        for number in range(sentences))


def test_similar_documents_are_reported_not_dropped():
    """Статья из сборника и та же статья отдельным файлом должны найтись."""
    coefficients = dedup.make_coefficients(config.DEFAULT)
    shared = article('а')

    left, left_size = dedup.document_signature([shared], coefficients)
    right, right_size = dedup.document_signature([shared], coefficients)
    other, other_size = dedup.document_signature([article('б')], coefficients)

    assert left_size > 0 and other_size > 0

    pairs = dedup.similar_pairs(
        ['сборник', 'статья', 'другое'], [left, right, other],
        {'сборник': left_size, 'статья': right_size, 'другое': other_size},
        config.DEFAULT)

    assert len(pairs) == 1
    assert {pairs[0]['left'], pairs[0]['right']} == {'сборник', 'статья'}
    assert pairs[0]['similarity'] > 0.9


def test_short_documents_are_excluded_from_comparison():
    """У документа из пары строк дробей не остаётся, и сравнивать его нечем."""
    coefficients = dedup.make_coefficients(config.DEFAULT)
    signature, size = dedup.document_signature(['Оглавление'], coefficients)
    assert size == 0
    assert dedup.similar_pairs(['мелкий'], [signature], {'мелкий': size}) == []


def pair(left, right, similarity):
    return {'left': left, 'right': right, 'similarity': similarity}


def test_copy_closest_to_the_root_survives():
    copies = dedup.duplicate_documents(
        [pair('Медь/Шлаки/отчёт', 'отчёт', 1.0)],
        {'Медь/Шлаки/отчёт': 50, 'отчёт': 50})
    assert copies == {'Медь/Шлаки/отчёт': 'отчёт'}


def test_three_copies_collapse_to_one():
    """Попарного решения мало: из трёх копий иначе выживут две."""
    copies = dedup.duplicate_documents(
        [pair('а/отчёт', 'б/в/отчёт', 1.0),
         pair('б/в/отчёт', 'г/д/отчёт', 1.0)],
        {'а/отчёт': 30, 'б/в/отчёт': 30, 'г/д/отчёт': 30})
    assert copies == {'б/в/отчёт': 'а/отчёт', 'г/д/отчёт': 'а/отчёт'}


def test_partly_matching_documents_are_left_alone():
    """Сборник и статья из него совпадают частично — это решать человеку."""
    assert dedup.duplicate_documents(
        [pair('сборник', 'статья', 0.84)], {'сборник': 900, 'статья': 40}) == {}


def test_at_equal_depth_the_better_parsed_copy_wins():
    copies = dedup.duplicate_documents(
        [pair('а/отчёт', 'б/отчёт', 1.0)], {'а/отчёт': 12, 'б/отчёт': 300})
    assert copies == {'а/отчёт': 'б/отчёт'}
