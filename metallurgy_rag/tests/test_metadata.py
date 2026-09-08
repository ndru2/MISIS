"""Метаданные документа с первой страницы и из пути папки."""

from pdfscan.parse.mineru import convert_document
from pdfscan.prepare import metadata


def test_category_and_type_come_from_the_folder_path():
    assert metadata.category_from_doc_id('Cu-Co сырье__185-Chikashi (1)') == 'Cu-Co сырье'
    assert metadata.doc_type_from_doc_id('Книги по цветной металлургии__Металлургия меди__Davenport') == 'book'
    assert metadata.doc_type_from_doc_id('ИТС 12-2019 Производство никеля и кобальта') == 'standard'
    assert metadata.doc_type_from_doc_id('Cu-Co сырье__185-Chikashi (1)') == 'article'


def test_year_is_read_from_the_filename_and_the_header():
    assert metadata.extract_document([{
        'doc_id': 'ИТС 12-2019 Производство никеля',
        'type': 'Title', 'text': 'Производство никеля', 'page': 1, 'order': 0,
        'category': 'ИТС',
    }])['year'] == 2019

    rows = convert_document([[
        {'type': 'title', 'content': {
            'title_content': [{'type': 'text', 'content': 'Influence of slag composition'}],
            'level': 1}, 'bbox': [1, 2, 3, 4]},
        {'type': 'page_header', 'content': {
            'page_header_content': [{'type': 'text',
                                     'content': 'Southern African Pyrometallurgy 2011'}]},
         'bbox': [1, 2, 3, 4]},
        {'type': 'paragraph', 'content': {
            'paragraph_content': [{'type': 'text', 'content': 'H.M. Chikashi Konkola Copper Mines plc'}]},
         'bbox': [1, 2, 3, 4]},
    ]], 'Cu-Co сырье__185-Chikashi (1)')
    meta = metadata.extract_document(rows)
    assert meta['year'] == 2011
    assert meta['lang'] == 'en'
    assert meta['title'].startswith('Influence of slag')
    assert any('Chikashi' in name for name in meta['authors'])


def test_russian_article_gets_title_year_and_language():
    rows = convert_document([[
        {'type': 'title', 'content': {
            'title_content': [{'type': 'text',
                               'content': 'Освоение процесса Ванюкова для переработки окисленных никелевых руд'}],
            'level': 1}, 'bbox': [1, 2, 3, 4]},
        {'type': 'page_footer', 'content': {
            'page_footer_content': [{'type': 'text',
                                     'content': 'ISSN 0372-2929 «Цветные металлы». 2007. № 12'}]},
         'bbox': [1, 2, 3, 4]},
        {'type': 'paragraph', 'content': {
            'paragraph_content': [{'type': 'text',
                                   'content': 'Плавка окисленных никелевых руд ведётся в печи Ванюкова.'}]},
         'bbox': [1, 2, 3, 4]},
    ]], 'Cu-Ni сырье__ОНР__ЦМ')
    meta = metadata.extract_document(rows)
    assert meta['year'] == 2007
    assert meta['lang'] == 'ru'
    assert 'Ванюкова' in meta['title']


def test_stamp_writes_fields_onto_every_block():
    rows = [{'doc_id': 'x', 'type': 'NarrativeText', 'text': 'шлак', 'page': 1, 'order': 0}]
    meta = {'year': 2011, 'authors': ['H.M. Chikashi'], 'title': 'Slag',
            'lang': 'en', 'doc_type': 'article', 'category': 'Cu-Co сырье'}
    metadata.stamp(rows, meta)
    assert rows[0]['year'] == 2011
    assert rows[0]['authors'] == 'H.M. Chikashi'
    assert rows[0]['lang'] == 'en'
