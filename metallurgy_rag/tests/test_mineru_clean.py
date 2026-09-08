"""Очистка MinerU: обрамление, пустые, подписи без описания, склейка, дедуп."""

from pdfscan.parse.mineru import convert_document
from pdfscan.prepare import furniture, metadata
from pdfscan.prepare.mineru_clean import clean_document


def _records(*blocks):
    return convert_document([list(blocks)], 'док')


def _para(text, page=1):
    return {
        'type': 'paragraph',
        'content': {'paragraph_content': [{'type': 'text', 'content': text}]},
        'bbox': [70, 100, 500, 120],
        '_page': page,
    }


def test_page_numbers_and_headers_are_dropped():
    data = [[
        {
            'type': 'title',
            'content': {'title_content': [{'type': 'text', 'content': 'Плавка в печи Ванюкова'}],
                        'level': 1},
            'bbox': [10, 10, 200, 40],
        },
        {
            'type': 'page_number',
            'content': {'page_number_content': [{'type': 'text', 'content': '185'}]},
            'bbox': [800, 900, 860, 940],
        },
        {
            'type': 'page_header',
            'content': {'page_header_content': [{'type': 'text', 'content': 'Pyrometallurgy 2011'}]},
            'bbox': [100, 20, 400, 40],
        },
        {
            'type': 'paragraph',
            'content': {'paragraph_content': [{'type': 'text', 'content': 'Шлак сливают в ковш.'}]},
            'bbox': [70, 200, 400, 230],
        },
    ]]
    rows = convert_document(data, 'док')
    report = clean_document(rows)
    kept = [r for r in rows if r['keep']]
    types = {r['type'] for r in kept}
    assert report['counts']['furniture'] >= 2
    assert 'PageNumber' not in types
    assert 'Header' not in types
    assert any('ковш' in (r.get('text_out') or '') for r in kept)
    assert any(r['type'] == 'Title' for r in kept)


def test_empty_paragraph_is_dropped_table_without_caption_is_kept():
    data = [[
        {'type': 'table', 'content': {
            'table_caption': [],
            'html': '<table><tr><td>Cu</td><td>19.46</td></tr></table>',
        }, 'bbox': [1, 2, 3, 4]},
    ]]
    rows = convert_document(data, 'док')
    rows.insert(0, {
        'doc_id': 'док', 'block_id': 'док#-1', 'order': -1, 'page': 1,
        'type': 'NarrativeText', 'text': '   ', 'table_html': None,
        'reliable': True,
    })
    report = clean_document(rows)
    assert report['counts']['empty'] >= 1
    tables = [r for r in rows if r['type'] == 'Table']
    assert tables and tables[0]['keep']
    assert tables[0]['table_html']


def test_unfinished_sentence_is_merged_across_the_page():
    data = [
        [{'type': 'paragraph', 'content': {'paragraph_content': [
            {'type': 'text', 'content': 'Медь извлекают из сульфидного'}]},
          'bbox': [70, 700, 400, 720]}],
        [{'type': 'paragraph', 'content': {'paragraph_content': [
            {'type': 'text', 'content': 'концентрата в печи Ванюкова.'}]},
          'bbox': [70, 80, 400, 100]}],
    ]
    rows = convert_document(data, 'док')
    report = clean_document(rows)
    assert report['counts']['merged'] >= 1
    kept = [r for r in rows if r['keep']]
    assert any('сульфидного концентрата' in (r.get('text_out') or '') for r in kept)


def test_exact_duplicate_inside_the_document_is_dropped():
    text = 'Один и тот же абзац длиной больше сорока символов стоит дважды.'
    data = [[
        {'type': 'paragraph', 'content': {'paragraph_content': [{'type': 'text', 'content': text}]},
         'bbox': [1, 2, 3, 4]},
        {'type': 'paragraph', 'content': {'paragraph_content': [{'type': 'text', 'content': text}]},
         'bbox': [1, 20, 3, 40]},
    ]]
    rows = convert_document(data, 'док')
    report = clean_document(rows)
    assert report['counts']['duplicate'] == 1
    assert sum(r['keep'] for r in rows if r['type'] == 'NarrativeText') == 1


def test_panel_caption_is_dropped_descriptive_caption_stays():
    data = [[
        {'type': 'chart', 'content': {
            'content': '',
            'chart_caption': [{'type': 'text', 'content': '(a)'}],
        }, 'bbox': [1, 2, 3, 4]},
        {'type': 'chart', 'content': {
            'content': '',
            'chart_caption': [{'type': 'text', 'content': 'Figure 1: Copper in SCF slag'}],
        }, 'bbox': [1, 20, 3, 40]},
        {'type': 'image', 'content': {
            'content': '',
            'image_caption': [{'type': 'text', 'content': 'Рис. 2.5. Структура льда'}],
        }, 'bbox': [1, 40, 3, 60]},
        {'type': 'image', 'content': {
            'content': '',
            'image_caption': [{'type': 'text', 'content': '900°C'}],
        }, 'bbox': [1, 60, 3, 80]},
    ]]
    rows = convert_document(data, 'док')
    report = clean_document(rows)
    kept = [r['text_out'] for r in rows if r['keep']]
    assert report['counts']['caption_only'] >= 2
    assert any('Copper in SCF slag' in text for text in kept)
    assert any('Структура льда' in text for text in kept)
    assert not any(text.strip() in {'(a)', '900°C'} for text in kept)


def test_named_caption_is_not_treated_as_panel_letters():
    from pdfscan.prepare.furniture import is_caption_only
    assert is_caption_only('(a)')
    assert is_caption_only('(a) (b)')
    assert is_caption_only('a)')
    assert is_caption_only('900°C')
    assert is_caption_only('Fig. 12')
    assert not is_caption_only('Тигель с расплавом')
    assert not is_caption_only('Pechenga')
    assert not is_caption_only('Figure 1: Copper in SCF slag')
    assert not is_caption_only('Рис. 2.5. Структура льда')


def test_figure_caption_stays_pixels_never_arrived():
    data = [[
        {'type': 'chart', 'content': {
            'content': '100.00 + Jul-09',
            'chart_caption': [{'type': 'text', 'content': 'Figure 1: Copper in SCF slag'}],
        }, 'bbox': [1, 2, 3, 4]},
    ]]
    rows = convert_document(data, 'док')
    clean_document(rows)
    kept = [r for r in rows if r['keep']]
    assert len(kept) == 1
    assert kept[0]['type'] == 'FigureCaption'
    assert 'Jul-09' not in kept[0]['text_out']


def test_drop_furniture_helper_only_touches_frame_types():
    rows = [
        {'type': 'NarrativeText', 'keep': True, 'text': 'шлак'},
        {'type': 'PageNumber', 'keep': True, 'text': '12'},
        {'type': 'Title', 'keep': True, 'text': 'Введение'},
    ]
    assert furniture.drop_furniture(rows) == 1
    assert rows[0]['keep'] and rows[2]['keep']
    assert not rows[1]['keep']
    assert rows[1]['drop_reason'] == 'furniture'
