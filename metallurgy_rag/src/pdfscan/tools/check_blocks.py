"""Показывает блоки до и после графовой кластеризации."""

import sys

from unstructured.partition.pdf import partition_pdf

from pdfscan.parse.extractor import SmartPDFExtractor
from pdfscan.parse.text_layer import rebind_elements

pdf_path = sys.argv[1]
page_filter = int(sys.argv[2]) if len(sys.argv) > 2 else None

extractor = SmartPDFExtractor(pdf_path)
extractor.elements = partition_pdf(
    filename=pdf_path, strategy='hi_res', infer_table_structure=True,
    include_page_breaks=True, languages=['rus'],
)
rebind_elements(extractor.elements, pdf_path)


def dump(title, elements):
    print('=' * 110)
    print(title)
    for el in elements:
        page = getattr(el.metadata, 'page_number', None)
        if page_filter and page != page_filter:
            continue
        info = getattr(el, 'text_layer', {})
        coords = getattr(el.metadata, 'coordinates', None)
        box = ''
        if coords and coords.points:
            xs = [p[0] for p in coords.points]
            ys = [p[1] for p in coords.points]
            box = f"x=[{min(xs):5.0f},{max(xs):5.0f}] y=[{min(ys):5.0f},{max(ys):5.0f}]"
        print(f"  p{page} [{str(el.category)[:18]:<18}] fs={info.get('font_size', 0):4.1f} "
              f"math={info.get('math_font_ratio', 0):.2f} {box}")
        print(f"      {str(el.text)[:150]!r}")


dump('ДО КЛАСТЕРИЗАЦИИ', list(extractor.elements))
extractor.apply_graph_clustering()
dump('ПОСЛЕ КЛАСТЕРИЗАЦИИ', list(extractor.elements))
