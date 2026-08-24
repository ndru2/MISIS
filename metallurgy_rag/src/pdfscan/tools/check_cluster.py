"""Регрессионная проверка графовой кластеризации.

Блок, чья вертикальная полоса накрывает другой блок той же страницы, — признак
слияния, проглотившего чужой текст и сломавшего порядок чтения.
"""

import sys
from collections import defaultdict

from unstructured.partition.pdf import partition_pdf

from pdfscan.parse.extractor import SmartPDFExtractor
from pdfscan.parse.text_layer import rebind_elements


def spans(elements):
    by_page = defaultdict(list)
    for el in elements:
        coords = getattr(el.metadata, 'coordinates', None)
        if not coords or not coords.points:
            continue
        ys = [p[1] for p in coords.points]
        by_page[getattr(el.metadata, 'page_number', 0)].append(
            (min(ys), max(ys), str(el.category), str(el.text))
        )
    return by_page


for pdf_path in sys.argv[1:]:
    extractor = SmartPDFExtractor(pdf_path)
    extractor.elements = partition_pdf(
        filename=pdf_path, strategy='hi_res', infer_table_structure=True,
        include_page_breaks=True, languages=['rus'],
    )
    rebind_elements(extractor.elements, pdf_path)


    def count_overlaps(elements, report):
        found = 0
        for page, items in spans(elements).items():
            items.sort()
            for i, (top1, bottom1, cat1, text1) in enumerate(items):
                for top2, bottom2, cat2, text2 in items[i + 1:]:
                    if top2 >= bottom1:
                        break
                    # Перекрытие больше половины меньшего блока — не случайность.
                    overlap = min(bottom1, bottom2) - max(top1, top2)
                    if overlap > 0.5 * min(bottom1 - top1, bottom2 - top2):
                        found += 1
                        if report:
                            print(f"  ⚠️ стр.{page} наложение [{cat1}] и [{cat2}]")
                            print(f"      {text1[:90]!r}")
                            print(f"      {text2[:90]!r}")
        return found


    kept = [el for el in extractor.elements if str(el.text).strip()]
    before, before_overlaps = len(kept), count_overlaps(kept, report=False)
    extractor.apply_graph_clustering()
    after_overlaps = count_overlaps(extractor.elements, report=True)

    print(f"{pdf_path}: блоков {before} -> {len(extractor.elements)}, "
          f"наложений {before_overlaps} -> {after_overlaps}")
