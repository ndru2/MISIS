"""Сравнение текста блоков до и после ресвязывания с текстовым слоем."""

import sys

from unstructured.partition.pdf import partition_pdf

from pdfscan.parse.text_layer import rebind_elements

pdf_path = sys.argv[1] if len(sys.argv) > 1 else 'formul.pdf'

elements = partition_pdf(
    filename=pdf_path,
    strategy='hi_res',
    infer_table_structure=True,
    include_page_breaks=True,
    languages=['rus'],
)
before = [str(el.text) for el in elements]

stats = rebind_elements(elements, pdf_path)
print('STATS:', stats)
print('=' * 110)

changed = 0
for old, el in zip(before, elements):
    new = str(el.text)
    if old.strip() == new.strip():
        continue
    changed += 1
    info = getattr(el, 'text_layer', {})
    print(f"[{el.category}] page={getattr(el.metadata, 'page_number', '?')} "
          f"math={info.get('math_font_ratio', 0):.2f} it={info.get('italic_ratio', 0):.2f} "
          f"sub={info.get('n_subscript', 0)} sup={info.get('n_superscript', 0)} "
          f"size={info.get('font_size', 0):.1f}")
    print(f"  БЫЛО:  {old[:220]!r}")
    print(f"  СТАЛО: {new[:220]!r}")
    print(f"  ЧИТАЕМО: {info.get('text_plain', '')[:220]!r}")
    print('-' * 110)

print(f'Изменено блоков: {changed} из {len(elements)}')
