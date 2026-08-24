"""Вырезка картинок формул по рамкам из первого прохода.

Рамка блока лежит в пунктах PDF с началом в левом верхнем углу, а рендер отдаёт
пиксели, поэтому координаты пересчитываются через размер страницы, записанный в
самом блоке: брать размер у PDF нельзя, страница могла быть повёрнута.

Страница рендерится один раз на все свои формулы. Разница не косметическая: в
учебнике по термодинамике на странице бывает десяток выключных формул, и
постраничный рендер на каждую означал бы десятикратную работу — при трёхстах
точках на дюйм это минуты на документ.

Результат — картинки на диске и манифест. Разделение нужно для продолжения
прерванной работы: распознавание тяжелее вырезки и запускается отдельно.
"""

import json
from collections import defaultdict
from pathlib import Path

from pdfscan import paths

from . import config, select


def resolve_pdf(block: dict):
    """Находит исходный PDF для блока.

    В выгрузке есть поле ``source`` с путём от корня проекта — этого почти всегда
    достаточно. Поиск по имени остаётся на случай, когда библиотеку перенесли.
    """
    source = block.get('source')
    if source:
        candidate = Path(source)
        if not candidate.is_absolute():
            candidate = paths.ROOT / candidate
        if candidate.is_file():
            return candidate

    doc_id = block.get('doc_id') or ''
    stem = Path(doc_id).name
    if stem:
        found = sorted(paths.PDF_DIR.rglob(f'{stem}.pdf'))
        if found:
            return found[0]
    return None


def crop_name(block: dict) -> str:
    """Имя файла картинки, устойчивое между запусками."""
    tail = str(block.get('block_id', '')).split('#')[-1] or str(block.get('order'))
    return f"p{int(block.get('page', 0)):04d}_b{tail}.png"


def _crop_box(block, image_size, page_size, cfg):
    """Пересчитывает рамку блока в пиксели картинки страницы."""
    image_width, image_height = image_size
    page_width = (page_size or {}).get('width')
    page_height = (page_size or {}).get('height')
    if not page_width or not page_height:
        return None

    scale_x = image_width / page_width
    scale_y = image_height / page_height
    box = block['bbox']

    left = max(0, box['x0'] * scale_x - cfg.padding_pt * scale_x)
    top = max(0, box['top'] * scale_y - cfg.padding_pt * scale_y)
    right = min(image_width, box['x1'] * scale_x + cfg.padding_pt * scale_x)
    bottom = min(image_height, box['bottom'] * scale_y + cfg.padding_pt * scale_y)

    width, height = right - left, bottom - top
    if width < cfg.min_crop_px or height < cfg.min_crop_px:
        return None
    if width > cfg.max_crop_px or height > cfg.max_crop_px:
        return None
    return int(left), int(top), int(right), int(bottom)


def crop_document(blocks_path, out_dir, cfg=config.DEFAULT, progress=True) -> list:
    """Вырезает все нужные формулы одного документа.

    Возвращает записи манифеста. Картинки, вырезанные ранее, не перерисовываются.
    """
    import pypdfium2 as pdfium

    candidates = list(select.iter_candidates(blocks_path, cfg))
    if not candidates:
        return []

    pdf_path = resolve_pdf(candidates[0][0])
    if pdf_path is None:
        if progress:
            print(f'  [!] PDF не найден для {blocks_path.parent.name}, пропускаю')
        return []

    by_page = defaultdict(list)
    for block, why in candidates:
        by_page[block['page']].append((block, why))

    out_dir.mkdir(parents=True, exist_ok=True)
    records, skipped = [], 0
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        scale = cfg.render_dpi / 72.0
        for page_number in sorted(by_page):
            index = page_number - 1
            if index < 0 or index >= len(pdf):
                skipped += len(by_page[page_number])
                continue

            pending = [(block, why) for block, why in by_page[page_number]
                       if not (out_dir / crop_name(block)).exists()]
            done = [(block, why) for block, why in by_page[page_number]
                    if (out_dir / crop_name(block)).exists()]

            for block, why in done:
                records.append(_record(block, why, out_dir / crop_name(block),
                                       pdf_path, blocks_path, cfg))

            if not pending:
                continue

            # Один рендер на страницу, а не на формулу.
            image = pdf[index].render(scale=scale, draw_annots=False).to_pil()
            for block, why in pending:
                box = _crop_box(block, image.size, block.get('page_size'), cfg)
                if box is None:
                    skipped += 1
                    continue
                target = out_dir / crop_name(block)
                image.crop(box).save(target)
                records.append(_record(block, why, target, pdf_path,
                                       blocks_path, cfg))
            image.close()
    finally:
        pdf.close()

    if progress and skipped:
        print(f'  пропущено рамок с негодной геометрией: {skipped}')
    return records


def _record(block, why, crop_path, pdf_path, blocks_path, cfg) -> dict:
    return {
        'doc_id': block['doc_id'],
        'block_id': block['block_id'],
        'page': block['page'],
        'type': block['type'],
        'text_source': block.get('text_source'),
        'text': block.get('text') or '',
        'reason': why,
        'pdf': str(pdf_path),
        'blocks': str(blocks_path),
        'crop': str(crop_path),
        'dpi': cfg.render_dpi,
    }


def build_manifest(blocks_files, cfg=config.DEFAULT, progress=True) -> Path:
    """Вырезает формулы по всем документам и пишет манифест."""
    config.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(config.MANIFEST, 'w', encoding='utf-8') as handle:
        for position, blocks_path in enumerate(blocks_files, 1):
            doc_dir = config.CROPS_DIR / blocks_path.parent.name
            records = crop_document(blocks_path, doc_dir, cfg, progress)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            total += len(records)
            if progress:
                print(f'[{position}/{len(blocks_files)}] '
                      f'{blocks_path.parent.name}: формул {len(records)}')
    if progress:
        print(f'всего вырезано формул: {total} → {config.MANIFEST}')
    return config.MANIFEST
