"""
Пакетный прогон mineru по дереву папок "Литература для чат-бота".

Особенности:
- обходит все вложенные папки, берёт pdf/docx (+ обычные картинки, если попадутся);
- .djvu и прочие неподдерживаемые форматы явно логируются как пропущенные, а не тонут молча;
- каждый документ обрабатывается ОТДЕЛЬНЫМ вызовом do_parse — если файл битый и роняет
  исключение, это не обрывает всю пачку: ошибка ловится, файл помечается как failed,
  и скрипт идёт дальше;
- после успешной обработки в папку результата кладётся маркер .done — при повторном
  запуске такие документы пропускаются (докачка после обрыва/перезапуска);
- прогресс печатается по документам (i/N) и по страницам внутри документа (штатные
  tqdm-бары самого mineru: Layout/OCR/Formula/Table идут по страницам текущего файла).
"""
import hashlib
import re
import sys
import time
from pathlib import Path

import pypdfium2 as pdfium
from loguru import logger

from mineru.cli.common import do_parse, read_fn
from mineru.cli.output_paths import build_parse_dir
from mineru.utils.pdfium_guard import (
    close_pdfium_document,
    get_pdfium_document_page_count,
    open_pdfium_document,
)

# ---- настройки ------------------------------------------------------------

INPUT_DIR = Path(r"pdf")
OUTPUT_DIR = Path("parsed_literature")

BACKEND = "hybrid-engine"   # то же, что дефолт CLI и то, что вы уже проверили на качество
PARSE_METHOD = "auto"
LANG = "ru"                 # используется только если переключите BACKEND на "pipeline"

PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".jp2"}
SUPPORTED_EXTS = PDF_EXTS | DOCX_EXTS | IMAGE_EXTS

DONE_MARKER = ".done"
FAILED_LOG = OUTPUT_DIR / "failed_files.log"

_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*]')


# ---- вспомогательное --------------------------------------------------------

def make_doc_id(rel_path: Path, max_len: int = 100) -> str:
    """Плоский уникальный id документа без разделителей пути (mineru не умеет
    во вложенные имена — сам путь конфликтует с именами файлов результата)."""
    parts = [_ILLEGAL_CHARS.sub("_", p) for p in rel_path.with_suffix("").parts]
    doc_id = "__".join(parts)
    if len(doc_id) > max_len:
        digest = hashlib.sha1(str(rel_path).encode("utf-8")).hexdigest()[:8]
        doc_id = f"{doc_id[:max_len]}_{digest}"
    return doc_id


def parse_dir_for(doc_id: str, suffix: str) -> Path:
    is_office = suffix in DOCX_EXTS
    return build_parse_dir(OUTPUT_DIR, doc_id, BACKEND, PARSE_METHOD, is_office=is_office)


def is_done(parse_dir: Path) -> bool:
    return (parse_dir / DONE_MARKER).exists()


def mark_done(parse_dir: Path) -> None:
    parse_dir.mkdir(parents=True, exist_ok=True)
    (parse_dir / DONE_MARKER).write_text("ok", encoding="utf-8")


def log_failed(rel_path: Path, error: Exception) -> None:
    FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FAILED_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{rel_path}\t{type(error).__name__}: {error}\n")


def page_count_hint(path: Path) -> str:
    if path.suffix.lower() not in PDF_EXTS:
        return ""
    try:
        doc = open_pdfium_document(pdfium.PdfDocument, str(path))
        try:
            n = get_pdfium_document_page_count(doc)
        finally:
            close_pdfium_document(doc)
        return f", {n} стр."
    except Exception:
        return ""


# ---- основной прогон --------------------------------------------------------

def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    )

    if not INPUT_DIR.exists():
        logger.error(f"Папка не найдена: {INPUT_DIR.resolve()}")
        return

    all_files = sorted(p for p in INPUT_DIR.rglob("*") if p.is_file())
    files = [p for p in all_files if p.suffix.lower() in SUPPORTED_EXTS]
    skipped_unsupported = [p for p in all_files if p.suffix.lower() not in SUPPORTED_EXTS]

    for p in skipped_unsupported:
        logger.warning(f"Пропуск — неподдерживаемый формат: {p.relative_to(INPUT_DIR)}")

    total = len(files)
    logger.info(f"К обработке: {total} файлов (backend={BACKEND})")

    done = 0
    skipped_already = 0
    failed = 0
    t_start = time.monotonic()

    for i, path in enumerate(files, start=1):
        rel = path.relative_to(INPUT_DIR)
        doc_id = make_doc_id(rel)
        parse_dir = parse_dir_for(doc_id, path.suffix.lower())

        if is_done(parse_dir):
            skipped_already += 1
            logger.info(f"[{i}/{total}] уже готово, пропуск: {rel}")
            continue

        pages = page_count_hint(path)
        logger.info(f"[{i}/{total}] обработка: {rel}{pages}")
        t0 = time.monotonic()

        try:
            pdf_bytes = read_fn(path)
            do_parse(
                output_dir=str(OUTPUT_DIR),
                pdf_file_names=[doc_id],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=[LANG],
                backend=BACKEND,
                parse_method=PARSE_METHOD,
                formula_enable=True,
                table_enable=True,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
            )
        except Exception as exc:
            failed += 1
            logger.exception(f"[{i}/{total}] ОШИБКА, пропускаю и иду дальше: {rel}")
            log_failed(rel, exc)
            continue

        mark_done(parse_dir)
        done += 1
        dt = time.monotonic() - t0
        logger.info(f"[{i}/{total}] готово за {dt:.1f} c: {rel}")

    elapsed = time.monotonic() - t_start
    logger.info(
        f"Итог: обработано {done}, уже было готово {skipped_already}, "
        f"с ошибками {failed}, всего файлов {total}, время {elapsed/60:.1f} мин"
    )
    if failed:
        logger.warning(f"Список ошибок: {FAILED_LOG.resolve()}")


if __name__ == "__main__":
    main()
