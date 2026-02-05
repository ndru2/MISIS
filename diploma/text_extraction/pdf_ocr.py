import os
import sys
from pathlib import Path
from typing import Optional, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
except ImportError as e:
    logger.error(f"Не установлены необходимые библиотеки: {e}")
    sys.exit(1)


class PDFOCRExtractor:    
    def __init__(self, 
                 lang: str = 'rus',
                 dpi: int = 300,
                 save_images: bool = False):
        self.lang = lang
        self.dpi = dpi
        self.save_images = save_images

        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract версия: {version}")

            langs = pytesseract.get_languages()
            logger.info(f"Доступные языки: {', '.join(langs)}")

            if lang not in langs and '+' not in lang:
                logger.warning(f"Язык '{lang}' может быть недоступен")

        except Exception as e:
            logger.error(f"Ошибка Tesseract: {e}")
            sys.exit(1)

    def extract_text_from_pdf(self, 
                              pdf_path: str, 
                              output_path: Optional[str] = None,
                              start_page: Optional[int] = None,
                              end_page: Optional[int] = None) -> str:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF не найден: {pdf_path}")

        logger.info(f"Обработка PDF: {pdf_path.name}")
        logger.info(f"Параметры: DPI={self.dpi}, язык={self.lang}")

        logger.info("Конвертация PDF в изображения...")
        try:
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=start_page,
                last_page=end_page
            )
            logger.info(f"Получено {len(images)} страниц")
        except Exception as e:
            logger.error(f"Ошибка конвертации PDF: {e}")
            raise

        if self.save_images:
            images_dir = pdf_path.parent / f"{pdf_path.stem}_images"
            images_dir.mkdir(exist_ok=True)
            logger.info(f"Изображения будут сохранены в: {images_dir}")


        full_text = ""

        for i, image in enumerate(images, start=1):
            logger.info(f"OCR страницы {i}/{len(images)}...")

            if self.save_images:
                img_path = images_dir / f"page_{i:04d}.png"
                image.save(img_path, 'PNG')

            try:
                page_text = pytesseract.image_to_string(
                    image,
                    lang=self.lang,
                    config='--psm 1'
                )

                full_text += f"\n\n{'='*60}\n"
                full_text += f"СТРАНИЦА {i}\n"
                full_text += f"{'='*60}\n\n"
                full_text += page_text

                logger.info(f"  Распознано {len(page_text)} символов")

            except Exception as e:
                logger.error(f"Ошибка OCR страницы {i}: {e}")
                full_text += f"\n[ОШИБКА НА СТРАНИЦЕ {i}]\n"
        logger.info(f"\nИтого:")
        logger.info(f"  Страниц обработано: {len(images)}")
        logger.info(f"  Всего символов: {len(full_text)}")

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)

            logger.info(f"Текст сохранен в: {output_path}")

        return full_text
