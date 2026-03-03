# Быстрый старт

## Подготовка

Создайте папку data и положите туда PDF или DOCX файлы

```bash
mkdir data
cp /path/to/files/*.pdf data/
```

## Вариант 1: Локальный запуск (без Docker)

Установите зависимости один раз

```bash
brew install tesseract tesseract-lang poppler
pip install -r requirements.txt
```

Запустите обработку

```bash
python extract_chemical_formulas.py all
```

Обработать только первые 3 файла

```bash
MAX_FILES=3 python extract_chemical_formulas.py all
```

Обработать конкретные файлы

```bash
python extract_chemical_formulas.py "file1.pdf" "file2.docx"
```

## Вариант 2: Docker

Соберите Docker образ один раз

```bash
docker-compose build
```

Запустите обработку всех файлов

```bash
docker-compose run --rm chemical-extractor python extract_chemical_formulas.py all
```

Обработать только первые 5 файлов

```bash
docker-compose run --rm -e MAX_FILES=5 chemical-extractor python extract_chemical_formulas.py all
```

Обработать конкретные файлы

```bash
docker-compose run --rm chemical-extractor python extract_chemical_formulas.py "file1.pdf"
```

## Результаты

Файлы сохраняются в папку chemical_results

- chemical_formulas.json - найденные формулы
- chemical_reactions.json - найденные реакции

## Справка

Показать все варианты запуска

```bash
python extract_chemical_formulas.py
```
