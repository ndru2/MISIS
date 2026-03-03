FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY extract_chemical_formulas.py .

RUN mkdir -p /app/data /app/chemical_results

ENV PYTHONUNBUFFERED=1

CMD ["python", "extract_chemical_formulas.py", "all"]
