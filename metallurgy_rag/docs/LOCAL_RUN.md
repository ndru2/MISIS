# Локальный запуск metallurgy RAG

## Что должно быть запущено

1. **Qdrant** на `http://localhost:6333` с коллекцией
   `metallurgy_search_units`.
2. **Ollama** на `http://localhost:11434` с Qwen, например `qwen3:8b`.
3. Python 3.11+ и виртуальное окружение проекта.

Проверка сервисов:

```bash
curl http://localhost:6333/collections
ollama list
```

Если Ollama установлен, но сервер не запущен:

```bash
ollama serve
```

## Первичная установка Python

Из корня `metallurgy_rag`:

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements/base.txt
venv/bin/python -m pip install -e '.[rag,web]'
```

`venv/bin/python -m pip` используется намеренно: так команда не зависит от
абсолютного пути, с которым когда-либо было создано окружение.

## Индексация корпуса

Этот шаг нужен только для пустого Qdrant или после изменения схемы индекса.
`--recreate` удаляет текущую коллекцию и пересоздаёт её.

```bash
PYTHONPATH=src venv/bin/python -m pdfscan.rag.build qdrant \
  --source split --recreate
```

Корпус берётся из `corpus_split/`. После индексации не удаляйте
`data/qdrant_vocab.json`: в нём хранится словарь sparse/BM25-термов.

## Запуск интерфейса

Терминал 1 — FastAPI:

```bash
PYTHONPATH=src venv/bin/python -m uvicorn pdfscan.web.api:app \
  --reload --port 8000
```

Терминал 2 — Streamlit:

```bash
PYTHONPATH=src venv/bin/python -m streamlit run src/pdfscan/web/app.py
```

Откройте адрес из вывода Streamlit, обычно `http://localhost:8501`.

Проверка API:

```bash
curl http://localhost:8000/health
```

## Поиск и диагностика без интерфейса

```bash
# Hybrid BM25 + dense-vector поиск по одному пространству
PYTHONPATH=src venv/bin/python -m pdfscan.rag.build qsearch 'SO2 в отходящих газах' \
  --kind chemistry

# Полная цепочка: RRF, reranker, порог релевантности и Evidence
PYTHONPATH=src venv/bin/python -m pdfscan.rag.build retrieve \
  'Каково содержание SO2 в отходящих газах?'

# Итоговый ответ Qwen с RAG только при прошедших порог фрагментах
PYTHONPATH=src venv/bin/python -m pdfscan.rag.answer ask \
  'Каково содержание SO2 в отходящих газах?'
```

При первом запросе могут быть скачаны embedding-модель `BAAI/bge-m3` и
reranker `BAAI/bge-reranker-v2-m3`. Модели кешируются в Hugging Face cache.

## Настройка порога

По умолчанию evidence проходит в LLM при `relevance_score >= 0.55`.
Перед запуском FastAPI можно изменить значение:

```bash
export RAG_RELEVANCE_THRESHOLD=0.65
PYTHONPATH=src venv/bin/python -m uvicorn pdfscan.web.api:app --reload --port 8000
```

Повышение порога уменьшает риск нерелевантных цитат; понижение увеличивает
вероятность использовать RAG. Настраивайте его по экспертным вопросам, а не по
одному запросу.
