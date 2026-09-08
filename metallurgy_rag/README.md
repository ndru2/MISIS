# metallurgy_rag

Плоский multi-vector RAG по металлургической литературе. PDF разбираются
MinerU и раскладываются в `corpus_split`; Qdrant хранит parent chunks и
дочерние search units. Графа и Wikontic в этом проекте больше нет.

## Данные

`corpus_split/` — нормализованный корпус из 284 документов:

```text
text/            проза, заголовки, подписи
formulas_chem/   химические формулы и реакции
formulas_math/   математические и физические формулы
tables/          подписи и HTML-таблицы
all/             объединённые JSONL
manifest.json    состав и статистика корпуса
```

## Индексация

Один документ превращается в плоские `SearchUnit` с устойчивыми ID:

```text
chunk
├─ formula      parent_chunk_id → chunk
├─ table_row    parent_chunk_id → chunk
├─ table_cell   parent_chunk_id → chunk
└─ quantity     parent_chunk_id → chunk
```

У каждого объекта есть исходный текст для цитирования, `text_search` для
лексического поиска, metadata/provenance и один специализированный dense-vector:

| Search unit | Dense space | Дополнительно |
|---|---|---|
| `chunk` | `dense_text` | весь нормализованный текст в sparse BM25 |
| `formula` (химия) | `dense_chemistry` | LaTeX, flat-форма, элементы, номер |
| `formula` (математика/физика) | `dense_math` | LaTeX, flat-форма, номер, контекст |
| `table_row`, `table_cell` | `dense_table` | подпись, строка, шапка, значение |
| `quantity` | `dense_unit` | value, raw/canonical/SI unit |

Dense-модель каждого пространства выбирается отдельно CLI-флагом. По умолчанию
это `BAAI/bge-m3`, с запасным `intfloat/multilingual-e5-base`; разные named
vectors можно переключать на специализированные модели без изменения схемы.

В Qdrant гибридный запрос объединяет sparse BM25 и выбранные dense spaces через
RRF. Вектора не склеиваются в один embedding.

## Запуск

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements/base.txt
venv/bin/python -m pip install -e ".[rag,classify]" --no-deps
docker compose up -d

# Извлечь/обновить corpus_split при необходимости
python -m pdfscan.prepare.split_corpus parsed_literature --out corpus_split

# Создать новый индекс. --recreate очищает и старые metallurgy_chunks/
# metallurgy_triples коллекции, и новую коллекцию перед полной пересборкой.
python -m pdfscan.rag.build qdrant --source split --recreate

# Проверить структуру без загрузки эмбеддингов
python -m pdfscan.rag.build qdrant --source split --dry-run

# Гибридный поиск по всем пространствам или по одному из них
python -m pdfscan.rag.build qsearch "Fe3O4 + C" --kind chemistry
python -m pdfscan.rag.build qsearch "уравнение 5.22 K_Ni/Ca" --kind math
python -m pdfscan.rag.build qsearch "Cu в шлаке FSF" --kind table
python -m pdfscan.rag.build qsearch "энергия активации кДж/моль" --kind unit

# Production retrieval: router выбирает каналы, убирает parent-child дубли
# и возвращает точный evidence вместе с контекстом родительского chunk.
python -m pdfscan.rag.build retrieve "Cu content in FSF slag Table I"
python -m pdfscan.rag.build retrieve "температура выше 1200 °C"
```

## Локальная LLM и ответы с evidence

Генерация вынесена из retrieval в независимый адаптер. По умолчанию это
локальный Ollama с `qwen3:8b`; его можно заменить совместимым с OpenAI
endpoint (например, vLLM или LM Studio), не меняя ни Qdrant, ни код поиска.

```bash
# Один раз: установить Ollama и загрузить модель Qwen.
brew install ollama
ollama pull qwen3:8b

# Один grounded-ответ. В ответ попадают только сведения из retrieved Evidence.
PYTHONPATH=src venv/bin/python -m pdfscan.rag.answer ask \
  "Как температура влияет на восстановление оксидов железа?"

# Прогнать вопросы из экспертной таблицы, не меняя исходный XLSX.
PYTHONPATH=src venv/bin/python -m pdfscan.rag.answer batch \
  --workbook "Экспертная таблица (Nord+Эксперты).xlsx"

# Быстрый пробный прогон и явный путь к результату.
PYTHONPATH=src venv/bin/python -m pdfscan.rag.answer batch --limit 2 \
  --out reports/llm_runs/qwen3_8b_smoke.jsonl
```

Каждая строка JSONL содержит вопрос, полный ответ, найденные Evidence с
provenance, ссылки `[E#]`, профиль router, модель, provider и коллекцию
Qdrant. Генератор может ответить и на основе общих знаний модели, если корпус
не даёт прямого подтверждения; тогда ссылки на нерелевантные Evidence не ставятся.
Это позволяет сравнивать Qwen разных размеров и будущую
серверную модель на одном и том же наборе вопросов. Источник
`Экспертная таблица (Nord+Эксперты).xlsx` остаётся неизменным: в нём есть
вопросы и оценки исторических ответов, но нет единого эталонного текста, по
которому можно автоматически оценить новый ответ.

Для внешнего сервера передайте адрес и модель; ключ читается только из
переменной окружения `RAG_LLM_API_KEY`:

```bash
PYTHONPATH=src venv/bin/python -m pdfscan.rag.answer ask "ваш вопрос" \
  --provider openai-compatible --base-url http://localhost:8000/v1 \
  --model YOUR_MODEL
```

## Интерфейс: FastAPI + Streamlit

FastAPI — единственная точка, которая вызывает retrieval, Qdrant и LLM.
Streamlit — отдельный клиент: он не знает адрес Qdrant и показывает отдельно
текст ответа и полный набор Evidence, из которого тот собран. Поэтому второй
графовый RAG позднее сможет подключиться тем же HTTP-контрактом.

```bash
# Устанавливается один раз в рабочее venv.
venv/bin/python -m pip install -e '.[web]'

# Терминал 1: API. RAG_QDRANT_* позволяют выбрать другой индекс.
PYTHONPATH=src venv/bin/python -m uvicorn pdfscan.web.api:app --reload --port 8000

# Терминал 2: веб-клиент.
PYTHONPATH=src venv/bin/python -m streamlit run src/pdfscan/web/app.py
```

Откройте адрес, который напечатает Streamlit (обычно `http://localhost:8501`).
В левом меню выбираются LLM-provider, название модели и число Evidence. Для
проверки состояния API: `curl http://127.0.0.1:8000/health`.

## Docker: воспроизводимый запуск для другого человека

В проекте есть три независимых runtime-сервиса: Qdrant хранит индекс, FastAPI
выполняет retrieval и вызывает LLM, Streamlit показывает интерфейс. Локальный
Ollama намеренно остаётся на хосте: на Mac он использует Metal напрямую, тогда
как запуск модели внутри Linux-контейнера Docker Desktop лишит его этого пути.
Контейнер API подключается к Ollama через `host.docker.internal`.

```bash
# Один раз после clone: настройка без секретов.
cp .env.example .env

# Собрать приложение и поднять Qdrant, API и интерфейс.
docker compose up --build -d

# Проверить, что API видит свою конфигурацию, затем открыть интерфейс.
curl http://127.0.0.1:8000/health
# http://localhost:8501
```

`data/qdrant` в текущей конфигурации — постоянное хранилище Qdrant. Для
готового рабочего поиска другому человеку нужно передать **весь** каталог
`data/qdrant` и `data/qdrant_vocab.json`; копировать каталог нужно при
остановленном Qdrant. Текущий индекс занимает около 11 ГБ. Их не следует
добавлять в Git или Docker image. Альтернатива — передать
`corpus_split` (около 462 МБ) и один раз собрать индекс:

```bash
# Внимание: удаляет выбранную Qdrant-коллекцию и пересобирает её из corpus_split.
docker compose --profile tools run --rm indexer
```

При переносе на другой ПК укажите в `.env` адрес внешнего LLM или оставьте
локальный Ollama. Qdrant, API и UI не зависят от выбранной модели. Для
остановки сервисов без удаления индекса: `docker compose down`. Не используйте
`docker compose down -v`: он удалит named volume с кешем embedding-моделей.

Для отдельной модели химии, например:

```bash
python -m pdfscan.rag.build qdrant --recreate \
  --chemistry-model YOUR_CHEMISTRY_MODEL \
  --math-model YOUR_MATH_MODEL
```

Автоматическая оценка качества остаётся отдельным следующим этапом. Будущий
графовый RAG на Neo4j должен возвращать такой же контракт evidence (`unit_id`,
`parent_chunk_id`, источник, страница, текст), поэтому сможет подключаться к
этому же FastAPI-интерфейсу, но не является зависимостью плоского
Qdrant-пайплайна.

## Retrieval contract

`pdfscan.rag.retrieval.retrieve()` возвращает `QueryProfile` и список
`Evidence`. Router выбирает dense channels по признакам запроса, BM25 остаётся
всегда. Qdrant объединяет ранги этих каналов RRF; затем дочерние объекты
группируются по `parent_chunk_id`. После этого multilingual cross-encoder
оценивает пару «вопрос — фрагмент». В генератор проходят только фрагменты с
`relevance_score >= RAG_RELEVANCE_THRESHOLD` (по умолчанию `0.55`).

Так RAG не обязан «найти хоть что-то»: когда top-K содержит лишь похожие, но
не отвечающие на вопрос куски, Evidence будет пустым и LLM вернёт собственный
ответ без фальшивых ссылок. При прошедших порог источниках ответ строится в два
шага: самостоятельный черновик LLM, затем его уточнение подтверждёнными
фрагментами с `[E#]`. Для простых единиц выражения «выше / ниже / more than /
less than» превращаются в Qdrant range-filter по `si_value`. Такой же контракт
должен соблюсти будущий Neo4j-retriever.
