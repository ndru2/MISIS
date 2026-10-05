# metallurgy_rag

Multi-vector и графовый RAG по металлургической литературе. PDF разбираются
MinerU и раскладываются в `corpus_split`; Qdrant хранит parent chunks и
дочерние search units; Neo4j хранит граф из `kg_rag`.

Подробное описание предметной области, данных, retrieval, RRF, reranker,
LLM Judge и экспериментов: [docs/SYSTEM_DESCRIPTION.md](docs/SYSTEM_DESCRIPTION.md).

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
provenance, ссылки `[E#]`, профиль router, модель, provider, коллекцию
Qdrant и вердикт LLM Judge (`judge_score`, `judge_model`, `iterations`,
`evidence_used`). Это позволяет сравнивать Qwen разных размеров и будущую
серверную модель на одном и том же наборе вопросов. Источник
`Экспертная таблица (Nord+Эксперты).xlsx` остаётся неизменным: в нём есть
вопросы и оценки исторических ответов, но нет единого эталонного текста, по
которому можно автоматически оценить новый ответ.

### Search-first, LLM Judge и явный fallback

`answer_question` (`src/pdfscan/rag/answering.py`) работает строго
search-first — сначала гибридный retrieval, LLM вызывается только если
нашлось релевантное evidence:

1. **Нет evidence** (`generation_mode = 'no_evidence'`) — LLM вообще не
   вызывается, возвращается явное сообщение "в корпусе нет релевантных
   документов". Ответ без RAG-подтверждения — это отдельный явный
   opt-in вызов, `generate_llm_only_answer` (в CLI — флаг
   `--allow-llm-fallback`, в API — `allow_llm_fallback: true`); сам
   pipeline его никогда не делает молча.
2. **Есть evidence** — ответ генерируется по top-1 фрагменту, затем
   независимый **LLM Judge** (`src/pdfscan/rag/judge.py`) оценивает,
   какую долю вопроса ответ покрывает (0..1). Если покрытие ниже порога
   (`--coverage-threshold`, по умолчанию 0.82) — контекст кумулятивно
   расширяется до top-2, top-3 (максимум `--max-iterations`, по
   умолчанию 3) и генерация повторяется, пока Judge не одобрит ответ
   или попытки не закончатся (тогда `generation_mode = 'hybrid_partial'`
   с лучшей из попыток).

Judge может быть отдельной, более тяжёлой моделью (например, бесплатным
tier-моделью через OpenRouter, `*:free`), не совпадающей с генератором:

```bash
PYTHONPATH=src venv/bin/python -m pdfscan.rag.answer ask "ваш вопрос" \
  --judge-provider openai-compatible \
  --judge-base-url https://openrouter.ai/api/v1 \
  --judge-model deepseek/deepseek-r1:free \
  --judge-api-key $OPENROUTER_API_KEY
```

Каждый `Evidence` уже несёт `relevance_score` (оценка cross-encoder
reranker) и полный provenance (`document_id`, `pages`, `unit_id`,
`retrieval_channels`, `structured`) — это и есть трассируемость "откуда
взят факт"; Streamlit UI показывает это в блоке "Источники RAG".

### Формат ввода: LaTeX и обычный текст работают одинаково

Запрос пользователя — обычная строка, без принудительного LaTeX и без
внешнего API-конвертера. `retrieval.retrieve` прогоняет её через
`normalize.normalize_query`, которая приводит только *формулоподобные*
токены к той же нотации, что и корпус (`\tau` → `τ`, `H_{2}O` → `H2O`,
кириллическое `Н2О` → `H2O`), а обычные слова — русские и английские —
не трогает (проверка идёт через тот же `parse_species`, что и при
индексации, поэтому «Медь» не превращается в мусор). Можно писать и
`Fe2O3`, и `\mathrm{Fe_2O_3}` — после нормализации это одна и та же
строка. Structured-поля найденной формулы/ячейки (`formula_latex`,
`equation_number`, `table_id`, `cell_value`, ...) передаются в промпт
LLM отдельной строкой `Точные данные:`, а не только через текст
парагрaфа — это уменьшает риск, что модель процитирует не тот фрагмент.

Для внешнего сервера передайте адрес и модель; ключ читается только из
переменной окружения `RAG_LLM_API_KEY`:

```bash
PYTHONPATH=src venv/bin/python -m pdfscan.rag.answer ask "ваш вопрос" \
  --provider openai-compatible --base-url http://localhost:8000/v1 \
  --model YOUR_MODEL
```

### MetalGPT-1 и отдельный сильный Judge

Для готового API без собственного GPU используйте Hugging Face Inference Providers:

```dotenv
RAG_LLM_PROVIDER=openai-compatible
RAG_LLM_MODEL=nn-tech/MetalGPT-1:featherless-ai
RAG_LLM_BASE_URL=https://router.huggingface.co/v1
RAG_LLM_API_KEY=hf_your_token
OPENROUTER_MODEL=qwen/qwen3.6-35b-a3b
OPENROUTER_API_KEY=your_openrouter_key
```

HF-токен должен иметь разрешение `Make calls to Inference Providers`.
API и UI при локальном запуске автоматически читают корневой `.env`;
переменные окружения процесса имеют приоритет. При наличии обоих
`OPENROUTER_*` Judge остаётся на указанной модели OpenRouter.
После изменения `.env` перезапустите API и UI (в Docker: `docker compose up -d --build api ui`).

Проверка конфигурации: `curl http://localhost:8000/health`.
В результате должны быть `model=nn-tech/MetalGPT-1:featherless-ai` и
`judge_model=qwen/qwen3.6-35b-a3b`. Для проверки токенов и реальных вызовов:

```bash
PYTHONPATH=src venv/bin/python scripts/check_llm_api.py
```

Проверка делает по одному короткому платному запросу к генератору и Judge,
не выводит токены. Полный RAG проверяется вопросом в UI; в ответе видны
модель Judge, оценка и источники.

`nn-tech/MetalGPT-1` уже поддерживается через существующий provider
`openai-compatible`: поднимите модель отдельным vLLM или SGLang сервером и
укажите его OpenAI-compatible URL. Модель основана на Qwen3-32B и опубликована
в BF16, поэтому её нельзя разумно добавлять в тот же Docker Compose, где живут
RAG API и UI: для исходной модели нужен отдельный GPU-сервер с большим объёмом
VRAM. На том же компьютере не используйте порт `8000`, он занят RAG API.
Например, сервер MetalGPT можно поднять на GPU-хосте через vLLM, как указано
в [карточке MetalGPT-1](https://huggingface.co/nn-tech/MetalGPT-1):

```bash
vllm serve nn-tech/MetalGPT-1 --port 8001
```

В `.env` RAG-сервера задайте:

```bash
RAG_LLM_PROVIDER=openai-compatible
RAG_LLM_MODEL=nn-tech/MetalGPT-1
RAG_LLM_BASE_URL=http://GPU_SERVER:8001/v1
```

После перезапуска `api` и `ui` этот профиль появится в интерфейсе как
`openai-compatible`; URL и модель также можно временно изменить прямо в
боковой панели. Ключ для локального vLLM обычно не нужен. Для облачного
провайдера добавьте `RAG_LLM_API_KEY` только в `.env`.

Judge настраивается независимо от модели, которая пишет ответ. Если не
задавать `RAG_JUDGE_*`, Judge использует тот же Qwen/MetalGPT. Чтобы ответы
оставались локальными, а проверка шла более сильной моделью, укажите в `.env`:

```bash
RAG_JUDGE_PROVIDER=openai-compatible
RAG_JUDGE_MODEL=~openai/gpt-latest
RAG_JUDGE_BASE_URL=https://openrouter.ai/api/v1
RAG_JUDGE_API_KEY=YOUR_LIMITED_OPENROUTER_KEY
RAG_JUDGE_TIMEOUT=240
RAG_JUDGE_THRESHOLD=0.82
RAG_JUDGE_MAX_ITERATIONS=3
```

OpenRouter использует совместимый с OpenAI endpoint `/api/v1/chat/completions`
и Bearer API key; вместо `~openai/gpt-latest` можно выбрать любой доступный
model slug из его каталога. Дайте ключу отдельный лимит расходов: Judge может
вызываться до `RAG_JUDGE_MAX_ITERATIONS` раз на один вопрос. Интеграция не
отправляет в Judge весь корпус: только вопрос, черновой ответ и отобранные
Evidence-фрагменты. [Документация OpenRouter](https://openrouter.ai/docs/quickstart).

## Интерфейс: FastAPI + Streamlit

FastAPI вызывает retrieval, Qdrant, Neo4j и LLM. Streamlit передаёт выбранные
методы поиска и показывает ответ, Evidence и происхождение каждого фрагмента.

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

Для сравнения методов используйте `pdfscan.rag.experiments`, описанный ниже.

## Retrieval contract

`pdfscan.rag.retrieval.retrieve()` возвращает `QueryProfile` и список
`Evidence`. Router выбирает dense channels по признакам запроса. Методы
`vector`, `bm25`, `graph` включаются независимо. Общий RRF объединяет
ранги выбранных каналов; затем дочерние объекты
группируются по `parent_chunk_id`. После этого multilingual cross-encoder
оценивает пару «вопрос — фрагмент». В генератор проходят только фрагменты с
`relevance_score >= RAG_RELEVANCE_THRESHOLD` (по умолчанию `0.55`).

Если релевантных источников нет, возвращается `no_evidence`; ответ только
LLM доступен по явному opt-in. При наличии evidence генерация и Judge
проверяют покрытие вопроса. Для простых единиц выражения «выше / ниже /
more than / less than» превращаются в Qdrant range-filter по `si_value`.

## Граф Neo4j и эксперименты по методам поиска

`kg_rag/dump/neo4j.dump` подключён к основному `docker-compose.yml`.
`neo4j-restore` загружает его в отдельный том `metallurgy_rag_neo4jdata`
только при отсутствии базы. Существующую базу служба не перезаписывает.
Процедура использует штатный [offline load Neo4j](https://neo4j.com/docs/operations-manual/current/docker/dump-load/).
Не запускайте одновременно отдельный Compose из `kg_rag`: у него те же host-порты.

```bash
# Полный стек с обновлёнными API и UI
docker compose up -d --build
# Только базы для запуска Python-команд на хосте
docker compose up -d qdrant neo4j
# Зависимости для команд на хосте
venv/bin/python -m pip install -e '.[rag,web]'
```

Neo4j Browser: http://localhost:7476, Bolt: `bolt://localhost:7690`.
По умолчанию логин `neo4j`, пароль `neo4jpass`; настройки в `.env.example`.
Compose читает `.env`, а при запуске Python на хосте переменные `NEO4J_*`
нужно экспортировать в окружение. Смена пароля в `.env` не меняет пароль уже
инициализированной базы. Дамп монтируется read-only и не попадает в Docker-образ.
При неудачном восстановлении проверьте логи `docker compose logs neo4j-restore`:
автоматического удаления/перезаписи частично созданной базы нет.

Три независимо включаемых метода:

- `vector`: специализированные dense-каналы text / chemistry / math / table / unit,
  выбранные существующим router.
- `bm25`: лексический sparse-индекс Qdrant, в UI — «Индекс (BM25)».
- `graph`: поиск терминов через Alias и fulltext `term_search`, обход REL,
  извлечение Statement → Sentence → Document из Neo4j.

В UI выберите любое непустое сочетание в «Методы поиска».
По умолчанию оставлены `vector + bm25`. API принимает, например,
`{"question":"Что такое автогенный процесс?","methods":["graph"]}`.
Ошибка выбранного сервиса возвращается как ошибка запроса; система не подменяет
эксперимент незаметным отключением метода.

Графовый канал использует лексические seed-термины, но **не** dense-векторы
`term_vec` из исходного `kg_rag`. Это позволяет испытывать граф отдельно.
Все выбранные ранжированные списки объединяются одним RRF (`1/(60+rank)`),
затем выполняются parent-child dedup, ограничение общего пула кандидатов
(по умолчанию 40, не менее `4*k`), общий reranker и порог релевантности.
Cross-encoder reranker остаётся общим для всех сочетаний, в том числе «только граф»
и «только BM25»: выбор методов управляет поиском кандидатов, не reranker.
При недоступности cross-encoder профиль явно указывает `lexical-fallback`.

Графовые evidence имеют ID `kg:<statement uid>`, страницу, документ и
`structured.graph` с путём и связями. ID графа и Qdrant различны, поэтому
между ними нет гарантированной дедупликации одинакового текста.
Числовые payload-фильтры router применяются к Qdrant; граф возвращает факты
для общей проверки reranker, без строгого числового range-фильтра.

```bash
# Один метод или любое сочетание; работает и для существующей команды batch
PYTHONPATH=src venv/bin/python -m pdfscan.rag.answer ask \
  'Что такое автогенный процесс?' --methods graph
PYTHONPATH=src venv/bin/python -m pdfscan.rag.answer batch \
  --methods vector graph --limit 10 --out reports/llm_runs/vector_graph.jsonl

# Все 7 сочетаний на одинаковых вопросах экспертной таблицы, без LLM и Judge
PYTHONPATH=src venv/bin/python -m pdfscan.rag.experiments \
  --all-combinations --retrieval-only --limit 10 \
  --out reports/experiments/retrieval_7_modes.jsonl

# Полный прогон с генерацией и Judge: убрать --retrieval-only
PYTHONPATH=src venv/bin/python -m pdfscan.rag.experiments \
  --all-combinations --limit 10 --model qwen3:8b \
  --out reports/experiments/answers_7_modes.jsonl

# Для своего списка: UTF-8 текст, один вопрос на строку
PYTHONPATH=src venv/bin/python -m pdfscan.rag.experiments \
  --questions reports/experiments/smoke_questions.txt \
  --methods bm25 graph --retrieval-only \
  --out reports/experiments/bm25_graph.jsonl
```

Runner сохраняет каждую попытку сразу в JSONL и рядом пишет `.summary.json`:
число ошибок, долю запросов с evidence, среднее число кандидатов/принятых
фрагментов, время и Judge coverage (только для полного прогона).
Существующий JSONL не перезаписывается. Время включает прогрев моделей;
первый вариант нельзя напрямую сравнивать с остальными как чистую latency.
Доля найденных evidence и оценки reranker/Judge — диагностика, **не**
precision/recall: для них нужна разметка релевантных источников.

Для ограниченной памяти reranker использует `RAG_RERANKER_MAX_LENGTH=1024`
токена на пару «вопрос + фрагмент» и `RAG_RERANKER_BATCH_SIZE=4`.
Длинный вход обрезается моделью; эти настройки одинаковы во всех режимах
и сохраняются в `profile.reranker_settings`. При необходимости задайте
`RAG_RERANKER_DEVICE=cpu` вместо автоматического выбора устройства.
Векторную модель также можно закрепить на CPU: `RAG_EMBEDDING_DEVICE=cpu`.
