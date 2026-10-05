# Перенос на другой ПК через внешний диск

Текущий вариант: MetalGPT через Hugging Face, Judge Qwen через OpenRouter.
Для запуска нужны Docker Desktop / Docker Engine с Compose и интернет.
Python и Ollama на новом ПК устанавливать не нужно.

## 1. Подготовить исходные данные

Остановите приложение и Qdrant перед копированием его хранилища:

```bash
cd /Users/andrejsuhanov/Applications/MISIS/metallurgy_rag
docker compose down
```

Если Qdrant запущен другим Compose-проектом или отдельно, остановите именно
тот процесс/контейнер, который использует `data/qdrant`.
Не используйте `down -v`: это удаляет постоянные тома.

В комплект входит `kg_rag/dump/neo4j.dump` (сейчас файл от 16 сентября 2026).
На новом ПК граф будет восстановлен именно из него. Если после этой даты
вы меняли граф в рабочей Neo4j, сначала сделайте свежий dump остановленной
базы командой `neo4j-admin database dump neo4j` в её окружении и замените
`kg_rag/dump/neo4j.dump`. Простой перенос папки проекта не переносит
Docker volume рабочей Neo4j.

## 2. Скопировать комплект на диск с Mac

Ниже замените `YOUR_DISK` настоящим именем внешнего диска. Для переноса между
Mac и Windows удобно использовать exFAT; FAT32 не поддерживает файлы больше
4 ГБ. Понадобится минимум 15–20 ГБ свободного места для этого комплекта.

```bash
cd /Users/andrejsuhanov/Applications/MISIS/metallurgy_rag
transfer_dir="/Volumes/YOUR_DISK/metallurgy_rag"
mkdir -p "$transfer_dir/data" "$transfer_dir/kg_rag/dump"
rsync -a src docker scripts docs tests requirements Dockerfile docker-compose.yml pyproject.toml README.md .dockerignore .env.example .env "$transfer_dir/"
rsync -a data/qdrant data/qdrant_vocab.json "$transfer_dir/data/"
rsync -a kg_rag/dump/neo4j.dump "$transfer_dir/kg_rag/dump/"
```

Скопированный `.env` содержит реальные ключи API. На новом ПК не заменяйте
его примером `.env.example`.

В `.env` комплекта задайте переносимый относительный путь:

```dotenv
QDRANT_STORAGE_PATH=./data/qdrant
```

Для первого запуска достаточно перечисленных файлов. При желании добавьте
`corpus_split/` для будущей переиндексации, `pdf/` и результаты разбора для
дальнейшей работы с исходными документами. Виртуальные окружения `venv/`,
`venv-formula/`, `mineru-venv/` переносить не нужно: они зависят от ОС.

## 3. Запустить на новом ПК

Установите Docker. На Windows используйте Docker Desktop с Linux containers
и WSL2. Скопируйте весь комплект с диска в локальный каталог, например
`C:\Projects\metallurgy_rag`. В PowerShell откройте этот каталог:

```powershell
cd C:\Projects\metallurgy_rag
docker compose up -d --build
```

На Linux/macOS та же команда выполняется в терминале из каталога проекта.
Docker скачает базовые образы и соберёт приложение под архитектуру нового
компьютера. Neo4j восстановит граф из dump, Qdrant откроет перенесённый индекс.
Embedding-модель и reranker будут скачаны при первом поиске; он может занять
больше времени. Сам индекс пересобирать не нужно.

Порты 8000, 8501, 6333, 6334, 7476 и 7690 должны быть свободны.

## 4. Проверить

```bash
docker compose ps
```

На Windows PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:6333/collections/metallurgy_search_units
```

На Linux/macOS:

```bash
curl http://localhost:8000/health
curl http://localhost:6333/collections/metallurgy_search_units
```

В `/health` ожидаются:

- `model`: `nn-tech/MetalGPT-1:featherless-ai`;
- `judge_model`: `qwen/qwen3.6-35b-a3b`.

Откройте http://localhost:8501 и задайте вопрос по корпусу. Проверьте сначала
vector + bm25, затем отдельно graph, затем все три метода. В ответе видны
источники, каналы поиска и оценка Judge. При ошибке:

```bash
docker compose logs --tail=100 api neo4j qdrant
```

## Нужны ли сами контейнеры на внешнем диске?

Для описанного переноса достаточно Dockerfile и Compose: новый компьютер
создаст контейнеры сам. Работающие контейнеры не являются резервной копией
баз. Базы и словарь переносятся отдельно, как указано выше.

`docker save` переносит образы, но ARM64-образы с Mac не подходят для
нативного запуска на AMD64 ПК. При необходимости переноса без скачивания
образов требуется отдельно подготовить образы под архитектуру целевого ПК.
Для вызовов MetalGPT и Judge интернет всё равно необходим.
