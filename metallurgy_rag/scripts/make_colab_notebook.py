"""Собирает блокнот для Google Colab.

Блокнот пишется скриптом, а не руками: так его легко пересобрать, когда
поменяется состав шагов, и не приходится править JSON вручную.
"""

import json
from pathlib import Path

CELLS = [
    ('md', """# Smart PDF Extractor в Google Colab

Разбор библиотеки PDF и сборка поискового индекса для RAG.

**Про выбор машины.** Видеокарта ускоряет только два шага: модель разметки
макета и векторизацию кусков. Само распознавание текста делает tesseract на
процессоре, и на сканах именно оно съедает почти всё время. Поэтому A100 здесь
почти не отличается от L4 или T4, а вот число ядер процессора и оперативная
память важны. Если считаете вычислительные единицы — берите L4 с режимом
«Высокий объём ОЗУ», а A100 приберегите для задач, где она действительно нужна.

**Порядок работы:** «Среда выполнения → Сменить среду выполнения → GPU», затем
ячейки сверху вниз."""),

    ('md', '## 1. Проверка машины'),
    ('code', """import subprocess
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                      '--format=csv,noheader'],
                     capture_output=True, text=True).stdout or 'Видеокарта не подключена')

import torch
print('torch:', torch.__version__, '| CUDA доступна:', torch.cuda.is_available())"""),

    ('md', """## 2. Системные библиотеки

`poppler` переводит страницы в изображения, `tesseract` распознаёт текст.
Языковые пакеты ставятся отдельно — без `rus` русские сканы читаться не будут."""),
    ('code', """%%bash
apt-get -qq update
apt-get -qq install -y poppler-utils tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng
tesseract --list-langs"""),

    ('md', '## 3. Python-зависимости'),
    ('code', """%pip install -q "unstructured[pdf]" pdfplumber pypdfium2 transformers faiss-cpu \\
    rank_bm25 scikit-learn joblib openpyxl python-dotenv nltk pyarrow duckdb"""),

    ('md', """## 4. Файлы проекта

На Диске нужны две вещи: код проекта и библиотека PDF. Ожидаемая раскладка:

```
MyDrive/pdf-scan/
├── src/pdfscan/          ← код пакета
├── models/               ← formula_classifier.pkl
├── pyproject.toml
├── .env                  ← ключ OpenRouter, если нужен этап инспекции
├── pdf/                  ← сюда PDF, можно во вложенных папках
│   ├── Книги по цветной металлургии/
│   ├── Литература по медным шлакам/
│   └── Сборники по Cu, Co и Ni/
└── out/                  ← создастся сам
```

Код копируется на локальный диск машины: с Диска Python импортируется заметно
медленнее. А вот исходники и результаты остаются на Диске, потому что сеанс
Colab не бесконечен, и всё, что лежит на локальном диске, исчезает вместе с ним.

Куда смотреть за данными, пакет узнаёт из переменных окружения. Так один и тот
же код работает и локально, и здесь, где корень проекта и библиотека документов
лежат в разных местах."""),
    ('code', """from google.colab import drive
drive.mount('/content/drive')

ROOT = '/content/drive/MyDrive/pdf-scan'   # правьте под свой путь

import glob, os, shutil, subprocess
WORK = '/content/work'
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK, exist_ok=True)

shutil.copytree(f'{ROOT}/src', f'{WORK}/src')
shutil.copy(f'{ROOT}/pyproject.toml', WORK)
for optional in ('.env',):
    if os.path.exists(f'{ROOT}/{optional}'):
        shutil.copy(f'{ROOT}/{optional}', WORK)
os.chdir(WORK)

# Пакет ставится ссылкой и без зависимостей: они уже стоят из ячейки 3, и
# повторная установка только сломала бы подобранные версии.
subprocess.run(['pip', 'install', '-q', '-e', '.', '--no-deps'], check=True)

# Данные и модель остаются на Диске, поэтому обрыв сеанса их не уносит.
os.environ['PDFSCAN_ROOT'] = ROOT
os.environ['PDFSCAN_PDF_DIR'] = f'{ROOT}/pdf'
os.environ['PDFSCAN_OUT_DIR'] = f'{ROOT}/out'
os.environ['PDFSCAN_MODELS_DIR'] = f'{ROOT}/models'
os.environ['PDFSCAN_DATA_DIR'] = f'{ROOT}/data'
os.environ['PDFSCAN_REPORTS_DIR'] = f'{ROOT}/reports'
os.environ['PDFSCAN_INDEX_DIR'] = f'{ROOT}/rag_index'

from pdfscan import paths
print('модель формул:', paths.FORMULA_CLASSIFIER.exists())
print('ключ OpenRouter:', 'есть' if os.path.exists('.env') else 'нет — инспекция не запустится')
print('документов найдено:', len(glob.glob(f'{paths.PDF_DIR}/**/*.pdf', recursive=True)))"""),

    ('md', """## 5. Разбор библиотеки

Папка обходится рекурсивно, и у каждого документа появляется своя папка с
результатами, повторяющая исходное дерево:

```
out/Книги по цветной металлургии/Название книги/
├── blocks.jsonl     ← сырьё для индекса
├── blocks.xlsx      ← та же разметка для просмотра глазами
└── pages/page1.png  ← страницы с рамками блоков
```

Результат пишется сразу на Диск, а `--skip-existing` пропускает уже разобранное,
поэтому обрыв сеанса не сжигает работу: выполните ячейки 1–4 и запустите разбор
заново.

`--no-images` заметно ускоряет работу и экономит место — отрисовка страниц нужна
только для проверки глазами, на индекс она не влияет. На первых двух-трёх
документах её лучше оставить, дальше отключить.

`--no-inspect` выключает этап 4: он обращается к OpenRouter на каждой странице,
и на библиотеке из сотен документов это тысячи сетевых запросов. Разберите всё
без него, а инспекцию запускайте потом точечно."""),
    ('code', """!python -m pdfscan.parse.extractor "$PDFSCAN_PDF_DIR" \\
    --out "$PDFSCAN_OUT_DIR" --skip-existing --no-images --no-inspect"""),

    ('md', """## 6. Очистка корпуса

Разбор даёт блоки как есть, вместе с шумом распознавания, колонтитулами и
обрывками строк. Стадия очистки сводит их к тексту, пригодному для чанкинга, и
пишет отчёт с выборками — по нему видно, что именно отброшено и не пора ли
сдвинуть порог.

Ничего не удаляется: у отброшенных блоков стоит `keep=false` и причина, поэтому
порог меняется без повторного разбора PDF."""),
    ('code', """!python -m pdfscan.prepare.cli all
print(open(f\"{os.environ['PDFSCAN_REPORTS_DIR']}/clean_audit.md\").read()[:3000])"""),

    ('md', '## 7. Сборка индекса'),
    ('code', """!python -m pdfscan.rag.build build

print('индекс:', os.environ['PDFSCAN_INDEX_DIR'])"""),

    ('md', '## 8. Поиск'),
    ('code', """from pdfscan.rag.index import RagIndex

index = RagIndex.load(os.environ['PDFSCAN_INDEX_DIR'])

for chunk in index.search('восстановление оксида железа', k=3):
    print(f\"[{chunk['doc_id']}, с. {chunk['pages']}] оценка {chunk['score']}\")
    print(chunk['text'][:300], '\\n')"""),

    ('md', """## 9. Переобучение классификатора (при необходимости)

Нужно только если вы поправили обучающие примеры."""),
    ('code', '!python -m pdfscan.formulas.train'),

    ('md', """## Повторное распознавание формул

Этого шага здесь нет намеренно. Модель формул работает на paddle, а paddle
конфликтует с тем стеком, который стоит в ячейке 3, поэтому шаг требует
отдельного окружения — в Colab это означает отдельный блокнот с чистой машиной.
Порядок описан в `requirements/formula-ocr.txt`."""),

    ('md', """## Если сеанс обрывается

Colab отключает машину после долгого простоя и в любом случае не держит её
бесконечно. Разбор в ячейке 5 продолжается с места остановки — документы, у
которых уже есть `blocks.jsonl`, пропускаются. Достаточно заново выполнить
ячейки 1–4 и запустить разбор.

Сколько разобрано, видно так:"""),
    ('code', """import glob
done = glob.glob(f'{OUT}/**/blocks.jsonl', recursive=True)
total = glob.glob(f'{PDF_DIR}/**/*.pdf', recursive=True)
print(f'разобрано {len(done)} из {len(total)}')
for path in sorted(set(p.rsplit("/", 1)[0] for p in done))[:20]:
    print('  ', path.replace(OUT + '/', ''))"""),
]


def build(path=None):
    path = path or Path(__file__).resolve().parents[1] / 'notebooks' / 'colab_pipeline.ipynb'
    cells = []
    for kind, source in CELLS:
        lines = source.split('\n')
        payload = [line + '\n' for line in lines[:-1]] + [lines[-1]]
        if kind == 'md':
            cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': payload})
        else:
            cells.append({'cell_type': 'code', 'metadata': {}, 'source': payload,
                          'execution_count': None, 'outputs': []})

    notebook = {
        'cells': cells,
        'metadata': {
            'accelerator': 'GPU',
            'colab': {'provenance': [], 'toc_visible': True},
            'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
            'language_info': {'name': 'python'},
        },
        'nbformat': 4,
        'nbformat_minor': 0,
    }

    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(notebook, handle, ensure_ascii=False, indent=1)
    print(f'💾 {path} — ячеек {len(cells)}')


if __name__ == '__main__':
    build()
