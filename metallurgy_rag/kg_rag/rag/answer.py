from __future__ import annotations
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
import retrieval
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(os.path.join(_ROOT, ".env"))


SYSTEM_PROMPT = """Ты отвечаешь на вопросы по пирометаллургии (медь, никель, кобальт, шлаки).
Ниже даны пронумерованные пути графа знаний [1], [2], ..., построенного по научной
литературе. У каждого пути показаны его утверждения в виде «субъект | предикат | объект»,
где в скобках стоят уточнения из фрейма («шлак (конвертерный)», «при 1250 °C»), а также
формулы и таблицы, относящиеся к этим утверждениям. Пути отсортированы: выше -- ближе к вопросу.

Правила:
- Отвечай только по этим путям и их фреймам, без общих знаний. Коротко: сначала сам ответ.
- После каждого утверждения ставь номер пути, на котором оно основано: [3] или [2][5].
  Ссылайся только на те пути, которые действительно использовал.
- Английские и русские пути равноправны: переводи и используй наравне.
- Числа, температуры, формулы и составы переписывай без изменений.
- Список источников в конце НЕ пиши -- его покажет интерфейс по номерам.
- Если ответа в путях нет, скажи прямо: «в графе этого нет».
"""


DEFAULT_MODEL = "qwen/qwen3.6-35b-a3b"


def default_model():

    return os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL


def ask_llm(question, context, model=None, return_model=False):
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise SystemExit("нет OPENROUTER_API_KEY в .env")
    model = model or default_model()

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Вопрос: %s\n\nКонтекст:\n%s" % (question, context)},
        ],
        temperature=0.1,
    )
    text = response.choices[0].message.content
    if return_model:
        return text, getattr(response, "model", None) or model
    return text


def cited_numbers(answer_text, sources):
    valid = {src["n"] for src in sources}
    found = []
    for group in re.findall(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]", answer_text or ""):
        for number in re.split(r"\s*[,;]\s*", group):
            n = int(number)
            if n in valid and n not in found:
                found.append(n)
    return found
