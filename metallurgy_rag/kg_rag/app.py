from __future__ import annotations

import html
import os
import re
import sys
import time

import streamlit as st

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "rag")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
import retrieval
from answer import ask_llm, cited_numbers, default_model

st.set_page_config(page_title="Граф знаний: металлургия", layout="wide")


@st.cache_resource
def get_driver():
    return retrieval.get_driver()


def load_state():
    with get_driver().session(database=config.NEO4J_DATABASE) as session:
        row = session.run("MATCH (s:LoadState {id:'load'}) "
                          "RETURN s.running AS running").single()
        aliases = session.run("MATCH (a:Alias) RETURN count(a) AS c").single()["c"]
    return {"running": bool(row and row["running"]), "aliases": aliases}


def graph_stats():
    with get_driver().session(database=config.NEO4J_DATABASE) as session:
        row = session.run(
            "MATCH (t:Term) WITH count(t) AS terms "
            "MATCH (s:Statement) WITH terms, count(s) AS statements "
            "MATCH (d:Document) RETURN terms, statements, count(d) AS documents"
        ).single()
    return dict(row) if row else {}


def run_question(question, settings):
    started = time.time()
    with get_driver().session(database=config.NEO4J_DATABASE) as session:
        paths, seeds, forms = retrieval.search(
            session, question, settings["depth"], settings["paths"],
            settings["min_count"], settings["with_words"],
        )
        sources = retrieval.sources_for(session, paths, settings["sources"])
    context = retrieval.build_context(sources)
    result = {"question": question, "paths": paths, "seeds": seeds, "forms": forms,
              "sources": sources, "used": [], "context": context,
              "answer": None, "model": None, "error": None,
              "search_sec": time.time() - started}

    if not sources:
        result["answer"] = "В графе не нашлось подходящих источников для этого вопроса."
        return result
    if settings["no_llm"]:
        result["answer"] = "_Модель не вызывалась — ниже все найденные предложения._"
        return result

    started = time.time()
    try:
        result["answer"], result["model"] = ask_llm(question, context, settings["model"],
                                                    return_model=True)
        result["used"] = cited_numbers(result["answer"], sources)
    except SystemExit as exc:
        result["error"] = str(exc)
    except Exception as exc:
        result["error"] = "Модель не ответила: %s" % exc
    result["llm_sec"] = time.time() - started
    return result


KIND_NAMES = {"term": "термин", "chem": "вещество", "word": "слово", "formula": "формула",
              "reaction": "реакция", "quantity": "величина"}

CSS = """
<style>
.src { border-left: 3px solid #c2410c; padding: 4px 0 4px 12px; margin: 10px 0 14px; }
.src .head { font-weight: 600; }
.src .meta { opacity: .65; font-size: .85em; }
.src .quote { margin-top: 4px; line-height: 1.45; opacity: .85; }
.src .fact { font-size: .95em; margin-top: 4px; }
.src .ref-src { margin-top: 8px; font-size: .85em; opacity: .8; }
.src.dim { border-left-color: rgba(128,128,128,.45); }
.ref { font-size: .8em; font-weight: 600; color: #c2410c; }
.src .tcap { font-size: .82em; opacity: .7; margin-top: 8px; }
.src .table { font-family: ui-monospace, Consolas, monospace; font-size: .8em;
              margin: 4px 0 0; padding: 8px; border-radius: 4px; overflow-x: auto;
              background: rgba(128,128,128,.10); white-space: pre; }
.src .formula { font-family: ui-monospace, Consolas, monospace; font-size: .92em;
                margin-top: 6px; padding: 4px 8px; border-radius: 4px;
                background: rgba(194,65,12,.08); display: inline-block; }
</style>
"""


def render_answer_text(text):
    safe = html.escape((text or "").strip())
    safe = re.sub(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]", r'<span class="ref">[\1]</span>', safe)
    return safe.replace("\n", "<br>")


def source_card(item, dim=False):
    facts = "".join('<div class="fact">%s</div>' % html.escape(f["line"])
                    for f in item.get("facts") or [])
    formulas = "".join('<div class="formula">%s</div>' % html.escape(f)
                       for f in (item.get("formulas") or [])[:3])
    table = ""
    if item.get("table"):
        table = ('<div class="tcap">%s</div><pre class="table">%s</pre>'
                 % (html.escape(item.get("table_caption") or ""),
                    html.escape(item["table"][:2500])))
    refs = "".join(
        '<div class="ref-src"><b>%s</b>, стр. %s<div class="quote">%s</div></div>'
        % (html.escape(src["book"]), src["page"], html.escape(src["sentence"][:400]))
        for src in (item.get("sources") or [])[:3])
    sim = item.get("sim")
    meta = "оценка %.1f" % item.get("score", 0)
    if sim is not None:
        meta += " · близость %.2f" % sim
    return ('<div class="src%s"><div class="head">[%d] %s</div>'
            '<div class="meta">%s</div>%s%s%s%s</div>'
            % (" dim" if dim else "", item["n"], html.escape(item["path"]),
               meta, facts, formulas, table, refs))


def show_result(result):
    if result["error"]:
        st.error(result["error"])
    else:
        st.markdown(render_answer_text(result["answer"]), unsafe_allow_html=True)
    if result.get("model"):
        st.caption("модель: %s" % result["model"])

    sources, used = result.get("sources") or [], result.get("used") or []
    cited = [s for s in sources if s["n"] in used]
    others = [s for s in sources if s["n"] not in used]

    if cited:
        st.markdown("**Пути, на которых построен ответ**")
        st.markdown("".join(source_card(s) for s in cited), unsafe_allow_html=True)
    elif result.get("answer") and not result.get("error") and result.get("model"):
        st.info("Модель не сослалась ни на один путь — ответ стоит проверить "
                "по списку ниже.")

    label = ("Остальные пути, отправленные модели (%d)" % len(others) if cited
             else "Пути, отправленные модели (%d)" % len(sources))
    with st.expander(label, expanded=bool(sources) and not result.get("model")):
        if others:
            st.markdown("".join(source_card(s, dim=True) for s in others),
                        unsafe_allow_html=True)
        else:
            st.caption("других путей не было")

    with st.expander("Как искали: вершины и пути в графе"):
        st.markdown("**Слова вопроса:** " + (", ".join(result["forms"]) or "—"))
        if result["seeds"]:
            st.markdown("**Вершины-входы:** " + " · ".join(
                "`%s` _(%s)_" % (s["label"], KIND_NAMES.get(s["kind"], s["kind"]))
                for s in result["seeds"]))
        tab_paths, tab_raw = st.tabs(["Пути (по убыванию оценки)", "Текст, ушедший в модель"])
        with tab_paths:
            for number, path in enumerate(result["paths"], start=1):
                st.text("%2d. %.1f  %s" % (number, path.get("score", 0),
                                          retrieval.path_line(path)))
        with tab_raw:
            st.code(result["context"] or "(пусто)", language=None)
        timing = "поиск %.1f с" % result["search_sec"]
        if result.get("llm_sec"):
            timing += " · модель %.1f с" % result["llm_sec"]
        st.caption(timing)


st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.header("Настройки")
    model = st.text_input("Модель (OpenRouter)", value=default_model(), key="model")
    sources_n = st.slider("Путей в модель", 5, 30, 15, key="sources",
                          help="топ путей графа после ранжирования")
    depth = st.slider("Шагов по графу", 1, 3, 2, key="depth",
                      help="1 — только соседи найденных вершин; 2–3 — пути через посредников")
    paths_n = st.slider("Путей-кандидатов", 5, 40, 20, key="paths",
                        help="из скольких лучших путей набираются предложения")
    min_count = st.slider("Минимум повторов ребра", 1, 5, 1, key="min_count",
                          help="2 и выше отсекает связи, встретившиеся один раз")
    with_words = st.checkbox("Выходить на слова вне словаря", value=True, key="with_words")
    no_llm = st.checkbox("Только поиск, без модели", value=False, key="no_llm")

    st.divider()
    try:
        state = load_state()
        if state["running"] or not state["aliases"]:
            st.warning("База перезагружается — поиск пока находит мало или ничего. "
                       "Дождитесь конца заливки.")
        stats = graph_stats()
        st.success("Neo4j: %s" % config.NEO4J_URI)
        st.caption("вершин %s · утверждений %s · документов %s"
                   % (stats.get("terms"), stats.get("statements"), stats.get("documents")))
    except Exception as exc:
        st.error("Нет связи с Neo4j (%s). Запустите `docker compose up -d`." % config.NEO4J_URI)
        st.caption(str(exc)[:200])

    if st.button("Очистить чат", key="clear"):
        st.session_state.history = []
        st.rerun()

settings = {"model": model.strip() or None, "sources": sources_n, "depth": depth,
            "paths": paths_n, "min_count": min_count, "with_words": with_words,
            "no_llm": no_llm}

st.title("Граф знаний: металлургия")
st.caption("Ответ строится только по литературе из графа; номера в ответе ведут "
           "к предложениям-источникам под ним.")

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant"):
        show_result(item)

question = st.chat_input("Например: чем снижают потери меди со шлаком?")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Ищу в графе и спрашиваю модель…"):
            try:
                result = run_question(question, settings)
            except Exception as exc:
                result = {"question": question, "paths": [], "seeds": [], "forms": [],
                          "sources": [], "used": [], "context": "", "answer": None,
                          "model": None, "error": "Поиск упал: %s" % exc, "search_sec": 0}
        show_result(result)
    st.session_state.history.append(result)
