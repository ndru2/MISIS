from __future__ import annotations

import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from lemma_normalize import query_terms

SEED_LIMIT = 12
PATH_LIMIT = 40
STATEMENTS_PER_REL = 3
STATEMENTS_FETCH = 8


def get_driver(uri=None, user=None, password=None):
    from neo4j import GraphDatabase

    return GraphDatabase.driver(uri or config.NEO4J_URI,
                                auth=(user or config.NEO4J_USER,
                                      password or config.NEO4J_PASSWORD))


def question_vector(question):
    try:
        import embed

        return embed.encode([question])[0]
    except Exception as exc:
        print("векторы недоступны: %s" % exc, file=sys.stderr)
        return None


CYPHER_SEED_EXACT = """
UNWIND $forms AS form
MATCH (a:Alias {form: form})-[:MEANS]->(t:Term)
RETURN t.id AS id, t.label AS label, t.kind AS kind, t.count AS count,
       t.formulas AS formulas, t.aliases AS aliases,
       t.label_ru AS label_ru, t.label_en AS label_en, 2.0 AS score
"""

CYPHER_SEED_FULLTEXT = """
CALL db.index.fulltext.queryNodes('term_search', $q) YIELD node, score
RETURN node.id AS id, node.label AS label, node.kind AS kind, node.count AS count,
       node.formulas AS formulas, node.aliases AS aliases,
       node.label_ru AS label_ru, node.label_en AS label_en, score AS score
LIMIT $limit
"""

CYPHER_SEED_VECTOR = """
CALL db.index.vector.queryNodes('term_vec', $limit, $vector) YIELD node, score
RETURN node.id AS id, node.label AS label, node.kind AS kind, node.count AS count,
       node.formulas AS formulas, node.aliases AS aliases,
       node.label_ru AS label_ru, node.label_en AS label_en, score AS score
"""


def find_seeds(session, question, limit=SEED_LIMIT, qvec=None):
    forms = query_terms(question)
    found = {}
    for row in session.run(CYPHER_SEED_EXACT, forms=forms):
        found[row["id"]] = dict(row)
    if forms:
        lucene = " OR ".join('"%s"' % f.replace('"', "") for f in forms)
        try:
            for row in session.run(CYPHER_SEED_FULLTEXT, q=lucene, limit=limit * 2):
                if row["kind"] not in ("word", "table"):
                    found.setdefault(row["id"], dict(row))
        except Exception as exc:
            print("полнотекстовый индекс недоступен: %s" % exc, file=sys.stderr)
    if qvec is not None:
        try:
            for row in session.run(CYPHER_SEED_VECTOR, limit=limit, vector=qvec):
                if row["kind"] != "table":
                    found.setdefault(row["id"], dict(row))
        except Exception as exc:
            print("векторный индекс вершин недоступен: %s" % exc, file=sys.stderr)
    seeds = sorted(found.values(), key=lambda r: (-(r["score"] or 0), -(r["count"] or 0)))
    return seeds[:limit], forms


CYPHER_SEED_PAIRS = """
MATCH p = (a:Term)-[rels:REL*1..%(depth)d]-(b:Term)
WHERE a.id IN $ids AND b.id IN $ids AND a.id < b.id
  AND all(r IN rels WHERE r.count >= $min_count)
  AND all(n IN nodes(p)[1..-1] WHERE $with_words OR n.kind <> 'word')
WITH p, rels, reduce(total = 0, r IN rels | total + r.count) AS weight
ORDER BY size(rels) ASC, weight DESC
LIMIT $limit
RETURN [n IN nodes(p) | {id: n.id, label: n.label, kind: n.kind}] AS nodes,
       [r IN rels | {predicate: r.predicate, count: r.count, origin: r.origin,
                     from: startNode(r).id, to: endNode(r).id}] AS rels
"""

CYPHER_NEIGHBOURS = """
UNWIND $ids AS seed
CALL (seed) {
  MATCH (s:Term {id: seed})-[r:REL]-(e:Term)
  WHERE r.count >= $min_count AND ($with_words OR e.kind <> 'word')
  RETURN s, r, e
  ORDER BY CASE WHEN e.kind = 'word' THEN 1 ELSE 0 END ASC, r.count DESC
  LIMIT $per_seed
}
RETURN [{id: s.id, label: s.label, kind: s.kind}, {id: e.id, label: e.label, kind: e.kind}] AS nodes,
       [{predicate: r.predicate, count: r.count, origin: r.origin,
         from: startNode(r).id, to: endNode(r).id}] AS rels
"""


def expand(session, seed_ids, depth=1, limit=PATH_LIMIT, min_count=1, with_words=True):
    params = {"ids": seed_ids, "min_count": min_count, "with_words": with_words}
    paths = []
    if len(seed_ids) >= 2:
        cypher = CYPHER_SEED_PAIRS % {"depth": max(1, depth)}
        for row in session.run(cypher, limit=limit * 2, **params):
            paths.append({"nodes": row["nodes"], "rels": row["rels"], "between_seeds": True})
    per_seed = max(3, (limit * 2) // max(1, len(seed_ids)))
    for row in session.run(CYPHER_NEIGHBOURS, per_seed=per_seed, **params):
        paths.append({"nodes": row["nodes"], "rels": row["rels"], "between_seeds": False})
    return paths


CYPHER_STATEMENTS = """
UNWIND $keys AS key
CALL (key) {
  MATCH (st:Statement {predicate: key.predicate})
  MATCH (st)-[:SUBJECT]->(:Term {id: key.from})
  MATCH (st)-[:OBJECT]->(:Term {id: key.to})
  RETURN st LIMIT $per_rel_fetch
}
MATCH (st)-[:SAID_IN]->(sent:Sentence)
OPTIONAL MATCH (sent)-[:FROM]->(doc:Document)
RETURN key.from AS from, key.to AS to, key.predicate AS predicate,
       st.uid AS uid, st.lang AS lang,
       st.subject_text AS subject_text, st.object_text AS object_text,
       st.subject_frame AS subject_frame, st.object_frame AS object_frame,
       st.predicate_frame AS predicate_frame,
       st.table_text AS table_text, st.table_caption AS table_caption,
       st.subject_inherited AS subject_inherited,
       sent.uid AS sentence_uid, sent.text AS sentence, sent.page AS page,
       sent.doc_id AS doc_id, doc.title AS doc_title,
       sent.formulas AS formulas, sent.seq AS seq
"""


def render_slot(raw, depth=0, for_vector=False):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            return raw
    if not isinstance(raw, dict) or depth > 4:
        return ""
    head = (raw.get("text") or "").strip()
    mask = raw.get("mask") or {}
    norm = mask.get("norm") or {}
    kind = mask.get("kind")
    if kind in ("formula", "table"):
        if for_vector:
            return ""
        head = (norm.get("canonical") or norm.get("latex_clean") or head
                if kind == "formula"
                else "%s [%s]" % (head, mask.get("label") or "таблица"))
    elif kind == "quantity":
        head = norm.get("canonical") or mask.get("canonical") or head
    children = [render_slot(child, depth + 1, for_vector) for child in raw.get("frame") or []]
    children = [c for c in children if c]
    if children:
        return "%s (%s)" % (head, " ".join(children)) if head else " ".join(children)
    return head


def vector_text(fact):
    parts = [fact["vector_frames"].get(role) for role in ("subject", "predicate", "object")]
    return " ".join(p for p in parts if p)


def add_similarity(facts, qvec):
    if not qvec or not facts:
        return
    import embed

    texts, index = [], {}
    for fact in facts:
        text = vector_text(fact)
        if not text:
            continue
        if text not in index:
            index[text] = len(texts)
            texts.append(text)
    if not texts:
        return
    vectors = embed.encode(texts)
    for fact in facts:
        text = vector_text(fact)
        position = index.get(text)
        if position is None:
            continue
        vector = vectors[position]
        fact["sim"] = sum(a * b for a, b in zip(vector, qvec))


def statements_for(session, paths, qvec=None, per_rel=STATEMENTS_PER_REL):
    keys, seen = [], set()
    for path in paths:
        for rel in path["rels"]:
            key = (rel["from"], rel["predicate"], rel["to"])
            if key in seen:
                continue
            seen.add(key)
            keys.append({"from": rel["from"], "predicate": rel["predicate"], "to": rel["to"]})
    if not keys:
        return {}
    by_rel, all_facts = {}, []
    rows = session.run(CYPHER_STATEMENTS, keys=keys,
                       per_rel_fetch=max(per_rel * 2, STATEMENTS_FETCH))
    for row in rows:
        fact = dict(row)
        subject_raw = fact.pop("subject_frame", None)
        predicate_raw = fact.pop("predicate_frame", None)
        object_raw = fact.pop("object_frame", None)
        fact["frames"] = {
            "subject": render_slot(subject_raw),
            "predicate": render_slot(predicate_raw),
            "object": render_slot(object_raw),
        }
        fact["vector_frames"] = {
            "subject": render_slot(subject_raw, for_vector=True),
            "predicate": render_slot(predicate_raw, for_vector=True),
            "object": render_slot(object_raw, for_vector=True),
        }
        by_rel.setdefault((row["from"], row["predicate"], row["to"]), []).append(fact)
        all_facts.append(fact)
    add_similarity(all_facts, qvec)
    for key, facts in by_rel.items():
        facts.sort(key=lambda f: -(f.get("sim") if f.get("sim") is not None else 0))
        by_rel[key] = facts[:per_rel]
    return by_rel


def path_similarity(path):
    sims = [f.get("sim") for f in path.get("facts", []) if f.get("sim") is not None]
    return max(sims) if sims else None


def score_path(path, seed_ids):
    ids = {n["id"] for n in path["nodes"]}
    seeds_on_path = len(ids & set(seed_ids))
    words = sum(1 for n in path["nodes"] if n["kind"] == "word")
    sim = path_similarity(path)
    return (10.0 * (sim or 0.0) + 1.0 * seeds_on_path
            - 0.5 * words - 0.3 * (len(path["rels"]) - 1))


def path_key(path):
    return tuple(n["id"] for n in path["nodes"]) + tuple(r["predicate"] for r in path["rels"])


def search(session, question, depth=1, paths_limit=15, min_count=1, with_words=True,
           use_vector=True):
    qvec = question_vector(question) if use_vector else None
    seeds, forms = find_seeds(session, question, qvec=qvec)
    if not seeds:
        return [], [], forms
    seed_ids = [s["id"] for s in seeds]
    raw_paths = expand(session, seed_ids, depth, paths_limit, min_count, with_words)
    by_rel = statements_for(session, raw_paths, qvec)

    unique, seen = [], set()
    for path in raw_paths:
        key = path_key(path)
        if key in seen:
            continue
        seen.add(key)
        path["facts"] = [st for rel in path["rels"]
                         for st in by_rel.get((rel["from"], rel["predicate"], rel["to"]), [])]
        if not path["facts"]:
            continue
        path["score"] = score_path(path, seed_ids)
        unique.append(path)
    unique.sort(key=lambda p: -p["score"])
    return unique[:paths_limit], seeds, forms


def book_name(doc_id, title=None):
    name = re.sub(r"\s*\(\d+\)$", "", (doc_id or "?").split("__")[-1]).strip()
    hashed = re.search(r"_[0-9a-f]{6,}$", name) or name.startswith("_")
    if hashed and title:
        return " ".join(title.split())[:80]
    return re.sub(r"_[0-9a-f]{6,}$", "", name).strip(" _") or (title or doc_id)


def path_line(path):
    parts = [path["nodes"][0]["label"]]
    for index, rel in enumerate(path["rels"]):
        parts.append("--%s-->" % rel["predicate"])
        parts.append(path["nodes"][index + 1]["label"])
    return " ".join(parts)


def fact_line(fact):
    frames = fact.get("frames") or {}
    subject = frames.get("subject") or fact.get("subject_text") or "[—]"
    predicate = frames.get("predicate") or fact.get("predicate") or ""
    obj = frames.get("object") or fact.get("object_text") or "[—]"
    return "%s | %s | %s" % (subject, predicate, obj)


def numbered_paths(paths, max_paths=15, max_facts=3):
    items = []
    for path in paths[:max_paths]:
        facts, sources, formulas, table, caption = [], [], [], "", ""
        for fact in path.get("facts", [])[:max_facts]:
            facts.append({
                "line": fact_line(fact),
                "predicate": fact.get("predicate"),
                "lang": fact.get("lang"),
                "inherited": bool(fact.get("subject_inherited")),
                "sim": fact.get("sim"),
            })
            source = {
                "book": book_name(fact.get("doc_id"), fact.get("doc_title")),
                "page": fact.get("page"),
                "sentence": " ".join((fact.get("sentence") or "").split()),
                "uid": fact.get("sentence_uid"),
                "doc_id": fact.get("doc_id"),
                "seq": fact.get("seq"),
            }
            if source["sentence"] and source not in sources:
                sources.append(source)
            for formula in fact.get("formulas") or []:
                if formula not in formulas:
                    formulas.append(formula)
            if fact.get("table_text") and not table:
                table = fact["table_text"]
                caption = fact.get("table_caption") or ""
        items.append({
            "n": len(items) + 1,
            "path": path_line(path),
            "score": path.get("score", 0),
            "sim": path_similarity(path),
            "facts": facts,
            "formulas": formulas[:3],
            "table": table,
            "table_caption": caption,
            "sources": sources,
        })
    return items


CYPHER_NEIGHBOUR_FORMULAS = """
UNWIND $rows AS row
MATCH (s:Sentence {doc_id: row.doc_id})
WHERE s.seq IN [row.seq - 1, row.seq + 1, row.seq + 2] AND size(s.formulas) > 0
RETURN row.n AS n, s.formulas AS formulas
"""

CYPHER_REFERENCED = """
UNWIND $rows AS row
MATCH (s:Sentence {uid: row.uid})-[:REFERS_TO]->(t:Sentence)
RETURN row.n AS n, t.table_text AS table_text, t.table_caption AS table_caption,
       t.formulas AS formulas
"""


def attach_neighbour_formulas(session, items):
    rows = [{"n": item["n"], "doc_id": src.get("doc_id"), "seq": src.get("seq")}
            for item in items if not item["formulas"]
            for src in item["sources"][:1]
            if src.get("doc_id") and src.get("seq") is not None]
    if not rows:
        return items
    by_number = {item["n"]: item for item in items}
    for row in session.run(CYPHER_NEIGHBOUR_FORMULAS, rows=rows):
        item = by_number.get(row["n"])
        if item is None or len(item["formulas"]) >= 3:
            continue
        for formula in row["formulas"] or []:
            if formula not in item["formulas"]:
                item["formulas"].append(formula)
    return items


def attach_referenced(session, items):
    rows = [{"n": item["n"], "uid": src.get("uid")}
            for item in items for src in item["sources"][:2] if src.get("uid")]
    if not rows:
        return items
    by_number = {item["n"]: item for item in items}
    for row in session.run(CYPHER_REFERENCED, rows=rows):
        item = by_number.get(row["n"])
        if item is None:
            continue
        if row["table_text"] and not item["table"]:
            item["table"] = row["table_text"]
            item["table_caption"] = row["table_caption"] or ""
        for formula in row["formulas"] or []:
            if formula not in item["formulas"] and len(item["formulas"]) < 3:
                item["formulas"].append(formula)
    return items


def sources_for(session, paths, max_paths=15, max_facts=3):
    items = numbered_paths(paths, max_paths, max_facts)
    items = attach_neighbour_formulas(session, items)
    return attach_referenced(session, items)


def build_context(items):
    blocks = []
    for item in items:
        lines = ["[%d] %s" % (item["n"], item["path"])]
        for fact in item["facts"]:
            lines.append("    %s" % fact["line"])
        for formula in item["formulas"]:
            lines.append("    формула: %s" % formula)
        if item["table"]:
            lines.append("    таблица: %s" % (item["table_caption"] or ""))
            lines.extend("      " + row for row in item["table"].splitlines()[:25])
        blocks.append("\n".join(lines))
    return "\n".join(blocks)
