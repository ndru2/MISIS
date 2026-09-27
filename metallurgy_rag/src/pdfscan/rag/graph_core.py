"""Graph traversal adapted from kg_rag; lexical seeds, no dense retrieval."""

from __future__ import annotations

import json

from pdfscan.rag.graph_terms import query_terms

SEED_LIMIT = 12
PATH_LIMIT = 40
STATEMENTS_PER_REL = 3
STATEMENTS_FETCH = 8


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

def find_seeds(session, question, limit=SEED_LIMIT):
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
            raise RuntimeError("Neo4j term_search недоступен") from exc
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


def statements_for(session, paths, per_rel=STATEMENTS_PER_REL):
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
    by_rel = {}
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
        by_rel.setdefault((row["from"], row["predicate"], row["to"]), []).append(fact)
    for key, facts in by_rel.items():
        by_rel[key] = facts[:per_rel]
    return by_rel


def score_path(path, seed_ids):
    ids = {n["id"] for n in path["nodes"]}
    seeds_on_path = len(ids & set(seed_ids))
    words = sum(1 for n in path["nodes"] if n["kind"] == "word")
    return (1.0 * seeds_on_path
            - 0.5 * words - 0.3 * (len(path["rels"]) - 1))


def path_key(path):
    return tuple(n["id"] for n in path["nodes"]) + tuple(r["predicate"] for r in path["rels"])


def search(session, question, depth=1, paths_limit=15, min_count=1, with_words=True,
           use_vector=False):
    if use_vector:
        raise ValueError("Graph channel uses lexical seeds only")
    seeds, forms = find_seeds(session, question)
    if not seeds:
        return [], [], forms
    seed_ids = [s["id"] for s in seeds]
    raw_paths = expand(session, seed_ids, depth, paths_limit, min_count, with_words)
    by_rel = statements_for(session, raw_paths)

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


def path_line(path):
    parts = [path["nodes"][0]["label"]]
    for index, rel in enumerate(path["rels"]):
        forward = rel.get("from") == path["nodes"][index].get("id")
        parts.append(("--%s-->" if forward else "<--%s--") % rel["predicate"])
        parts.append(path["nodes"][index + 1]["label"])
    return " ".join(parts)


def fact_line(fact):
    frames = fact.get("frames") or {}
    subject = frames.get("subject") or fact.get("subject_text") or "[—]"
    predicate = frames.get("predicate") or fact.get("predicate") or ""
    obj = frames.get("object") or fact.get("object_text") or "[—]"
    return "%s | %s | %s" % (subject, predicate, obj)
