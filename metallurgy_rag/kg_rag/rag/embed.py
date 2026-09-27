from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
import retrieval

MODEL_NAME = "BAAI/bge-m3"
DIM = 1024
BATCH = 256
MAX_TOKENS = 512

_model = None
_tokenizer = None
_device = None


def _load():
    global _model, _tokenizer, _device
    if _model is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16 if _device == "cuda" else torch.float32)
        _model = _model.to(_device).eval()
    return _model, _tokenizer, _device


def encode(texts):
    import torch

    model, tokenizer, device = _load()
    out = []
    for start in range(0, len(texts), BATCH):
        chunk = [t or " " for t in texts[start:start + BATCH]]
        batch = tokenizer(chunk, padding=True, truncation=True,
                          max_length=MAX_TOKENS, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        out.extend(hidden.float().cpu().tolist())
    return out


CYPHER_TERMS_TODO = """
MATCH (t:Term) WHERE t.embedding IS NULL
RETURN t.id AS id, t.label AS label, t.label_ru AS ru, t.label_en AS en,
       t.aliases AS aliases, t.formulas AS formulas
LIMIT $limit
"""

CYPHER_TERMS_SET = """
UNWIND $rows AS row
MATCH (t:Term {id: row.id})
CALL db.create.setNodeVectorProperty(t, 'embedding', row.embedding)
"""

CYPHER_STATEMENTS_TODO = """
MATCH (s:Statement) WHERE s.embedding IS NULL AND s.text IS NOT NULL
RETURN s.uid AS uid, s.text AS text LIMIT $limit
"""

CYPHER_STATEMENTS_SET = """
UNWIND $rows AS row
MATCH (s:Statement {uid: row.uid})
CALL db.create.setNodeVectorProperty(s, 'embedding', row.embedding)
"""


def term_text(row):
    parts = [row["ru"], row["en"]]
    if not any(parts):
        parts = [row["label"]]
    parts += list(row["aliases"] or [])[:6]
    parts += list(row["formulas"] or [])[:3]
    return " / ".join(p for p in parts if p)


def fill(session, todo_cypher, set_cypher, text_of, key, label, batch):
    total = 0
    while True:
        rows = [dict(r) for r in session.run(todo_cypher, limit=batch)]
        if not rows:
            break
        vectors = encode([text_of(r) for r in rows])
        session.run(set_cypher, rows=[{key: r[key], "embedding": v}
                                      for r, v in zip(rows, vectors)])
        total += len(rows)
        print("  %s: %d" % (label, total), flush=True)
    return total
