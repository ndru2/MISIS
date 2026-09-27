"""Query router и единый Evidence-контракт плоского RAG.

Модуль объединяет выбранные dense, BM25 и Neo4j каналы в Evidence.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace

from pdfscan.rag import qdrant_index, graph
from pdfscan.rag.normalize import extract_units, normalize_query
from pdfscan.rag.rerank import DEFAULT_RELEVANCE_THRESHOLD, default_reranker

_CHEM_RE = re.compile(r'\b(?:[A-Z][a-z]?\d*){2,}\b')
_MATH_RE = re.compile(r'(?:\b(?:equation|formula|уравнен|формул)\b|[=∫∑∂τθ]|\b[a-zA-Z]_[A-Za-zА-Яа-я]+)')
_TABLE_RE = re.compile(r'\b(?:table|таблиц|табл|состав|content|composition|строк[аи])\b', re.I)
_CHEM_WORD_RE = re.compile(r'\b(?:oxide|slag|matte|reaction|оксид|шлак|штейн|реакц|магнетит)\b', re.I)
_LOWER_RE = re.compile(r'(?:выше|более|больше|не\s+менее|above|over|more\s+than|at\s+least)', re.I)
_UPPER_RE = re.compile(r'(?:ниже|менее|меньше|не\s+более|below|under|less\s+than|at\s+most)', re.I)


@dataclass(frozen=True)
class QueryProfile:
    query: str
    vector_kinds: tuple[str, ...]
    filters: dict
    channels: tuple[str, ...]
    reranker: str = 'cross-encoder'
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    candidate_count: int = 0
    accepted_count: int = 0
    methods: tuple[str, ...] = ('vector', 'bm25')
    channel_counts: dict | None = None
    dense_models: dict | None = None
    reranker_settings: dict | None = None


@dataclass(frozen=True)
class Evidence:
    """Один недублирующийся результат, готовый для генератора или API."""

    unit_id: str
    unit_type: str
    text: str
    context: str
    parent_chunk_id: str | None
    document_id: str | None
    pages: tuple[int, ...]
    score: float
    retrieval_channels: tuple[str, ...]
    structured: dict
    relevance_score: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _numeric_filter(query: str) -> dict | None:
    values = extract_units(query)
    if not values:
        return None
    first = values[0]
    # Сложные единицы пока не имеют однозначного si_value; exact lexical search
    # для них остаётся включённым, а range применяем только к простым.
    si_value = first.get('si_value')
    if si_value is None:
        return None
    before = query[:query.find(first['text'])]
    after = query[query.find(first['text']) + len(first['text']):]
    scope = f'{before} {after}'
    numeric = {'canonical': first['canonical']}
    if _LOWER_RE.search(scope):
        numeric['gte'] = si_value
    elif _UPPER_RE.search(scope):
        numeric['lte'] = si_value
    else:
        # Без оператора число — поисковый якорь, а не фильтр равенства.
        return None
    return numeric


def route_query(query: str) -> QueryProfile:
    """Детерминированный router; позже его можно заменить классификатором."""
    kinds = []
    if _CHEM_RE.search(query) or _CHEM_WORD_RE.search(query):
        kinds.append('chemistry')
    if _MATH_RE.search(query):
        kinds.append('math')
    if _TABLE_RE.search(query):
        kinds.append('table')
    numeric = _numeric_filter(query)
    if numeric or extract_units(query):
        kinds.append('unit')
    # Текстовый канал всегда даёт объясняющий контекст, а BM25 всегда добавляет
    # точные формулы, номера и единицы.
    kinds.append('text')
    unique = tuple(dict.fromkeys(kinds))
    filters = {'numeric': numeric, 'unit_type': 'quantity'} if numeric else {}
    return QueryProfile(query=query, vector_kinds=unique, filters=filters,
                        channels=(*unique, 'bm25'))


def _group_key(hit: dict) -> str:
    return hit.get('parent_chunk_id') or hit.get('unit_id') or hit.get('chunk_id') or ''


def _structured(hit: dict) -> dict:
    keys = ('formula_latex', 'formula_flat', 'equation_number', 'formula_class',
            'table_id', 'row_id', 'row_label', 'column_header', 'cell_value',
            'value', 'unit_raw', 'unit_canonical', 'si_value', 'dimension', 'graph')
    return {key: hit[key] for key in keys if hit.get(key) is not None}


METHODS = ('vector', 'bm25', 'graph')


def normalize_methods(methods=None):
    selected = ('vector', 'bm25') if methods is None else tuple(methods)
    if not selected or set(selected) - set(METHODS):
        raise ValueError('Выберите один или несколько методов: vector, bm25, graph')
    return tuple(method for method in METHODS if method in selected)


def fuse_rankings(rankings, rrf_k=60):
    """One RRF over every enabled ranked list; record actual hit provenance."""
    merged = {}
    for channel, hits in rankings.items():
        seen = set()
        for rank, hit in enumerate(hits, 1):
            key = hit.get('unit_id') or hit.get('chunk_id')
            if not key or key in seen:
                continue
            seen.add(key)
            item = merged.setdefault(key, dict(hit, score=0.0, retrieval_channels=[]))
            item['score'] += 1.0 / (rrf_k + rank)
            item['retrieval_channels'].append(channel)
    return sorted(merged.values(), key=lambda hit: -hit['score'])


def retrieve(query: str, *, k=8, candidates=40, url=qdrant_index.DEFAULT_URL,
             collection=qdrant_index.DEFAULT_COLLECTION, model_names=None,
             client=None, relevance_threshold=DEFAULT_RELEVANCE_THRESHOLD,
             reranker=None, methods=None) -> tuple[QueryProfile, list[Evidence]]:
    """Hybrid RRF → parent-child dedup → reranker → relevance gate → Evidence.

    Запрос приводится к той же нотации, что и корпус (LaTeX/кириллические
    омографы формул разворачиваются, обычные слова не трогаются — см.
    ``normalize_query``), и уже в этом виде идёт и в router, и в Qdrant.
    """
    search_query = normalize_query(query)
    profile = route_query(search_query)
    methods = normalize_methods(methods)
    kinds = profile.vector_kinds if 'vector' in methods else ()
    profile = replace(profile, vector_kinds=kinds, methods=methods,
                      channels=(*kinds, *(('bm25',) if 'bm25' in methods else ()),
                                *(('graph',) if 'graph' in methods else ())))
    profile = replace(profile, dense_models={kind: qdrant_index._pick_model(
        (model_names or {}).get(kind) or qdrant_index.DEFAULT_MODEL) for kind in kinds})
    rankings = {}
    if 'vector' in methods or 'bm25' in methods:
        rankings = qdrant_index.search(
            search_query, k=max(candidates, k * 4), candidates=max(candidates, k * 4), url=url,
            collection=collection, model_names=model_names, filters=profile.filters or None,
            vector_kinds=kinds, use_bm25='bm25' in methods, return_rankings=True, client=client)
        # Supports callers that provide a pre-fused candidate list.
        if isinstance(rankings, list):
            rankings = {'qdrant': rankings}
    if 'graph' in methods:
        rankings['graph'] = graph.search(search_query, candidates=max(candidates, k * 4))
    profile = replace(profile, channel_counts={name: len(items) for name, items in rankings.items()})
    hits = fuse_rankings(rankings)
    best = {}
    for hit in hits:
        key = _group_key(hit)
        if key and (key not in best or hit.get('score', 0) > best[key].get('score', 0)):
            best[key] = hit
    candidates_after_dedup = sorted(
        best.values(), key=lambda item: item.get('score', 0), reverse=True)[:max(candidates, k * 4)]
    parent_ids = [item['parent_chunk_id'] for item in candidates_after_dedup
                  if item.get('parent_chunk_id')]
    parents = (qdrant_index.retrieve_units(parent_ids, url=url, collection=collection, client=client)
               if parent_ids else {})
    reranker = reranker or default_reranker()
    texts = []
    for hit in candidates_after_dedup:
        parent = parents.get(hit.get('parent_chunk_id') or '')
        context = (parent or hit).get('text') or hit.get('parent_text') or ''
        texts.append((hit.get('text') or '') + '\n' + context)
    relevance = reranker.score(query, texts)
    ranked = sorted(zip(candidates_after_dedup, relevance), key=lambda pair: pair[1], reverse=True)
    accepted = [(hit, score) for hit, score in ranked if score >= relevance_threshold][:k]
    profile = replace(profile, reranker=('lexical-fallback' if getattr(reranker, '_failed', False)
                                        else getattr(reranker, 'model_name', reranker.__class__.__name__)),
                      reranker_settings={name: getattr(reranker, name, None)
                                         for name in ('max_length', 'batch_size', 'device')},
                      relevance_threshold=relevance_threshold,
                      candidate_count=len(candidates_after_dedup), accepted_count=len(accepted))
    evidence = []
    for hit, relevance_score in accepted:
        parent = parents.get(hit.get('parent_chunk_id') or '')
        context = (parent or hit).get('text') or hit.get('parent_text') or ''
        evidence.append(Evidence(
            unit_id=hit.get('unit_id') or hit.get('chunk_id') or '',
            unit_type=hit.get('unit_type') or 'chunk', text=hit.get('text') or '', context=context,
            parent_chunk_id=hit.get('parent_chunk_id'), document_id=hit.get('doc_id'),
            pages=tuple(hit.get('pages') or ()), score=float(hit.get('score') or 0),
            retrieval_channels=tuple(hit.get('retrieval_channels', ())), structured=_structured(hit),
            relevance_score=round(float(relevance_score), 4),
        ))
    return profile, evidence
