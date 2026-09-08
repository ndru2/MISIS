"""Qdrant-индекс плоских multi-vector search units.

Каждый объект имеет BM25/sparse представление и специализированный dense-вектор:
text, chemistry, math, table или unit. Parent-child связь — только metadata.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from pdfscan.paths import DATA_DIR
from pdfscan.rag.tokenize import tokenize as domain_tokenize

DEFAULT_COLLECTION = 'metallurgy_search_units'
LEGACY_COLLECTIONS = ('metallurgy_chunks', 'metallurgy_triples')
DEFAULT_MODEL = 'BAAI/bge-m3'
FALLBACK_MODEL = 'intfloat/multilingual-e5-base'
DEFAULT_URL = 'http://localhost:6333'
VOCAB_PATH = DATA_DIR / 'qdrant_vocab.json'
UPSERT_BATCH = 64
PARENT_TEXT_LIMIT = 4000
VECTOR_NAMES = {'text': 'dense_text', 'chemistry': 'dense_chemistry',
                'math': 'dense_math', 'table': 'dense_table', 'unit': 'dense_unit'}


class SparseVocab:
    """Устойчивое соответствие доменных термов индексам sparse-вектора."""

    def __init__(self, token_to_id=None):
        self.token_to_id = dict(token_to_id or {})

    def encode(self, tokens: list[str]):
        counts = Counter(tokens)
        indices, values = [], []
        for token, count in counts.items():
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id) + 1
            indices.append(self.token_to_id[token])
            values.append(float(count))
        return indices, values

    def save(self, path=VOCAB_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.token_to_id, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, path=VOCAB_PATH):
        path = Path(path)
        return cls(json.loads(path.read_text(encoding='utf-8'))) if path.exists() else cls()


def _clip(text, limit=PARENT_TEXT_LIMIT):
    if not text:
        return None
    return text if len(text) <= limit else text[:limit] + '…'


def unit_payload(unit: dict) -> dict:
    """Payload содержит provenance, родителя и structured values."""
    payload = dict(unit)
    payload.pop('vector_kind', None)
    payload.setdefault('source', 'chunk' if payload.get('unit_type', 'chunk') == 'chunk'
                       else payload.get('unit_type'))
    payload['parent_text'] = _clip(payload.get('parent_text'))
    payload['units'] = [value.get('unit') if isinstance(value, dict) else value
                        for value in payload.get('units') or []]
    return payload


chunk_payload = unit_payload  # legacy import name


def _model_is_cached(name: str) -> bool:
    return (Path.home() / '.cache/huggingface/hub' / f'models--{name.replace("/", "--")}').exists()


def _pick_model(name: str) -> str:
    if _model_is_cached(name) or name == FALLBACK_MODEL:
        return name
    return FALLBACK_MODEL if _model_is_cached(FALLBACK_MODEL) else name


class _DenseBackend:
    def __init__(self, model_name: str):
        self.name = _pick_model(model_name)
        self.kind = 'encoder'
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.name)
            self.kind = 'st'
        except Exception:
            from pdfscan.rag.index import Encoder
            self.model = Encoder(self.name)

    def encode(self, texts, *, is_query=False, progress=False):
        from pdfscan.rag.index import PASSAGE_PREFIX, QUERY_PREFIX
        prefix = QUERY_PREFIX if is_query and 'e5' in self.name.lower() else ''
        if not is_query and 'e5' in self.name.lower():
            prefix = PASSAGE_PREFIX
        if self.kind == 'st':
            vectors = self.model.encode([prefix + text for text in texts],
                                        normalize_embeddings=True, show_progress_bar=progress,
                                        batch_size=16)
            return [row.tolist() for row in vectors], len(vectors[0])
        matrix = self.model.encode(texts, prefix=prefix, progress=progress)
        return [row.tolist() for row in matrix], matrix.shape[1]


_BACKENDS: dict[str, _DenseBackend] = {}


def _encode(texts: list[str], model_name: str, *, is_query=False, progress=False):
    chosen = _pick_model(model_name)
    backend = _BACKENDS.get(chosen)
    if backend is None:
        backend = _BACKENDS[chosen] = _DenseBackend(chosen)
    return backend.encode(texts, is_query=is_query, progress=progress)


def _client(url: str):
    from qdrant_client import QdrantClient
    return QdrantClient(url=url, timeout=60, check_compatibility=False)


def _point_id(unit_id: str) -> int:
    digest = hashlib.blake2b(unit_id.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'big') % (2 ** 63)


def _models(model_names: dict[str, str] | None) -> dict[str, str]:
    model_names = model_names or {}
    return {kind: model_names.get(kind) or DEFAULT_MODEL for kind in VECTOR_NAMES}


def _encoded_units(units: list[dict], model_names: dict[str, str], progress: bool):
    vectors, dimensions, grouped = {}, {}, defaultdict(list)
    for position, unit in enumerate(units):
        grouped[unit['vector_kind']].append((position, unit.get('text_search') or unit.get('text') or ''))
    for kind, items in grouped.items():
        rows, dim = _encode([text for _, text in items], model_names[kind], progress=progress)
        dimensions[VECTOR_NAMES[kind]] = dim
        for (position, _), row in zip(items, rows):
            vectors[position] = (VECTOR_NAMES[kind], row)
    return vectors, dimensions


def ensure_collection(client, name: str, dimensions: dict[str, int], *, recreate=False):
    from qdrant_client import models
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={vector: models.VectorParams(size=dim, distance=models.Distance.COSINE)
                        for vector, dim in dimensions.items()},
        sparse_vectors_config={'bm25': models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )


def initialize_collection(*, url=DEFAULT_URL, collection=DEFAULT_COLLECTION,
                          model_names=None, recreate=False):
    """Создаёт все named vector spaces до потоковой загрузки документов."""
    configured = _models(model_names)
    dimensions = {}
    for kind, vector_name in VECTOR_NAMES.items():
        _, dimension = _encode(['dimension probe'], configured[kind], progress=False)
        dimensions[vector_name] = dimension
    client = _client(url)
    ensure_collection(client, collection, dimensions, recreate=recreate)
    return dimensions


def drop_legacy_collections(*, url=DEFAULT_URL, client=None):
    """Удаляет старые Wikontic/Qdrant-коллекции при явной пересборке."""
    client = client or _client(url)
    removed = []
    for name in LEGACY_COLLECTIONS:
        if client.collection_exists(name):
            client.delete_collection(name)
            removed.append(name)
    return removed


def upsert_units(units: list[dict], *, url=DEFAULT_URL, collection=DEFAULT_COLLECTION,
                 model_names=None, vocab=None, progress=True, recreate=False, save_vocab=True):
    """Индексирует parent и child units в одной Qdrant-коллекции."""
    from qdrant_client import models
    if not units:
        raise ValueError('нет search units для индекса')
    configured = _models(model_names)
    dense, dimensions = _encoded_units(units, configured, progress)
    vocab = vocab or SparseVocab.load()
    client = _client(url)
    ensure_collection(client, collection, dimensions, recreate=recreate)
    points = []
    for position, unit in enumerate(units):
        vector_name, vector = dense[position]
        indices, values = vocab.encode(domain_tokenize(unit.get('text_search') or ''))
        points.append(models.PointStruct(
            id=_point_id(unit['unit_id']),
            vector={vector_name: vector, 'bm25': models.SparseVector(indices=indices, values=values)},
            payload=unit_payload(unit),
        ))
    for start in range(0, len(points), UPSERT_BATCH):
        client.upsert(collection_name=collection, points=points[start:start + UPSERT_BATCH])
        if progress:
            print(f'   записано {min(start + UPSERT_BATCH, len(points))}/{len(points)}')
    if save_vocab:
        vocab.save()
    return {'collection': collection, 'points': len(points), 'dimensions': dimensions}


def _filter(filters: dict | None):
    from qdrant_client import models
    if not filters:
        return None
    mapping = {'has_formula': 'has_formula', 'has_table': 'has_table',
               'has_chemistry': 'has_chemistry', 'doc_id': 'doc_id', 'lang': 'lang',
               'doc_type': 'doc_type', 'category': 'category', 'year': 'year',
               'unit_type': 'unit_type'}
    conditions = [models.FieldCondition(key=field, match=models.MatchValue(value=filters[key]))
                  for key, field in mapping.items() if key in filters]
    if filters.get('element'):
        conditions.append(models.FieldCondition(key='elements', match=models.MatchValue(value=filters['element'])))
    numeric = filters.get('numeric') or {}
    if numeric.get('canonical'):
        conditions.append(models.FieldCondition(
            key='unit_canonical', match=models.MatchValue(value=numeric['canonical'])))
    range_args = {key: numeric[key] for key in ('gt', 'gte', 'lt', 'lte') if key in numeric}
    if range_args:
        conditions.append(models.FieldCondition(key='si_value', range=models.Range(**range_args)))
    return models.Filter(must=conditions) if conditions else None


def retrieve_units(unit_ids: list[str], *, url=DEFAULT_URL, collection=DEFAULT_COLLECTION,
                   client=None) -> dict[str, dict]:
    """Читает parent units по стабильным ID для small-to-big retrieval."""
    if not unit_ids:
        return {}
    client = client or _client(url)
    points = client.retrieve(collection_name=collection,
                             ids=[_point_id(unit_id) for unit_id in unit_ids],
                             with_payload=True, with_vectors=False)
    result = {}
    for point in points:
        payload = dict(point.payload or {})
        if payload.get('unit_id'):
            result[payload['unit_id']] = payload
    return result


def search(query: str, *, k=8, url=DEFAULT_URL, collection=DEFAULT_COLLECTION,
           model_names=None, vocab=None, filters=None, candidates=40, client=None,
           vector_kinds=None):
    """Sparse + выбранные dense spaces → RRF; графового retriever-а нет."""
    from qdrant_client import models
    configured = _models(model_names)
    vector_kinds = vector_kinds or tuple(VECTOR_NAMES)
    query_filter, prefetch = _filter(filters), []
    for kind in vector_kinds:
        if kind not in VECTOR_NAMES:
            continue
        dense, _ = _encode([query], configured[kind], is_query=True)
        prefetch.append(models.Prefetch(query=dense[0], using=VECTOR_NAMES[kind],
                                        limit=candidates, filter=query_filter))
    vocab = vocab or SparseVocab.load()
    indices, values = vocab.encode(domain_tokenize(query))
    prefetch.append(models.Prefetch(query=models.SparseVector(indices=indices, values=values),
                                    using='bm25', limit=candidates, filter=query_filter))
    client = client or _client(url)
    result = client.query_points(collection_name=collection, prefetch=prefetch,
                                 query=models.FusionQuery(fusion=models.Fusion.RRF),
                                 limit=k, with_payload=True)
    hits = []
    for point in result.points:
        payload = dict(point.payload or {})
        payload['score'], payload['point_id'] = round(float(point.score), 5), point.id
        hits.append(payload)
    return hits
