"""Векторизация, индексация и поиск по кускам документа.

Поиск гибридный. Плотный вектор понимает смысл запроса и находит нужное, даже
если слова не совпали, но плохо ловит обозначения: ``Fe_2O_3`` и ``σ_т`` для него
почти неразличимы. Лексический поиск, наоборот, точен на символах и формулах, но
беспомощен к синонимам. Оба списка сводятся вместе по рангу, поэтому кусок,
поднявшийся в любом из них, попадает в ответ.
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np

from pdfscan.paths import RAG_INDEX_DIR

EMBEDDING_MODEL = 'intfloat/multilingual-e5-base'

# E5 обучена на парах «запрос — отрывок» и различает их по этой пометке. Без неё
# запрос и документ ложатся в разные части пространства, и поиск заметно хуже.
QUERY_PREFIX = 'query: '
PASSAGE_PREFIX = 'passage: '

_TOKEN_RE = re.compile(r'[^\W_]+', re.UNICODE)


def _load_faiss():
    """faiss, но только после torch.

    Обе библиотеки приносят с собой свою сборку OpenMP, и на macOS та, что
    инициализируется второй, роняет процесс по SIGSEGV — не на импорте, а позже,
    на первом же счёте в несколько потоков. Порядок «сначала torch» устойчив,
    обратный воспроизводимо падает, поэтому импорт спрятан сюда: иначе он
    зависел бы от того, какая из функций модуля вызвана первой.
    """
    import torch  # noqa: F401  — обязан быть загружен раньше faiss
    import faiss
    return faiss


class Encoder:
    """Переводит текст в вектор усреднением токенов, как обучалась модель."""

    def __init__(self, model_name=EMBEDDING_MODEL, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts, prefix='', batch_size=16, progress=False):
        vectors = []
        texts = [prefix + t for t in texts]

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=512, return_tensors='pt')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with self.torch.no_grad():
                output = self.model(**encoded)
                mask = encoded['attention_mask'].unsqueeze(-1).float()
                pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vectors.append(pooled.cpu().numpy())
            if progress:
                print(f'   векторизовано {min(start + batch_size, len(texts))}/{len(texts)}')

        matrix = np.vstack(vectors).astype('float32')
        # Косинусная близость считается скалярным произведением нормированных
        # векторов, поэтому длина снимается заранее.
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-9)
        return matrix


def tokenize(text: str) -> list[str]:
    """Разбивает текст на слова и приводит их к основе.

    Основы нужны русскому: без них «деформации» в запросе не находит
    «деформация» в тексте. Обозначения и формулы вроде ``Fe2O3`` остаются как
    есть — сокращать их не по чему.
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [_stem(t) for t in tokens if len(t) > 1]


_STEMMERS = {}


def _stem(token: str) -> str:
    if any(c.isdigit() for c in token):
        return token
    language = 'russian' if any('а' <= c <= 'я' or c == 'ё' for c in token) else 'english'
    if language not in _STEMMERS:
        from nltk.stem.snowball import SnowballStemmer
        _STEMMERS[language] = SnowballStemmer(language)
    return _STEMMERS[language].stem(token)


def _search_text(chunk: dict) -> str:
    """Строка, по которой кусок ищется: текст плюс разобранные метаданные.

    Названия элементов и приведённые единицы дописываются к тексту, чтобы запрос
    «оксид железа» находил ``Fe_2O_3``, а «kJ/mol» — «кДж/моль».
    """
    extra = list(chunk.get('element_names', []))
    extra += sorted({u['canonical'] for u in chunk.get('units', [])})
    extra += sorted({u['unit'] for u in chunk.get('units', [])})
    return chunk['text_search'] + (' ' + ' '.join(extra) if extra else '')


class RagIndex:
    def __init__(self, chunks, dense, corpus_tokens, encoder=None):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.dense = dense
        self.corpus_tokens = corpus_tokens
        self.bm25 = BM25Okapi(corpus_tokens)
        self.encoder = encoder
        self._faiss = None

    # ------------------------------------------------------------------ сборка
    @classmethod
    def build(cls, chunks, model_name=EMBEDDING_MODEL, encoder=None, progress=True):
        encoder = encoder or Encoder(model_name)
        texts = [_search_text(c) for c in chunks]
        if progress:
            print(f'🔢 Векторизация {len(texts)} кусков...')
        dense = encoder.encode(texts, prefix=PASSAGE_PREFIX, progress=progress)
        corpus_tokens = [tokenize(t) for t in texts]
        return cls(chunks, dense, corpus_tokens, encoder)

    def save(self, out_dir=RAG_INDEX_DIR):
        faiss = _load_faiss()

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        index = faiss.IndexFlatIP(self.dense.shape[1])
        index.add(self.dense)
        faiss.write_index(index, str(out_dir / 'dense.faiss'))

        with open(out_dir / 'chunks.jsonl', 'w', encoding='utf-8') as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        with open(out_dir / 'lexical.pkl', 'wb') as handle:
            pickle.dump(self.corpus_tokens, handle)

        print(f'💾 Индекс сохранён: {out_dir} — кусков {len(self.chunks)},'
              f' размерность {self.dense.shape[1]}')
        return out_dir

    @classmethod
    def load(cls, out_dir=RAG_INDEX_DIR, model_name=EMBEDDING_MODEL, encoder=None):
        faiss = _load_faiss()

        out_dir = Path(out_dir)
        with open(out_dir / 'chunks.jsonl', encoding='utf-8') as handle:
            chunks = [json.loads(line) for line in handle if line.strip()]
        with open(out_dir / 'lexical.pkl', 'rb') as handle:
            corpus_tokens = pickle.load(handle)

        index = faiss.read_index(str(out_dir / 'dense.faiss'))
        dense = index.reconstruct_n(0, index.ntotal)
        return cls(chunks, dense, corpus_tokens, encoder or Encoder(model_name))

    # ------------------------------------------------------------------- поиск
    def search(self, query, k=5, candidates=40, filters=None, rrf_k=60):
        """Гибридный поиск: смысловая близость и совпадение слов вместе.

        Списки сводятся по рангу, а не по величине оценки: у плотного поиска и
        у лексического они в разных шкалах, и сравнивать их напрямую нельзя.
        """
        allowed = self._filter(filters)
        if not allowed:
            return []

        vector = self.encoder.encode([query], prefix=QUERY_PREFIX)[0]
        dense_scores = self.dense @ vector
        lexical_scores = np.asarray(self.bm25.get_scores(tokenize(query)))

        ranking = {}
        for scores in (dense_scores, lexical_scores):
            order = [i for i in np.argsort(-scores) if i in allowed][:candidates]
            for rank, index in enumerate(order):
                ranking[index] = ranking.get(index, 0.0) + 1.0 / (rrf_k + rank + 1)

        best = sorted(ranking, key=ranking.get, reverse=True)[:k]
        results = []
        for index in best:
            chunk = dict(self.chunks[index])
            chunk['score'] = round(float(ranking[index]), 5)
            chunk['dense_score'] = round(float(dense_scores[index]), 4)
            chunk['lexical_score'] = round(float(lexical_scores[index]), 3)
            results.append(chunk)
        return results

    def _filter(self, filters):
        """Отбор кусков по метаданным до ранжирования."""
        if not filters:
            return set(range(len(self.chunks)))

        allowed = set()
        for index, chunk in enumerate(self.chunks):
            if not all(_matches(chunk, key, value) for key, value in filters.items()):
                continue
            allowed.add(index)
        return allowed


def _matches(chunk, key, value):
    actual = chunk.get(key)
    if isinstance(value, (list, tuple, set)):
        if isinstance(actual, list):
            return bool(set(actual) & set(value))
        return actual in value
    if isinstance(actual, list):
        return value in actual
    return actual == value
