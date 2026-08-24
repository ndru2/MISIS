"""Hybrid formula classifier: ruBERT embeddings + structural features + LightGBM."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer

from pdfscan import paths
from pdfscan.formulas.features import (
    STRUCTURAL_COLUMNS,
    build_bert_input,
    normalize_notation,
    structural_features_matrix,
)

# Многоязычная модель: документы приходят и на русском, и на английском, а
# признаки классификатора обучены в одном пространстве, поэтому переключать
# модель по языку документа нельзя — она должна понимать оба сразу.
BERT_MODEL_NAME = 'intfloat/multilingual-e5-base'

# Модель E5 обучена с усреднением токенов и ожидает пометку роли текста.
BERT_INPUT_PREFIX = 'query: '
FORMULA_CLASSES = ('math', 'physics', 'chemistry')


class FormulaClassifier:
    """BERT + structural features + HistGradientBoosting for formula subtype classification."""

    def __init__(self):
        self.boosting: HistGradientBoostingClassifier | None = None
        self.char_vectorizer: TfidfVectorizer | None = None
        self.char_svd: TruncatedSVD | None = None
        self.label_encoder = LabelEncoder()
        self.bert_name = BERT_MODEL_NAME
        self.tokenizer = None
        self.bert_model = None
        self._bert_device = None

    # ------------------------------------------------------------------ BERT
    def _ensure_bert(self):
        if self.bert_model is not None:
            return
        print(f'🔄 Загрузка BERT: {self.bert_name}...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_name)
        self.bert_model = AutoModel.from_pretrained(self.bert_name)
        self._bert_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model.to(self._bert_device)
        self.bert_model.eval()

    def _encode_bert(self, formulas: list[str], contexts: list[str], batch_size: int = 16) -> np.ndarray:
        self._ensure_bert()
        texts = [BERT_INPUT_PREFIX + build_bert_input(f, c) for f, c in zip(formulas, contexts)]
        chunks = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt',
            )
            encoded = {k: v.to(self._bert_device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.bert_model(**encoded)
                # Усреднение по значимым токенам: дополнение до общей длины в
                # среднее не входит, иначе короткие записи размывались бы.
                mask = encoded['attention_mask'].unsqueeze(-1).float()
                summed = (outputs.last_hidden_state * mask).sum(dim=1)
                pooled = (summed / mask.sum(dim=1).clamp(min=1e-9)).cpu().numpy()
            chunks.append(pooled)

        return np.vstack(chunks).astype(np.float32)

    # ----------------------------------------------------------- feature matrix
    def _build_feature_matrix(
        self,
        formulas: list[str],
        contexts: list[str],
        layouts: list[dict | None] | None = None,
        *,
        fit_char: bool = False,
    ) -> np.ndarray:
        # Обучающие примеры записаны в LaTeX, а из парсера приходит юникод.
        # Приведение к общей нотации обязано случиться до всех признаков,
        # иначе модель учится на одном языке записи, а работает на другом.
        formulas = [normalize_notation(f) for f in formulas]
        if layouts is None:
            layouts = [None] * len(formulas)

        bert_features = self._encode_bert(formulas, contexts)
        struct_features = structural_features_matrix(formulas, contexts, layouts)

        if fit_char:
            self.char_vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 5),
                max_features=500,
            )
            char_sparse = self.char_vectorizer.fit_transform(formulas)
            self.char_svd = TruncatedSVD(n_components=64, random_state=42)
            char_features = self.char_svd.fit_transform(char_sparse)
        else:
            if self.char_vectorizer is None or self.char_svd is None:
                raise RuntimeError('Char vectorizer not fitted. Load or train the model first.')
            char_sparse = self.char_vectorizer.transform(formulas)
            char_features = self.char_svd.transform(char_sparse)

        return np.hstack([bert_features, struct_features, char_features.astype(np.float32)])

    # ----------------------------------------------------------------- training
    @staticmethod
    def _normalize_label(label: str) -> str | None:
        label = str(label).strip().lower()
        if label.startswith('formula ('):
            label = label.replace('formula (', '').replace(')', '').strip()
        if label in FORMULA_CLASSES:
            return label
        return None

    @staticmethod
    def _parse_layout(raw) -> dict | None:
        """Разбор колонки с начертанием; у синтетики её нет."""
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
        return None

    @classmethod
    def prepare_training_data(
        cls,
        base_df: pd.DataFrame,
        extra_csv=paths.DATASETS_DIR / 'retrain_dataset.csv',
        harvested_csv=paths.HARVESTED_DATASET,
    ) -> pd.DataFrame:
        rows = []
        for _, row in base_df.iterrows():
            label = cls._normalize_label(row['class'])
            if label:
                rows.append({
                    'class': label,
                    'latex_code': str(row['latex_code']),
                    'context': str(row.get('context', '')),
                    'layout': None,
                })

        for path, label_column in ((extra_csv, 'correct_class'), (harvested_csv, 'class')):
            if not Path(path).exists():
                continue
            extra = pd.read_csv(path)
            for _, row in extra.iterrows():
                label = cls._normalize_label(row.get(label_column, ''))
                if not label:
                    continue
                # Строка, размеченная самой моделью и никем не проверенная,
                # лишь закрепила бы её собственную ошибку.
                if int(row.get('needs_review', 0) or 0):
                    continue
                rows.append({
                    'class': label,
                    'latex_code': str(row.get('text', row.get('latex_code', ''))),
                    'context': str(row.get('context', '')),
                    'layout': cls._parse_layout(row.get('layout')),
                })

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError('Training dataset is empty after filtering formula classes.')
        return df.drop_duplicates(subset=['class', 'latex_code', 'context']).reset_index(drop=True)

    def train(
        self,
        df: pd.DataFrame,
        *,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['class'],
        )

        y_train = self.label_encoder.fit_transform(train_df['class'])
        y_test = self.label_encoder.transform(test_df['class'])

        print(f'📚 Обучение на {len(train_df)} примерах, тест: {len(test_df)}...')
        x_train = self._build_feature_matrix(
            train_df['latex_code'].tolist(),
            train_df['context'].tolist(),
            train_df['layout'].tolist(),
            fit_char=True,
        )
        x_test = self._build_feature_matrix(
            test_df['latex_code'].tolist(),
            test_df['context'].tolist(),
            test_df['layout'].tolist(),
            fit_char=False,
        )

        self.boosting = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=8,
            class_weight='balanced',
            random_state=random_state,
        )
        self.boosting.fit(x_train, y_train)

        y_pred = self.boosting.predict(x_test)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        report = classification_report(
            y_test,
            y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0,
        )
        print(f'✅ Macro F1 на тесте: {macro_f1:.3f}')
        print(report)
        return {'macro_f1': macro_f1, 'report': report}

    # -------------------------------------------------------------- persistence
    def save(self, path=paths.FORMULA_CLASSIFIER):
        if self.boosting is None:
            raise RuntimeError('Model is not trained.')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'version': 2,
            'bert_name': self.bert_name,
            'boosting': self.boosting,
            'char_vectorizer': self.char_vectorizer,
            'char_svd': self.char_svd,
            'label_encoder': self.label_encoder,
            'structural_columns': STRUCTURAL_COLUMNS,
        }
        joblib.dump(payload, path)
        print(f'💾 Модель сохранена: {path}')

    @classmethod
    def load(cls, path=paths.FORMULA_CLASSIFIER) -> 'FormulaClassifier | None':
        if not Path(path).exists():
            return None

        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get('version') != 2:
            print(f'⚠️ Устаревший формат {Path(path).name} — '
                  'запустите python -m pdfscan.formulas.train')
            return None

        model = cls()
        model.bert_name = payload['bert_name']
        model.boosting = payload['boosting']
        model.char_vectorizer = payload['char_vectorizer']
        model.char_svd = payload['char_svd']
        model.label_encoder = payload['label_encoder']
        return model

    # --------------------------------------------------------------- inference
    def predict_batch(self, formulas: list[str], contexts: list[str], layouts=None) -> list[str]:
        if not formulas:
            return []
        if self.boosting is None:
            raise RuntimeError('Model is not loaded.')

        x = self._build_feature_matrix(formulas, contexts, layouts, fit_char=False)
        y_pred = self.boosting.predict(x)
        return self.label_encoder.inverse_transform(y_pred).tolist()

    def predict(self, formula: str, context: str, layout: dict | None = None) -> str:
        return self.predict_batch([formula], [context], [layout])[0]

    def predict_proba(self, formula: str, context: str, layout: dict | None = None) -> dict[str, float]:
        if self.boosting is None:
            raise RuntimeError('Model is not loaded.')
        x = self._build_feature_matrix([formula], [context], [layout], fit_char=False)
        probas = self.boosting.predict_proba(x)[0]
        return {
            cls: float(prob)
            for cls, prob in zip(self.label_encoder.classes_, probas)
        }
