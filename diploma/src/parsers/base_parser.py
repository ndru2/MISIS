from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Token:
    """Токен с метаданными и эмбеддингом"""
    text: str                           # Исходный текст слова
    lemma: str                          # Лемма (нормальная форма)
    pos: str                            # Часть речи (NOUN, VERB, etc.)
    
    embedding: Optional[np.ndarray] = None  # Контекстуальный эмбеддинг
    position: Optional[int] = None          # Позиция в тексте
    sentence: Optional[str] = None          # Предложение, из которого извлечен токен
    confidence: Optional[float] = None      # Уверенность (если применимо)
    
    def __repr__(self) -> str:
        emb_info = f", emb_dim={len(self.embedding)}" if self.embedding is not None else ""
        return f"Token(text='{self.text}', lemma='{self.lemma}', pos={self.pos}{emb_info})"
    
    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'lemma': self.lemma,
            'pos': self.pos,
            'position': self.position,
            'sentence': self.sentence,
            'confidence': self.confidence,
            'has_embedding': self.embedding is not None,
            'embedding_dim': len(self.embedding) if self.embedding is not None else None
        }
    
@dataclass
class Triplet:    
    subject: str
    predicate: str
    object: str

    subject_pos: Optional[int] = None
    predicate_pos: Optional[int] = None
    object_pos: Optional[int] = None
    confidence: Optional[float] = None
    sentence: Optional[str] = None

    def __repr__(self) -> str:
        return f"({self.subject} -> {self.predicate} -> {self.object})"

    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def to_dict(self) -> dict:
        return {
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.object,
            'subject_pos': self.subject_pos,
            'predicate_pos': self.predicate_pos,
            'object_pos': self.object_pos,
            'confidence': self.confidence,
            'sentence': self.sentence
        }


class BaseParser(ABC):
    def __init__(self, name: str = "BaseParser"):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def extract_triplets(self, text: str) -> List[Triplet]:
        """Извлечение триплетов (субъект-предикат-объект)"""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Проверка готовности парсера к работе"""
        pass
    
    @abstractmethod
    def extract_tokens(self, text: str, pos_filter: Optional[List[str]] = None) -> List[Token]:
        """
        Args:
            text: Входной текст
            pos_filter: Список частей речи для фильтрации ['NOUN', 'VERB']
        
        Returns:
            Список токенов
        """
        pass
        
    def extract_and_display(self, text: str) -> List[Triplet]:
        triplets = self.extract_triplets(text)

        print(f"\n=== {self.name}: Извлечено {len(triplets)} триплетов ===")
        for i, triplet in enumerate(triplets, 1):
            print(f"{i}. {triplet}")
            if triplet.confidence:
                print(f"   Уверенность: {triplet.confidence:.3f}")

        return triplets

    def get_info(self) -> dict:
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'ready': self.is_ready()
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', ready={self.is_ready()})"
