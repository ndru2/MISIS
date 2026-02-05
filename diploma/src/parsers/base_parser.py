from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        pass

    @abstractmethod
    def is_ready(self) -> bool:
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
