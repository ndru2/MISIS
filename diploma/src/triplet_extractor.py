from typing import List, Tuple, Union, Optional
import logging
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from parsers import BaseParser, Triplet, SpacyParser, BiLSTMParser, StanzaParser, KeyBertParser


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TripletExtractor:    
    @staticmethod
    def _get_available_backends(): 
        backends = {
            'spacy': SpacyParser,
            'bilstm': BiLSTMParser,
            'stanza': StanzaParser,
            'keybert': KeyBertParser,
        }

        return backends

    def __init__(self, backend: str = "spacy", **kwargs):
        self.backend_name = backend.lower()

        AVAILABLE_BACKENDS = self._get_available_backends()

        if self.backend_name not in AVAILABLE_BACKENDS:
            available = ', '.join(AVAILABLE_BACKENDS.keys())
            raise ValueError(
                f"Backend '{backend}' не поддерживается. "
                f"Доступные: {available}"
            )

        parser_class = AVAILABLE_BACKENDS[self.backend_name]
        self.parser: BaseParser = parser_class(**kwargs)

        if not self.parser.is_ready():
            logger.warning(
                f"Парсер '{self.backend_name}' не готов к работе. "
                f"Проверьте установку зависимостей."
            )
        else:
            logger.info(f"Инициализирован экстрактор с backend: {self.backend_name}")

    def extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        triplet_objects = self.parser.extract_triplets(text)

        return [t.to_tuple() for t in triplet_objects]

    def extract_triplets_detailed(self, text: str) -> List[Triplet]:
        return self.parser.extract_triplets(text)

    def extract_and_display(self, text: str) -> List[Tuple[str, str, str]]:
        triplet_objects = self.parser.extract_and_display(text)
        return [t.to_tuple() for t in triplet_objects]

    def get_parser_info(self) -> dict:
        return self.parser.get_info()

    @classmethod
    def list_backends(cls) -> List[str]:
        return list(cls._get_available_backends().keys())

    def is_ready(self) -> bool:
        return self.parser.is_ready()

    def __repr__(self) -> str:
        return f"TripletExtractor(backend='{self.backend_name}', ready={self.is_ready()})"
