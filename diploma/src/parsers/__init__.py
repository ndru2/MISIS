"""
Модуль парсеров для извлечения триплетов.

Содержит базовый интерфейс и различные реализации синтаксических анализаторов.
"""

from .base_parser import BaseParser, Triplet
from .spacy_parser import SpacyParser
from .bilstm_parser import BiLSTMParser

try:
    from .stanza_parser import StanzaParser
    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False
    StanzaParser = None

try:
    from .keybert_parser import KeyBertParser
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    KeyBertParser = None

__all__ = [
    'BaseParser',
    'Triplet',
    'SpacyParser',
    'BiLSTMParser',
    'StanzaParser',
    'KeyBertParser',
]
