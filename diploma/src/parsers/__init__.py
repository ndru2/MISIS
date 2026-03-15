"""
Модуль парсеров для извлечения триплетов.

Содержит базовый интерфейс и различные реализации синтаксических анализаторов.
"""

from .base_parser import BaseParser, Triplet, Token
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

try:
    from .bert_noun_parser import BertNounParser
    BERT_NOUN_AVAILABLE = True
except ImportError:
    BERT_NOUN_AVAILABLE = False
    BertNounParser = None

try:
    from .bert_token_parser import BertTokenParser
    BERT_TOKEN_AVAILABLE = True
except ImportError:
    BERT_TOKEN_AVAILABLE = False
    BertTokenParser = None

__all__ = [
    'BaseParser',
    'Triplet',
    'Token',
    'SpacyParser',
    'BiLSTMParser',
    'StanzaParser',
    'KeyBertParser',
    'BertNounParser',
    'BertTokenParser',
]
