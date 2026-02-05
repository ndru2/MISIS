from typing import List, Optional
from .base_parser import BaseParser, Triplet

SUPAR_AVAILABLE = False
SUPAR_ERROR = ""

try:
    from supar import Parser
    SUPAR_AVAILABLE = True
except ImportError as e:
    SUPAR_ERROR = f"ImportError: {str(e)}"
except ModuleNotFoundError as e:
    SUPAR_ERROR = f"ModuleNotFoundError: {str(e)}"
except Exception as e:
    SUPAR_ERROR = f"Exception: {str(e)}"


class BiLSTMParser(BaseParser):    
    def __init__(self, model_name: str = "biaffine-dep-xlmr"):
        super().__init__(name=f"BiLSTMParser({model_name})")
        self.model_name = model_name
        self.parser = None
        
        if not SUPAR_AVAILABLE:
            self.logger.error(f"Библиотека SuPar не доступна: {SUPAR_ERROR}")
            return

        self._load_model()

    def _load_model(self):
        try:
            self.logger.info(f"Загрузка модели SuPar '{self.model_name}'...")
            self.parser = Parser.load(self.model_name)
            self.logger.info(f"Модель SuPar '{self.model_name}' загружена успешно")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели SuPar: {e}")
            self.parser = None

    def is_ready(self) -> bool:
        return SUPAR_AVAILABLE and self.parser is not None

    def extract_triplets(self, text: str) -> List[Triplet]:
        if not self.is_ready():
            self.logger.error("Модель BiLSTM не готова к работе")
            return []
        

        sentences = self._split_sentences(text)

        triplets = []
        for sent_text in sentences:
            if not sent_text.strip():
                continue

            sent_triplets = self._extract_from_sentence(sent_text)
            triplets.extend(sent_triplets)
        
        self.logger.info(f"Извлечено {len(triplets)} триплетов из текста")
        return triplets
    
    def _split_sentences(self, text: str) -> List[str]:
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_from_sentence(self, sent_text: str) -> List[Triplet]:
        try:
            dataset = self.parser.predict([sent_text], verbose=False)

            triplets = []

            for sent_data in dataset:
                words = sent_data.values[1]
                heads = sent_data.values[6]
                deps = sent_data.values[7]

                dep_dict = {}

                for i, (word, head_idx, dep_type) in enumerate(zip(words, heads, deps)):
                    if dep_type in ('nsubj', 'nsubj:pass'):
                        if head_idx not in dep_dict:
                            dep_dict[head_idx] = {}
                        dep_dict[head_idx]['subj'] = (i, word)

                    elif dep_type in ('obj', 'dobj', 'iobj', 'obl'):
                        if head_idx not in dep_dict:
                            dep_dict[head_idx] = {}
                        dep_dict[head_idx]['obj'] = (i, word)

                for verb_idx, relations in dep_dict.items():
                    if 'subj' in relations and 'obj' in relations:
                        subj_idx, subj_word = relations['subj']
                        obj_idx, obj_word = relations['obj']

                        if 0 <= verb_idx - 1 < len(words):
                            verb_word = words[verb_idx - 1]

                            triplet = Triplet(
                                subject=subj_word.lower(),
                                predicate=verb_word.lower(),
                                object=obj_word.lower(),
                                subject_pos=subj_idx,
                                predicate_pos=verb_idx - 1,
                                object_pos=obj_idx,
                                sentence=sent_text.strip()
                            )
                            triplets.append(triplet)
                            
                            self.logger.debug(f"Найден триплет: {triplet}")
            
            return triplets
            
        except Exception as e:
            self.logger.error(f"Ошибка при парсинге предложения '{sent_text}': {e}")
            return []
    
    def get_info(self) -> dict:
        info = super().get_info()
        info['model'] = self.model_name
        info['library'] = 'SuPar'
        info['available'] = SUPAR_AVAILABLE
        return info

# if __name__ == "__main__":
#     # Тестовый запуск
#     parser = BiLSTMParser()
    
#     if parser.is_ready():
#         test_text = """
#         Студент изучает программирование. 
#         Программист создает приложения.
#         Алгоритм решает задачу.
#         """
        
#         triplets = parser.extract_and_display(test_text)
#         print(f"\nИнформация о парсере: {parser.get_info()}")
#     else:
#         print("BiLSTM парсер недоступен. Установите: pip install -U supar")
