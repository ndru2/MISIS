from typing import List, Dict, Set
from .base_parser import BaseParser, Triplet
from collections import defaultdict

KEYBERT_AVAILABLE = False
STANZA_AVAILABLE = False
KEYBERT_ERROR = ""
STANZA_ERROR = ""

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError as e:
    KEYBERT_ERROR = f"ImportError: {str(e)}"
except Exception as e:
    KEYBERT_ERROR = f"Exception: {str(e)}"

try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError as e:
    STANZA_ERROR = f"ImportError: {str(e)}"
except Exception as e:
    STANZA_ERROR = f"Exception: {str(e)}"


class KeyBertParser(BaseParser):
    def __init__(
        self,
        lang: str = "ru",
        top_n: int = 150,
        diversity: float = 0.6,
        min_keyword_score: float = 0.2,
        ngram_range: tuple = (1, 2)
    ):
        super().__init__(name=f"KeyBertParser({lang})")
        
        self.lang = lang
        self.top_n = top_n
        self.diversity = diversity
        self.min_keyword_score = min_keyword_score
        self.ngram_range = ngram_range
        
        self.kw_model = None
        self.nlp = None
        
        self.key_concepts: Set[str] = set()
        self.keyword_data: List[Dict] = []
        
        self._load_models()
    
    def _load_models(self):
        if not KEYBERT_AVAILABLE:
            self.logger.error(f"KeyBERT недоступен: {KEYBERT_ERROR}")
            return
        
        if not STANZA_AVAILABLE:
            self.logger.error(f"Stanza недоступна: {STANZA_ERROR}")
            return
        
        try:
            self.logger.info("Загрузка KeyBERT...")
            self.kw_model = KeyBERT()
            self.logger.info("KeyBERT загружен успешно")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки KeyBERT: {e}")
            self.kw_model = None
        
        try:
            self.logger.info(f"Загрузка модели Stanza для '{self.lang}'...")
            self.nlp = stanza.Pipeline(
                lang=self.lang,
                processors='tokenize,pos,lemma,depparse',
                use_gpu=False,
                verbose=False
            )
            self.logger.info(f"Модель Stanza '{self.lang}' загружена успешно")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели Stanza: {e}")
            self.nlp = None
    
    def is_ready(self) -> bool:
        return (KEYBERT_AVAILABLE and STANZA_AVAILABLE and 
                self.kw_model is not None and self.nlp is not None)
    
    def extract_triplets(self, text: str) -> List[Triplet]:
        if not self.is_ready():
            self.logger.error("Модели не готовы к работе")
            return []
        
        self.logger.info("Извлечение ключевых концепций с KeyBERT...")
        self._extract_key_concepts(text)
        
        if not self.key_concepts:
            self.logger.warning("Не удалось извлечь ключевые концепции")
            return []
        
        self.logger.info(f"Извлечено {len(self.key_concepts)} уникальных концепций")
        
        self.logger.info("Извлечение триплетов с Stanza...")
        
        try:
            doc = self.nlp(text)
            all_triplets = []
            
            for sent in doc.sentences:
                sent_triplets = self._extract_from_sentence(sent)
                all_triplets.extend(sent_triplets)
            
            self.logger.info(f"Stanza извлекла {len(all_triplets)} триплетов")
            
            self.logger.info("Фильтрация триплетов по ключевым концепциям...")
            filtered_triplets = self._filter_by_key_concepts(all_triplets)
            
            filter_ratio = len(filtered_triplets) / len(all_triplets) * 100 if all_triplets else 0
            self.logger.info(f"После фильтрации осталось {len(filtered_triplets)} триплетов ({filter_ratio:.1f}%)")
            
            return filtered_triplets
        
        except Exception as e:
            self.logger.error(f"Ошибка при обработке текста: {e}")
            return []
    
    def _extract_key_concepts(self, text: str) -> None:
        try:
            raw_keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=self.ngram_range,
                stop_words=None,
                top_n=self.top_n * 2,
                diversity=self.diversity,
                use_mmr=True
            )
            
            lemma_groups = {}
            
            for keyword, score in raw_keywords:
                if score < self.min_keyword_score:
                    continue
                
                doc = self.nlp(keyword)
                
                if not doc.sentences or not doc.sentences[0].words:
                    continue
                
                lemmas = [word.lemma.lower() for word in doc.sentences[0].words]
                lemma = " ".join(lemmas)
                
                if lemma not in lemma_groups:
                    lemma_groups[lemma] = {
                        'score': score,
                        'forms': [keyword],
                        'original': keyword
                    }
                else:
                    if score > lemma_groups[lemma]['score']:
                        lemma_groups[lemma]['score'] = score
                    lemma_groups[lemma]['forms'].append(keyword)
            
            sorted_concepts = sorted(
                lemma_groups.items(),
                key=lambda x: x[1]['score'],
                reverse=True
            )[:self.top_n]
            
            self.key_concepts = set(concept[0] for concept in sorted_concepts)
            self.keyword_data = [
                {
                    'lemma': lemma,
                    'score': data['score'],
                    'forms': data['forms'],
                    'original': data['original']
                }
                for lemma, data in sorted_concepts
            ]
            
            self.logger.debug("Топ-10 ключевых концепций:")
            for i, kw in enumerate(self.keyword_data[:10], 1):
                forms_str = ', '.join(kw['forms'][:3])
                self.logger.debug(f"  {i}. {kw['lemma']} (score: {kw['score']:.3f}, формы: {forms_str})")
        
        except Exception as e:
            self.logger.error(f"Ошибка при извлечении ключевых концепций: {e}")
            self.key_concepts = set()
            self.keyword_data = []
    
    def _extract_from_sentence(self, sent) -> List[Triplet]:
        triplets = []
        words_by_id = {word.id: word for word in sent.words}
        sent_text = sent.text
        
        phrases = defaultdict(lambda: {'subj': [], 'verbs': [], 'obj': []})
        xcomp_chain = {}
        
        for word in sent.words:
            if word.head not in words_by_id:
                continue
            
            head_word = words_by_id[word.head]
            
            if head_word.upos == 'VERB':
                verb_id = head_word.id
                
                if word.deprel in ('nsubj', 'nsubj:pass') and word.upos == 'NOUN':
                    phrases[verb_id]['subj'].append(word)
                    phrases[verb_id]['verbs'].append(head_word)
                
                elif word.deprel in ('obj', 'iobj') and word.upos == 'NOUN':
                    phrases[verb_id]['obj'].append(word)
                    phrases[verb_id]['verbs'].append(head_word)
                
                elif word.deprel == 'obl' and word.upos == 'NOUN':
                    phrases[verb_id]['obj'].append(word)
                    phrases[verb_id]['verbs'].append(head_word)
        
        for word in sent.words:
            if word.head not in words_by_id:
                continue
            
            head_word = words_by_id[word.head]
            
            if word.deprel == 'xcomp' and word.upos == 'VERB' and head_word.upos == 'VERB':
                main_verb_id = head_word.id
                
                if main_verb_id not in xcomp_chain:
                    xcomp_chain[main_verb_id] = main_verb_id
                initial_verb_id = xcomp_chain[main_verb_id]
                xcomp_chain[word.id] = initial_verb_id
                
                phrases[initial_verb_id]['verbs'].append(word)
                
                if word.id in phrases:
                    phrases[initial_verb_id]['obj'].extend(phrases[word.id]['obj'])
            
            if word.upos == 'VERB' and word.deprel == 'amod' and head_word.upos == 'NOUN':
                verb_id = word.id
                phrases[verb_id]['obj'].append(head_word)
                phrases[verb_id]['verbs'].append(word)
                
                for child in sent.words:
                    if child.head == verb_id and child.deprel in ('obl', 'obl:agent') and child.upos == 'NOUN':
                        phrases[verb_id]['subj'].append(child)
        
        for word in sent.words:
            if word.head not in words_by_id:
                continue
            head_word = words_by_id[word.head]
            
            if word.deprel == 'conj' and word.upos == 'VERB' and head_word.upos == 'VERB':
                main_verb_id = head_word.id
                
                initial_verb_id = xcomp_chain.get(main_verb_id, main_verb_id)
                
                if not phrases[word.id]['subj'] and phrases[initial_verb_id]['subj']:
                    phrases[word.id]['subj'] = phrases[initial_verb_id]['subj']
                    phrases[word.id]['verbs'].append(word)
                    
                    if initial_verb_id != main_verb_id:
                        phrases[word.id]['verbs'].extend(phrases[initial_verb_id]['verbs'])
            
            if word.deprel == 'conj' and word.upos == 'NOUN' and head_word.upos == 'NOUN':
                for verb_id, phrase_data in phrases.items():
                    if head_word in phrase_data['subj']:
                        phrase_data['subj'].append(word)
                    elif head_word in phrase_data['obj']:
                        phrase_data['obj'].append(word)
        
        for verb_id, phrase_data in phrases.items():
            subjects = phrase_data['subj']
            verbs = phrase_data['verbs']
            objects = phrase_data['obj']
            
            if not subjects or not verbs or not objects:
                continue
            
            for subj in subjects:
                for obj in objects:
                    main_verb = min(verbs, key=lambda v: v.id)
                    triplet = Triplet(
                        subject=subj.lemma.lower(),
                        predicate=main_verb.lemma.lower(),
                        object=obj.lemma.lower(),
                        subject_pos=subj.id - 1,
                        predicate_pos=main_verb.id - 1,
                        object_pos=obj.id - 1,
                        sentence=sent_text
                    )
                    triplets.append(triplet)
        
        unique_triplets = self._remove_duplicates(triplets)
        
        return unique_triplets
    
    def _remove_duplicates(self, triplets: List[Triplet]) -> List[Triplet]:
        seen = set()
        unique = []
        
        for triplet in triplets:
            key = (triplet.subject, triplet.predicate, triplet.object)
            if key not in seen:
                seen.add(key)
                unique.append(triplet)
        
        return unique
    
    def _filter_by_key_concepts(self, triplets: List[Triplet]) -> List[Triplet]:
        filtered = []
        
        for triplet in triplets:
            subject_is_key = self._is_key_concept(triplet.subject)
            object_is_key = self._is_key_concept(triplet.object)
            
            if subject_is_key or object_is_key:
                filtered.append(triplet)
                
                reasons = []
                if subject_is_key:
                    reasons.append(f"subj={triplet.subject}")
                if object_is_key:
                    reasons.append(f"obj={triplet.object}")
                
                self.logger.debug(f"[{', '.join(reasons)}] ({triplet.subject}, {triplet.predicate}, {triplet.object})")
        
        return filtered
    
    def _is_key_concept(self, lemma: str) -> bool:
        lemma_lower = lemma.lower()
        
        if lemma_lower in self.key_concepts:
            return True
        
        for concept in self.key_concepts:
            if ' ' in concept:
                if lemma_lower in concept or concept in lemma_lower:
                    return True
            else:
                if concept in lemma_lower.split():
                    return True
        
        return False
    
    def get_key_concepts(self) -> List[Dict]:
        return self.keyword_data
    
    def get_info(self) -> dict:
        info = super().get_info()
        info['approach'] = 'Hybrid: KeyBERT + Stanza'
        info['keybert_params'] = {
            'top_n': self.top_n,
            'diversity': self.diversity,
            'min_score': self.min_keyword_score,
            'ngram_range': self.ngram_range
        }
        info['language'] = self.lang
        info['num_key_concepts'] = len(self.key_concepts)
        info['keybert_available'] = KEYBERT_AVAILABLE
        info['stanza_available'] = STANZA_AVAILABLE
        return info
    