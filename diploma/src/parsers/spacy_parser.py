import spacy
from typing import List, Optional
from .base_parser import BaseParser, Triplet


class SpacyParser(BaseParser):
    def __init__(self, model_name: str = "ru_core_news_sm"):
        super().__init__(name=f"SpacyParser({model_name})")
        self.model_name = model_name
        self.nlp = None

        self._load_model()

    def _load_model(self):
        try:
            self.nlp = spacy.load(self.model_name)
            self.logger.info(f"Модель spaCy '{self.model_name}' загружена успешно")
        except OSError:
            self.logger.error(
                f"Модель {self.model_name} не найдена. "
                f"Установите: python -m spacy download {self.model_name}"
            )
            self.nlp = None

    def is_ready(self) -> bool:
        return self.nlp is not None

    def extract_triplets(self, text: str) -> List[Triplet]:
        if not self.is_ready():
            self.logger.error("Модель spaCy не загружена")
            return []

        doc = self.nlp(text)
        triplets = []

        for sent in doc.sents:
            sent_triplets = self._extract_from_sentence(sent)
            triplets.extend(sent_triplets)

        self.logger.info(f"Извлечено {len(triplets)} триплетов из текста")
        return triplets

    def _extract_from_sentence(self, sent) -> List[Triplet]:
        triplets = []
        verbs = [token for token in sent if token.pos_ == "VERB"]

        for verb in verbs:
            # Прямые зависимости от глагола
            triplets.extend(self._extract_direct_dependencies(verb, sent))
            
            # Через предлоги 
            triplets.extend(self._extract_through_prepositions(verb, sent))
            
            # Составные глагольные конструкции
            triplets.extend(self._extract_compound_verbs(verb, sent))
            
            # Причастные обороты
            triplets.extend(self._extract_from_participles(verb, sent))

        # дубликаты
        unique_triplets = self._remove_duplicates(triplets)
        
        return unique_triplets
    
    def _extract_direct_dependencies(self, verb, sent) -> List[Triplet]:
        triplets = []
        subjects = []
        objects = []

        for child in verb.children:
            # Ищем субъекты (расширенный набор)
            if child.dep_ in ("nsubj", "nsubj:pass") and child.pos_ == "NOUN":
                subjects.append(child)
            
            # Ищем объекты (расширенный набор)
            elif child.dep_ in ("obj", "dobj", "iobj") and child.pos_ == "NOUN":
                objects.append(child)

        for subj in subjects:
            for obj in objects:
                triplet = Triplet(
                    subject=subj.lemma_.lower(),
                    predicate=verb.lemma_.lower(),
                    object=obj.lemma_.lower(),
                    subject_pos=subj.i,
                    predicate_pos=verb.i,
                    object_pos=obj.i,
                    sentence=sent.text.strip()
                )
                triplets.append(triplet)
                self.logger.debug(f"[Direct] {triplet}")

        return triplets
    
    def _extract_through_prepositions(self, verb, sent) -> List[Triplet]:

        triplets = []
        subjects = []
        objects = []

        for child in verb.children:
            # Субъекты
            if child.dep_ in ("nsubj", "nsubj:pass") and child.pos_ == "NOUN":
                subjects.append(child)
            
            # Объекты через предлоги
            elif child.dep_ == "obl":
                for grandchild in child.children:
                    if grandchild.pos_ == "NOUN":
                        objects.append(grandchild)
                if child.pos_ == "NOUN":
                    objects.append(child)

        for subj in subjects:
            for obj in objects:
                triplet = Triplet(
                    subject=subj.lemma_.lower(),
                    predicate=verb.lemma_.lower(),
                    object=obj.lemma_.lower(),
                    subject_pos=subj.i,
                    predicate_pos=verb.i,
                    object_pos=obj.i,
                    sentence=sent.text.strip()
                )
                triplets.append(triplet)
                self.logger.debug(f"[Preposition] {triplet}")

        return triplets
    
    def _extract_compound_verbs(self, verb, sent) -> List[Triplet]:
        triplets = []
        subjects = []
    
        for child in verb.children:
            if child.dep_ in ("nsubj", "nsubj:pass") and child.pos_ == "NOUN":
                subjects.append(child)
        
    
        for child in verb.children:
            if child.dep_ == "xcomp" and child.pos_ == "VERB":
    
                for grandchild in child.children:
                    if grandchild.dep_ in ("obj", "dobj") and grandchild.pos_ == "NOUN":
                        for subj in subjects:
                            triplet = Triplet(
                                subject=subj.lemma_.lower(),
                                predicate=child.lemma_.lower(),
                                object=grandchild.lemma_.lower(),
                                subject_pos=subj.i,
                                predicate_pos=child.i,
                                object_pos=grandchild.i,
                                sentence=sent.text.strip()
                            )
                            triplets.append(triplet)
                            self.logger.debug(f"[Compound] {triplet}")

        return triplets
    
    def _extract_from_participles(self, verb, sent) -> List[Triplet]:
        triplets = []
        
        if verb.dep_ == "amod":
            head_noun = verb.head
            if head_noun.pos_ == "NOUN":
                for child in verb.children:
                    if child.dep_ in ("obl", "obl:agent") and child.pos_ == "NOUN":
                        triplet = Triplet(
                            subject=child.lemma_.lower(),
                            predicate=verb.lemma_.lower(),
                            object=head_noun.lemma_.lower(),
                            subject_pos=child.i,
                            predicate_pos=verb.i,
                            object_pos=head_noun.i,
                            sentence=sent.text.strip()
                        )
                        triplets.append(triplet)
                        self.logger.debug(f"[Participle] {triplet}")

        return triplets

    def _remove_duplicates(self, triplets: List[Triplet]) -> List[Triplet]:
        seen = set()
        unique = []

        for triplet in triplets:
            key = (triplet.subject, triplet.predicate, triplet.object)
            if key not in seen:
                seen.add(key)
                unique.append(triplet)

        return unique

    def get_info(self) -> dict:
        info = super().get_info()
        info['model'] = self.model_name
        if self.nlp:
            info['pipeline'] = self.nlp.pipe_names
        return info
