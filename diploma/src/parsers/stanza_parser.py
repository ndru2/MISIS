from typing import List, Optional
from .base_parser import BaseParser, Triplet

STANZA_AVAILABLE = False
STANZA_ERROR = ""

try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError as e:
    STANZA_ERROR = f"ImportError: {str(e)}"
except Exception as e:
    STANZA_ERROR = f"Exception: {str(e)}"


class StanzaParser(BaseParser):
    def __init__(self, lang: str = "ru"):
        super().__init__(name=f"StanzaParser({lang})")
        self.lang = lang
        self.nlp = None

        self._load_model()

    def _load_model(self):
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
        return STANZA_AVAILABLE and self.nlp is not None

    def extract_triplets(self, text: str) -> List[Triplet]:
        if not self.is_ready():
            self.logger.error("Модель Stanza не готова к работе")
            return []

        try:

            doc = self.nlp(text)

            triplets = []
            for sent in doc.sentences:
                sent_triplets = self._extract_from_sentence(sent)
                triplets.extend(sent_triplets)

            self.logger.info(f"Извлечено {len(triplets)} триплетов из текста")
            return triplets

        except Exception as e:
            self.logger.error(f"Ошибка при обработке текста: {e}")
            return []

    def _extract_from_sentence(self, sent) -> List[Triplet]:
        from collections import defaultdict

        triplets = []
        words_by_id = {word.id: word for word in sent.words}
        sent_text = sent.text

        phrases = defaultdict(lambda: {'subj': [], 'verbs': [], 'obj': []})
        xcomp_chain = {}

        # базовые зависимости
        for word in sent.words:
            if word.head not in words_by_id:
                continue

            head_word = words_by_id[word.head]


            if head_word.upos == 'VERB':
                verb_id = head_word.id

                # Субъекты
                if word.deprel in ('nsubj', 'nsubj:pass') and word.upos == 'NOUN':
                    phrases[verb_id]['subj'].append(word)
                    phrases[verb_id]['verbs'].append(head_word)

                # объекты
                elif word.deprel in ('obj', 'iobj') and word.upos == 'NOUN':
                    phrases[verb_id]['obj'].append(word)
                    phrases[verb_id]['verbs'].append(head_word)

                # Объекты через предлоги
                elif word.deprel == 'obl' and word.upos == 'NOUN':
                    phrases[verb_id]['obj'].append(word)
                    phrases[verb_id]['verbs'].append(head_word)

        # составные глаголы, причастия
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

            # Причастные обороты
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

        # создание триплетов
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
                    self.logger.debug(f"[Hybrid] {triplet}")

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

    def get_info(self) -> dict:
        info = super().get_info()
        info['language'] = self.lang
        info['library'] = 'Stanza (Stanford NLP)'
        info['architecture'] = 'BiLSTM'
        info['available'] = STANZA_AVAILABLE
        return info
