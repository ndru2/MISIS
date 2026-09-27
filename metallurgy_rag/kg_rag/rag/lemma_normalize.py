from __future__ import annotations

import re

_morph = None


def _get_morph():
    global _morph
    if _morph is None:
        try:
            import inspect

            if not hasattr(inspect, "getargspec"):
                def _getargspec(func):
                    spec = inspect.getfullargspec(func)
                    return spec.args, spec.varargs, spec.varkw, spec.defaults

                setattr(inspect, "getargspec", _getargspec)
            import pymorphy2

            _morph = pymorphy2.MorphAnalyzer()
        except Exception:
            _morph = False
    return _morph


def normal_form(word):
    text = (word or "").strip().lower()
    if not re.fullmatch(r"[а-яё-]+", text):
        return text
    morph = _get_morph()
    if not morph:
        return text
    try:
        return morph.parse(text)[0].normal_form
    except Exception:
        return text


STOP = {
    "какой", "какая", "какие", "каков", "сколько", "что", "как", "где", "когда",
    "почему", "зачем", "чем", "это", "быть", "мочь", "надо", "нужно", "при",
    "для", "или", "и", "в", "на", "с", "по", "из", "от", "до", "за", "не",
    "со", "во", "ко", "об", "обо", "о", "у", "к", "же", "ли",
    "the", "a", "an", "is", "are", "what", "which", "how", "why", "when",
    "where", "of", "in", "on", "for", "to", "and", "or", "be", "does", "do",
}


LIGHT_PHRASES = re.compile(
    r"\b(?:в качестве|в результате|в течение|в ходе|в случае|в процессе|за счёт|за счет|"
    r"с помощью|по сравнению с|в зависимости от|в связи с|"
    r"as a result of|in terms of|by means of|in the course of|due to)\b",
    re.IGNORECASE)


def query_terms(question):
    question = LIGHT_PHRASES.sub(" ", question or "")
    raw = re.findall(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9_·%°-]*", question)
    out, seen = [], set()
    for word in raw:
        forms = [word.lower()]
        if re.search(r"[A-Z]", word) and re.search(r"[0-9A-Z]", word[1:]):
            forms.insert(0, word)
        lemma = normal_form(word)
        if lemma and lemma not in forms:
            forms.append(lemma)
        for form in forms:
            if len(form) < 2 or form in STOP or form in seen:
                continue
            seen.add(form)
            out.append(form)
    return out
