import re
import nltk
import pandas as pd

for pkg in ["wordnet", "omw-1.4", "stopwords", "punkt", "punkt_tab",
            "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(pkg, quiet=True)

from nltk.corpus import wordnet as wn, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

LEMMATIZER = WordNetLemmatizer()
STOP = set(stopwords.words("english"))
POS_MAP = {"J": wn.ADJ, "V": wn.VERB, "N": wn.NOUN, "R": wn.ADV}

TARGET_WORDS = ["bank", "bat", "plant", "book", "mouse"]

# (sentence, target word, POS, keyword expected in the correct gloss)
SAMPLES = [
    ("He deposited the money in the bank before noon.", "bank", "n", "financ"),
    ("They sat on the bank of the river and watched the water.", "bank", "n", "sloping"),
    ("The bat flew out of the cave at night.", "bat", "n", "mammal"),
    ("He swung the bat and hit the ball.", "bat", "n", "club"),
    ("The plant needs sunlight and water to grow.", "plant", "n", "organism"),
    ("The steel plant employs thousands of workers.", "plant", "n", "industrial"),
    ("She read a book about ancient history.", "book", "n", "written"),
    ("Please book a table for two at the restaurant.", "book", "v", "reserve"),
    ("The mouse ran across the kitchen floor.", "mouse", "n", "rodent"),
    ("Click the left button on the mouse.", "mouse", "n", "device"),
]

# Labelled data for supervised WSD (scikit-learn)
TRAIN = {
    "bank": {
        "money": ["He deposited the money in the bank before noon.",
                  "The bank approved my loan application.",
                  "She opened a savings account at the bank.",
                  "The bank charges interest on every loan."],
        "river": ["They sat on the bank of the river.",
                  "Fishermen stood on the muddy bank of the stream.",
                  "The river overflowed its bank after the storm.",
                  "Grass grew along the steep bank of the canal."],
    },
    "bat": {
        "animal": ["The bat flew out of the cave at night.",
                   "A bat hung upside down from the branch.",
                   "The bat uses echolocation to catch insects.",
                   "Bats sleep in dark caves during the day."],
        "sports": ["He swung the bat and hit the ball.",
                   "The cricket bat was made of willow.",
                   "She gripped the bat before the pitch.",
                   "The batter dropped the bat and ran to first base."],
    },
    "plant": {
        "organism": ["The plant needs sunlight and water to grow.",
                     "She watered the plant on the windowsill.",
                     "This plant has green leaves and small flowers.",
                     "The seed grew into a tall plant."],
        "factory": ["The steel plant employs thousands of workers.",
                    "The power plant supplies electricity to the city.",
                    "The chemical plant was shut down for safety checks.",
                    "Workers at the manufacturing plant went on strike."],
    },
    "mouse": {
        "animal": ["The mouse ran across the kitchen floor.",
                   "A cat chased the mouse into a hole.",
                   "The mouse nibbled the cheese.",
                   "A small grey mouse hid in the barn."],
        "device": ["Click the left button on the mouse.",
                   "He connected a wireless mouse to the laptop.",
                   "Move the mouse cursor over the icon.",
                   "The optical mouse stopped tracking on the desk."],
    },
}
SUPERVISED_WORDS = list(TRAIN)


# ---------- Sense inventory and lexical relations ----------
def sense_details(word):
    out = []
    for s in wn.synsets(word):
        out.append({
            "synset": s.name(),
            "pos": s.pos(),
            "definition": s.definition(),
            "examples": s.examples(),
            "synonyms": [l.name() for l in s.lemmas()],
            "antonyms": [a.name() for l in s.lemmas() for a in l.antonyms()],
            "hypernyms": [h.name() for h in s.hypernyms()],
            "hyponyms": [h.name() for h in s.hyponyms()[:5]],
            "meronyms": [m.name() for m in s.part_meronyms()],
        })
    return out


def build_sense_inventory(words):
    rows = []
    for w in words:
        for d in sense_details(w):
            rows.append({
                "word": w,
                "synset": d["synset"],
                "pos": d["pos"],
                "definition": d["definition"],
                "examples": " | ".join(d["examples"]),
                "synonyms": ", ".join(d["synonyms"]),
                "antonyms": ", ".join(d["antonyms"]),
                "hypernyms": ", ".join(d["hypernyms"]),
                "hyponyms": ", ".join(d["hyponyms"]),
                "part_meronyms": ", ".join(d["meronyms"]),
            })
    return pd.DataFrame(rows)


# ---------- Knowledge-based WSD (Lesk) ----------
def clean(text, pos=None):
    out = []
    for t in re.findall(r"[a-z]+", text.lower()):
        if t in STOP or len(t) < 3:
            continue
        out.append(LEMMATIZER.lemmatize(t, pos) if pos else LEMMATIZER.lemmatize(t))
    return out


def signature(syn, extended):
    text = [syn.definition()] + syn.examples()
    if extended:
        related = syn.hypernyms() + syn.hyponyms() + syn.part_meronyms() + syn.part_holonyms()
        for r in related:
            text.append(r.definition())
            text += r.examples()
        text += [l.name().replace("_", " ") for l in syn.lemmas()]
    return set(clean(" ".join(text)))


def lesk(word, pos, context, extended=False):
    senses = wn.synsets(word, pos)
    if not senses:
        return None, 0
    best, best_score = senses[0], 0
    for s in senses:
        score = len(signature(s, extended) & context)
        if score > best_score:
            best, best_score = s, score
    return best, best_score


def first_sense(word, pos):
    senses = wn.synsets(word, pos)
    return (senses[0], 0) if senses else (None, 0)


def context_words(sentence, exclude=None):
    ctx = set()
    for tok, tag in nltk.pos_tag(word_tokenize(sentence)):
        wpos = POS_MAP.get(tag[0])
        if wpos and tok.isalpha() and tok.lower() not in STOP:
            lem = LEMMATIZER.lemmatize(tok.lower(), wpos)
            if lem != exclude:
                ctx.add(lem)
    return ctx


def disambiguate_sentence(sentence, method="extended_lesk"):
    """All-words WSD: predict a sense for every content word."""
    rows = []
    for tok, tag in nltk.pos_tag(word_tokenize(sentence)):
        wpos = POS_MAP.get(tag[0])
        if not wpos or not tok.isalpha() or tok.lower() in STOP:
            continue
        lemma = LEMMATIZER.lemmatize(tok.lower(), wpos)
        if method == "first_sense":
            syn, score = first_sense(lemma, wpos)
        else:
            ctx = context_words(sentence, exclude=lemma)
            syn, score = lesk(lemma, wpos, ctx, extended=(method == "extended_lesk"))
        if syn:
            rows.append({"sentence": sentence, "word": tok, "pos": wpos,
                         "predicted_synset": syn.name(),
                         "definition": syn.definition(),
                         "overlap_score": score})
    return rows


def evaluate_lesk():
    rows = []
    for sent, word, pos, hint in SAMPLES:
        ctx = context_words(sent, exclude=word)
        preds = {
            "first_sense": first_sense(word, pos),
            "simple_lesk": lesk(word, pos, ctx, extended=False),
            "extended_lesk": lesk(word, pos, ctx, extended=True),
        }
        for method, (syn, score) in preds.items():
            rows.append({"sentence": sent, "target": word, "method": method,
                         "predicted_synset": syn.name(),
                         "definition": syn.definition(),
                         "overlap_score": score,
                         "correct": hint in syn.definition().lower()})
    results = pd.DataFrame(rows)
    accuracy = results.groupby("method")["correct"].mean().mul(100).round(1)
    return results, accuracy


# ---------- Supervised WSD (scikit-learn) ----------
def _xy(word):
    X, y = [], []
    for label, sents in TRAIN[word].items():
        X += sents
        y += [label] * len(sents)
    return X, y


def _make_model():
    return make_pipeline(TfidfVectorizer(stop_words="english", ngram_range=(1, 2)),
                         LogisticRegression(max_iter=1000))


_MODELS = {}


def supervised_predict(word, sentence):
    if word not in TRAIN:
        raise ValueError(f"No training data for '{word}'")
    if word not in _MODELS:
        X, y = _xy(word)
        _MODELS[word] = _make_model().fit(X, y)
    model = _MODELS[word]
    label = model.predict([sentence])[0]
    confidence = float(model.predict_proba([sentence])[0].max())
    return label, round(confidence * 100, 1)


def supervised_cv_accuracy():
    """4-fold cross-validated accuracy for each word."""
    per_word = {}
    for word in TRAIN:
        X, y = _xy(word)
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        pred = cross_val_predict(_make_model(), X, y, cv=cv)
        per_word[word] = round(sum(p == t for p, t in zip(pred, y)) / len(y) * 100, 1)
    overall = round(sum(per_word.values()) / len(per_word), 1)
    return per_word, overall