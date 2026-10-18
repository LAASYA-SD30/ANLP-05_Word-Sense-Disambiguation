
from flask import Flask, render_template, request, jsonify
import nltk
import re
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Download required NLTK resources.
for resource in ["wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass


def tokenize(text):
    """Tokenize text using NLTK, with a regex fallback."""
    try:
        return nltk.word_tokenize(text.lower())
    except LookupError:
        return re.findall(r"[a-zA-Z]+", text.lower())


def get_relations(synset):
    """Retrieve WordNet lexical and semantic relations."""

    synonyms = sorted(set(
        lemma.name().replace("_", " ")
        for lemma in synset.lemmas()
    ))

    antonyms = sorted(set(
        antonym.name().replace("_", " ")
        for lemma in synset.lemmas()
        for antonym in lemma.antonyms()
    ))

    def describe(items):
        return [
            {
                "name": item.name(),
                "definition": item.definition()
            }
            for item in items[:10]
        ]

    return {
        "synonyms": synonyms,
        "antonyms": antonyms,
        "hypernyms": describe(synset.hypernyms()),
        "hyponyms": describe(synset.hyponyms()),
        "meronyms": describe(
            synset.part_meronyms() +
            synset.substance_meronyms()
        ),
        "holonyms": describe(
            synset.part_holonyms() +
            synset.substance_holonyms()
        ),
        "similar": describe(synset.similar_tos())
    }


def get_senses(word, pos=None):
    """Return all WordNet senses for a word."""

    word = word.strip().lower().replace(" ", "_")

    if not word:
        return []

    synsets = (
        wn.synsets(word, pos=pos)
        if pos else wn.synsets(word)
    )

    results = []

    for number, synset in enumerate(synsets, start=1):
        examples = synset.examples()

        results.append({
            "id": synset.name(),
            "name": synset.name(),
            "synset": synset.name(),
            "number": number,
            "definition": synset.definition(),
            "pos": synset.pos(),
            "pos_name": {
                "n": "Noun",
                "v": "Verb",
                "a": "Adjective",
                "s": "Adjective satellite",
                "r": "Adverb"
            }.get(synset.pos(), synset.pos()),
            "lemmas": [
                lemma.name().replace("_", " ")
                for lemma in synset.lemmas()
            ],
            "examples": examples,
            "relations": get_relations(synset),
            **get_relations(synset)
        })

    return results


def predict_lesk(sentence, word, method="extended"):
    """Predict a WordNet sense using a Lesk-style method."""

    tokens = tokenize(sentence)
    target = word.strip().lower().replace(" ", "_")

    if method == "simple":
        predicted = lesk(tokens, target)
    else:
        predicted = lesk(tokens, target, synsets=wn.synsets(target))

    candidates = wn.synsets(target)

    if predicted is None and candidates:
        predicted = candidates[0]

    if not candidates or predicted is None:
        return None, 0

    # Additional contextual overlap scoring.
    context = set(tokens)
    scored = []

    for synset in candidates:
        signature = set(tokenize(synset.definition()))

        for lemma in synset.lemmas():
            signature.update(
                tokenize(lemma.name().replace("_", " "))
            )

        for example in synset.examples():
            signature.update(tokenize(example))

        overlap = len(context.intersection(signature))

        scored.append((synset, overlap))

    if scored:
        best = max(scored, key=lambda item: item[1])

        if best[1] > 0:
            predicted = best[0]

    return predicted, len(context)


# Small labelled dataset for supervised demonstration.
TRAINING_DATA = {
    "bank": [
        ("She deposited money in the bank.", "bank.n.02"),
        ("The bank approved my loan.", "bank.n.02"),
        ("He cashed a cheque at the bank.", "bank.n.02"),
        ("The bank of the river was muddy.", "bank.n.01"),
        ("They sat on the bank beside the river.", "bank.n.01"),
        ("The boat reached the river bank.", "bank.n.01"),
    ],
    "bat": [
        ("The bat flew out of the cave at dusk.", "bat.n.01"),
        ("A bat hangs upside down in the cave.", "bat.n.01"),
        ("The nocturnal bat hunts insects.", "bat.n.01"),
        ("He hit the ball with a wooden bat.", "bat.n.02"),
        ("The cricket player bought a new bat.", "bat.n.02"),
        ("She swung the bat during the game.", "bat.n.02"),
    ],
    "plant": [
        ("The plant needs water and sunlight.", "plant.n.01"),
        ("She watered the plant in the garden.", "plant.n.01"),
        ("The plant grew beside the window.", "plant.n.01"),
        ("The factory plant manufactures cars.", "plant.n.02"),
        ("Workers entered the manufacturing plant.", "plant.n.02"),
        ("The industrial plant produces steel.", "plant.n.02"),
    ],
}


def train_supervised_model(word):
    """Train a TF-IDF and Logistic Regression classifier."""

    examples = TRAINING_DATA.get(word)

    if not examples:
        return None

    texts = [text for text, label in examples]
    labels = [label for text, label in examples]

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(max_iter=1000, random_state=42)
        )
    ])

    model.fit(texts, labels)

    return model


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health")
def health():
    try:
        count = len(wn.synsets("bank"))

        return jsonify({
            "status": "connected",
            "wordnet": True,
            "sample_senses": count
        })

    except LookupError:
        return jsonify({
            "status": "missing",
            "wordnet": False,
            "message": "WordNet corpus is unavailable."
        }), 503


@app.route("/api/senses", methods=["GET"])
def senses():
    word = request.args.get("word", "bank").strip()

    if not word:
        return jsonify({
            "success": False,
            "error": "Enter a word."
        }), 400

    try:
        results = get_senses(word)

        return jsonify({
            "success": True,
            "word": word,
            "total": len(results),
            "senses": results
        })

    except LookupError:
        return jsonify({
            "success": False,
            "error": "WordNet corpus is unavailable."
        }), 503


@app.route("/api/relations", methods=["GET"])
def relations():
    word = request.args.get("word", "bank").strip()
    sense_name = request.args.get("sense", "").strip()

    try:
        if sense_name:
            synset = wn.synset(sense_name)
        else:
            candidates = wn.synsets(word)

            if not candidates:
                return jsonify({
                    "success": False,
                    "error": "No WordNet senses found."
                }), 404

            synset = candidates[0]

        return jsonify({
            "success": True,
            "word": word,
            "sense": synset.name(),
            "definition": synset.definition(),
            "relations": get_relations(synset)
        })

    except (LookupError, ValueError) as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}

    sentence = data.get("sentence", "").strip()
    target = data.get("target", "").strip().lower()

    if not sentence or not target:
        return jsonify({
            "success": False,
            "error": "Enter both a sentence and a target word."
        }), 400

    tokens = tokenize(sentence)
    target_tokens = target.replace("_", " ").split()

    if not all(token in tokens for token in target_tokens):
        return jsonify({
            "success": False,
            "error": "The target word must appear in the sentence."
        }), 400

    try:
        predicted, _ = predict_lesk(sentence, target)

        if predicted is None:
            return jsonify({
                "success": False,
                "error": "No WordNet sense could be identified."
            }), 400

        all_senses = get_senses(target)
        selected = next(
            (
                sense for sense in all_senses
                if sense["id"] == predicted.name()
            ),
            None
        )

        return jsonify({
            "success": True,
            "result": {
                "sentence": sentence,
                "target": target,
                "predicted": selected,
                "total_senses": len(all_senses),
                "senses": all_senses,
                "relations": get_relations(predicted),
                "method": "Lesk algorithm + contextual overlap",
                "confidence_note": (
                    "This is a lexical-overlap baseline, "
                    "not a calibrated probability."
                )
            }
        })

    except LookupError:
        return jsonify({
            "success": False,
            "error": "WordNet corpus is unavailable."
        }), 503


@app.route("/api/disambiguate", methods=["POST"])
def disambiguate():
    data = request.get_json(silent=True) or {}

    sentence = data.get("sentence", "").strip()
    method = data.get("method", "extended").lower()

    if not sentence:
        return jsonify({
            "success": False,
            "error": "Enter sample text."
        }), 400

    tokens = tokenize(sentence)
    results = []

    # Identify candidate content words using WordNet.
    for word in tokens:
        if not word.isalpha() or len(word) < 2:
            continue

        candidates = wn.synsets(word)

        if not candidates:
            continue

        predicted, _ = predict_lesk(sentence, word, method)

        if predicted is None:
            continue

        context = set(tokens)
        signature = set(tokenize(predicted.definition()))

        for lemma in predicted.lemmas():
            signature.update(
                tokenize(lemma.name().replace("_", " "))
            )

        overlap = len(context.intersection(signature))

        results.append({
            "word": word,
            "pos": predicted.pos(),
            "predicted_synset": predicted.name(),
            "definition": predicted.definition(),
            "overlap_score": overlap
        })

    return jsonify({
        "success": True,
        "method": method,
        "results": results
    })


@app.route("/api/supervised", methods=["POST"])
def supervised():
    data = request.get_json(silent=True) or {}

    word = data.get("word", "").strip().lower()
    sentence = data.get("sentence", "").strip()

    if not word or not sentence:
        return jsonify({
            "success": False,
            "error": "Enter a target word and sentence."
        }), 400

    if word not in tokenize(sentence):
        return jsonify({
            "success": False,
            "error": "The target word must appear in the sentence."
        }), 400

    if word not in TRAINING_DATA:
        return jsonify({
            "success": False,
            "error": "Choose bank, bat, or plant."
        }), 400

    model = train_supervised_model(word)
    predicted = model.predict([sentence])[0]
    probabilities = model.predict_proba([sentence])[0]
    confidence = round(float(max(probabilities)) * 100, 2)

    try:
        definition = wn.synset(predicted).definition()
    except LookupError:
        definition = "Definition unavailable."

    return jsonify({
        "success": True,
        "word": word,
        "sentence": sentence,
        "sense": predicted,
        "definition": definition,
        "confidence": confidence,
        "method": "TF-IDF + Logistic Regression"
    })


@app.route("/api/evaluate", methods=["GET"])
def evaluate():
    samples = [
        {
            "sentence": "She deposited money in the bank.",
            "target": "bank",
            "expected": "bank.n.02"
        },
        {
            "sentence": "He sat on the bank beside the river.",
            "target": "bank",
            "expected": "bank.n.01"
        },
        {
            "sentence": "The bat flew out of the cave at dusk.",
            "target": "bat",
            "expected": "bat.n.01"
        },
        {
            "sentence": "He hit the ball with a wooden bat.",
            "target": "bat",
            "expected": "bat.n.02"
        },
    ]

    lesk_correct = 0
    supervised_correct = 0
    per_word = {}
    detailed = []

    for sample in samples:
        word = sample["target"]
        sentence = sample["sentence"]
        expected = sample["expected"]

        predicted, _ = predict_lesk(sentence, word, "extended")
        lesk_name = predicted.name() if predicted else ""

        if lesk_name == expected:
            lesk_correct += 1

        model = train_supervised_model(word)
        supervised_name = model.predict([sentence])[0]

        if supervised_name == expected:
            supervised_correct += 1

        word_stats = per_word.setdefault(
            word, {"correct": 0, "total": 0}
        )
        word_stats["total"] += 1

        if supervised_name == expected:
            word_stats["correct"] += 1

        detailed.append({
            "sentence": sentence,
            "target": word,
            "method": "Extended Lesk",
            "predicted_synset": lesk_name,
            "correct": lesk_name == expected
        })

    per_word_accuracy = {
        word: round(stats["correct"] / stats["total"] * 100, 2)
        for word, stats in per_word.items()
    }

    return jsonify({
        "success": True,
        "lesk_accuracy": {
            "Extended Lesk": round(
                lesk_correct / len(samples) * 100, 2
            )
        },
        "supervised_per_word": per_word_accuracy,
        "supervised_overall": round(
            supervised_correct / len(samples) * 100, 2
        ),
        "samples": detailed,
        "note": (
            "Evaluation uses four demonstration sentences. "
            "Results do not represent general-purpose WSD accuracy."
        )
    })


@app.route("/api/report", methods=["POST"])
def report():
    data = request.get_json(silent=True) or {}

    sentence = data.get("sentence", "").strip()
    target = data.get("target", "").strip().lower()

    if not sentence or not target:
        return jsonify({
            "success": False,
            "error": "Enter a sentence and target word."
        }), 400

    if target not in tokenize(sentence):
        return jsonify({
            "success": False,
            "error": "The target word must appear in the sentence."
        }), 400

    try:
        predicted, _ = predict_lesk(sentence, target)

        if predicted is None:
            return jsonify({
                "success": False,
                "error": "Unable to identify a sense."
            }), 400

        relations_data = get_relations(predicted)

        report_text = f"""
WORD SENSE DISAMBIGUATION — TASK 5
==================================

Input sentence:
{sentence}

Target word:
{target}

Method:
Lesk algorithm + contextual overlap

Predicted sense:
{predicted.name()}

Part of speech:
{predicted.pos()}

Definition:
{predicted.definition()}

Examples:
{', '.join(predicted.examples()) or 'No examples available.'}

Synonyms:
{', '.join(relations_data['synonyms']) or 'None found'}

Antonyms:
{', '.join(relations_data['antonyms']) or 'None found'}

Analysis:
Candidate meanings are retrieved from WordNet.
The system selects a likely sense using contextual lexical overlap.
The result is a baseline prediction, not a calibrated probability.
"""

        return jsonify({
            "success": True,
            "report": report_text.strip()
        })

    except LookupError:
        return jsonify({
            "success": False,
            "error": "WordNet corpus is unavailable."
        }), 503


if __name__ == "__main__":
    app.run(debug=True)
