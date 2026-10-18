
from flask import Flask, render_template, request, jsonify
import nltk
import pandas as pd
import re
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Download the required NLTK resources if missing.
for resource in ["wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass


def tokenize(text):
    """Tokenize text using NLTK, with a fallback."""
    try:
        return word_tokenize(text.lower())
    except LookupError:
        return re.findall(r"[a-zA-Z]+", text.lower())


def get_senses(word, pos=None):
    """Return WordNet senses for the given target word."""
    word = word.strip().lower().replace(" ", "_")

    if not word:
        return []

    synsets = wn.synsets(word, pos=pos) if pos else wn.synsets(word)

    results = []

    for index, synset in enumerate(synsets, start=1):
        lemmas = [lemma.name().replace("_", " ")
                  for lemma in synset.lemmas()]

        examples = synset.examples()

        results.append({
            "id": synset.name(),
            "name": synset.name(),
            "number": index,
            "definition": synset.definition(),
            "pos": synset.pos(),
            "pos_name": {
                "n": "Noun",
                "v": "Verb",
                "a": "Adjective",
                "s": "Adjective satellite",
                "r": "Adverb"
            }.get(synset.pos(), synset.pos()),
            "lemmas": lemmas,
            "examples": examples,
            "example_text": examples[0] if examples else
                            "No example sentence available."
        })

    return results


def lexical_relations(synset):
    """Extract useful WordNet lexical and semantic relations."""
    hypernyms = synset.hypernyms()
    hyponyms = synset.hyponyms()
    similar = synset.similar_tos()
    meronyms = synset.part_meronyms() + synset.substance_meronyms()
    holonyms = synset.part_holonyms() + synset.substance_holonyms()

    synonyms = sorted(set(
        lemma.name().replace("_", " ")
        for lemma in synset.lemmas()
    ))

    antonyms = sorted(set(
        antonym.name().replace("_", " ")
        for lemma in synset.lemmas()
        for antonym in lemma.antonyms()
    ))

    return {
        "synonyms": synonyms,
        "antonyms": antonyms,
        "hypernyms": [
            {"name": s.name(), "definition": s.definition()}
            for s in hypernyms[:10]
        ],
        "hyponyms": [
            {"name": s.name(), "definition": s.definition()}
            for s in hyponyms[:10]
        ],
        "similar": [
            {"name": s.name(), "definition": s.definition()}
            for s in similar[:10]
        ],
        "meronyms": [
            {"name": s.name(), "definition": s.definition()}
            for s in meronyms[:10]
        ],
        "holonyms": [
            {"name": s.name(), "definition": s.definition()}
            for s in holonyms[:10]
        ]
    }


def analyze_context(sentence, target):
    """Disambiguate a target word using NLTK's Lesk algorithm."""
    sentence = sentence.strip()
    target = target.strip().lower()

    if not sentence or not target:
        raise ValueError("Enter both a sentence and a target word.")

    tokens = tokenize(sentence)
    target_tokens = target.replace("_", " ").split()

    if not all(token in tokens for token in target_tokens):
        raise ValueError(
            "The target word must appear in the sentence."
        )

    # Find a WordNet lemma and run the Lesk baseline.
    predicted = lesk(tokens, target.replace(" ", "_"))

    # Fall back to the first available sense if Lesk returns no result.
    if predicted is None:
        candidates = wn.synsets(target.replace(" ", "_"))
        if candidates:
            predicted = candidates[0]

    all_senses = get_senses(target)

    if not all_senses:
        raise ValueError(
            f"No WordNet senses were found for '{target}'."
        )

    # Score every candidate using overlap between context words
    # and the candidate's definition, lemmas and examples.
    context_words = set(tokens) - set(target_tokens)
    scored = []

    for sense in wn.synsets(target.replace(" ", "_")):
        signature = set(tokenize(sense.definition()))

        for lemma in sense.lemmas():
            signature.update(tokenize(lemma.name().replace("_", " ")))

        for example in sense.examples():
            signature.update(tokenize(example))

        overlap = len(context_words.intersection(signature))

        scored.append({
            "sense": sense,
            "overlap": overlap
        })

    # Use the highest-overlap sense when the overlap method
    # has evidence; otherwise retain the Lesk prediction.
    if scored:
        best = max(scored, key=lambda item: item["overlap"])

        if best["overlap"] > 0:
            predicted = best["sense"]

    if predicted is None:
        raise ValueError("Unable to identify a WordNet sense.")

    selected = next(
        (sense for sense in all_senses
         if sense["id"] == predicted.name()),
        all_senses[0]
    )

    selected_relations = lexical_relations(predicted)

    return {
        "sentence": sentence,
        "target": target,
        "predicted": selected,
        "confidence_note": (
            "The prediction is a Lesk-style lexical overlap baseline, "
            "not a calibrated probability."
        ),
        "method": "Lesk algorithm + contextual overlap",
        "total_senses": len(all_senses),
        "senses": all_senses,
        "relations": selected_relations
    }


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
            "message": "WordNet corpus is not available."
        }), 503


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}

    try:
        result = analyze_context(
            data.get("sentence", ""),
            data.get("target", "")
        )
        return jsonify({"success": True, "result": result})
    except (ValueError, LookupError) as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 400
    except Exception as error:
        app.logger.exception("Analysis failed")
        return jsonify({
            "success": False,
            "error": f"Analysis failed: {error}"
        }), 500


@app.route("/api/senses", methods=["GET"])
def senses():
    word = request.args.get("word", "bank")
    results = get_senses(word)

    return jsonify({
        "success": True,
        "word": word,
        "total": len(results),
        "senses": results
    })


@app.route("/api/relations", methods=["GET"])
def relations():
    word = request.args.get("word", "bank")
    sense_name = request.args.get("sense", "")

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
        "relations": lexical_relations(synset)
    })


@app.route("/api/report", methods=["POST"])
def report():
    data = request.get_json(silent=True) or {}

    try:
        result = analyze_context(
            data.get("sentence", ""),
            data.get("target", "")
        )

        predicted = result["predicted"]

        report_text = f"""
WORD SENSE DISAMBIGUATION — TASK 5
==================================

Input sentence:
{result['sentence']}

Target word:
{result['target']}

Method:
{result['method']}

Total candidate senses:
{result['total_senses']}

Predicted sense:
{predicted['name']}

Part of speech:
{predicted['pos_name']}

Definition:
{predicted['definition']}

WordNet examples:
{', '.join(predicted['examples']) or 'No examples available.'}

Synonyms:
{', '.join(result['relations']['synonyms']) or 'None found'}

Antonyms:
{', '.join(result['relations']['antonyms']) or 'None found'}

Analysis:
The system retrieves candidate meanings from WordNet and applies
a Lesk-style context-overlap baseline to select a likely meaning.
The result depends on the sentence context and the lexical evidence.
The prediction is not a calibrated confidence probability.
"""

        return jsonify({
            "success": True,
            "report": report_text.strip(),
            "result": result
        })

    except (ValueError, LookupError) as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)
