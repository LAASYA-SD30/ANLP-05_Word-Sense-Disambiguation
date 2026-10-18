import os
import pandas as pd
import nltk
import wsd_core as core

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # Deliverable 1: sense inventory
    inventory = core.build_sense_inventory(core.TARGET_WORDS)
    inventory.to_csv(f"{OUT_DIR}/sense_inventory.csv", index=False)
    print(f"Sense inventory saved: {len(inventory)} senses for {len(core.TARGET_WORDS)} words")
    print(inventory.groupby("word").size().to_string())

    # Deliverable 2: WSD results
    results, accuracy = core.evaluate_lesk()
    results.to_csv(f"{OUT_DIR}/wsd_results.csv", index=False)
    print("\n=== Lesk accuracy on sample sentences (%) ===")
    print(accuracy.to_string())

    per_word, overall = core.supervised_cv_accuracy()
    print("\n=== Supervised WSD 4-fold CV accuracy (%) ===")
    print(pd.Series(per_word).to_string())
    print(f"overall: {overall}")

    text = ("The farmer sat on the bank of the river. "
            "He opened a savings account at the bank to deposit his money.")
    all_words = []
    for s in nltk.sent_tokenize(text):
        all_words += core.disambiguate_sentence(s, "extended_lesk")
    pd.DataFrame(all_words).to_csv(f"{OUT_DIR}/wsd_all_words.csv", index=False)

    # Deliverable 3: short analysis report
    wrong = results[(results.method == "extended_lesk") & (~results.correct)]
    avg_senses = inventory.groupby("word").size().mean()
    lines = [
        "WORD SENSE DISAMBIGUATION - ANALYSIS REPORT",
        "=" * 45,
        "",
        "1. Objective",
        "Identify the senses of ambiguous words using WordNet, explore lexical",
        "relations, and apply knowledge-based and supervised WSD to sample text.",
        "",
        "2. Sense inventory",
        f"Target words: {', '.join(core.TARGET_WORDS)}",
        f"Total senses: {len(inventory)} (average {avg_senses:.1f} per word).",
        "See output/sense_inventory.csv for glosses, synonyms, antonyms,",
        "hypernyms, hyponyms and meronyms.",
        "",
        "3. Methods compared",
        "- First-sense baseline: most frequent WordNet sense.",
        "- Simple Lesk: overlap of sentence context with gloss and examples.",
        "- Extended Lesk: adds hypernym, hyponym, meronym glosses and synonyms.",
        "- Supervised: TF-IDF and Logistic Regression (scikit-learn).",
        "",
        "4. Results",
        "Lesk accuracy on sample sentences (%):",
        accuracy.to_string(),
        "Supervised 4-fold CV accuracy (%):",
        pd.Series(per_word).to_string(),
        f"Supervised overall: {overall}",
        "",
        "5. Error analysis (Extended Lesk misclassified):",
    ]
    if wrong.empty:
        lines.append("None.")
    for _, r in wrong.iterrows():
        lines.append(f"- '{r.sentence}' -> {r.predicted_synset}: {r.definition}")
    lines += [
        "",
        "6. Observations",
        "- Gloss overlap fails when a sentence shares few words with any gloss.",
        "- Extending signatures with related synsets increases overlap coverage.",
        "- Short sentences give little context, so ties fall back to the first sense.",
        "- The supervised model needs labelled data per word and does not",
        "  generalise to words without training examples.",
    ]
    with open(f"{OUT_DIR}/analysis_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\nSaved to 'output': sense_inventory.csv, wsd_results.csv, "
          "wsd_all_words.csv, analysis_report.txt")


if __name__ == "__main__":
    main()