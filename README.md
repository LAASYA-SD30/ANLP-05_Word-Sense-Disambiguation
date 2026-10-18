# Word Sense Disambiguation Using WordNet

## 1. Project Overview

This project implements a **Word Sense Disambiguation (WSD)** system using WordNet, the Extended Lesk algorithm, and supervised machine learning. It identifies the appropriate meaning of ambiguous words based on their context and provides information about their lexical relationships.

The application is developed using Python and Flask, with an interactive web interface built using HTML, CSS, and JavaScript.

## 2. Objectives

- Retrieve different senses of ambiguous words using WordNet.
- Explore synonyms, antonyms, hypernyms, hyponyms, meronyms, and holonyms.
- Implement the Extended Lesk algorithm for context-based word sense disambiguation.
- Implement supervised WSD using TF-IDF and Logistic Regression.
- Evaluate predictions using sample sentences.
- Present predictions and evaluation results through a web interface.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend implementation |
| Flask | Web application framework |
| NLTK WordNet | Word sense inventory and lexical relations |
| Scikit-learn | TF-IDF and Logistic Regression |
| Pandas | Result processing and tabular outputs |
| HTML | Webpage structure |
| CSS | Interface styling |
| JavaScript | Frontend interactions |

## 4. Features

### Sense Inventory and Lexical Relations
Retrieves WordNet senses for a given word and displays definitions, examples, parts of speech, and available lexical relations.

### WSD on Sample Text
Uses the Extended Lesk algorithm to predict WordNet senses based on contextual overlap.

### Supervised WSD
Uses TF-IDF features and Logistic Regression trained on labelled example sentences to predict the sense of a target word.

### Evaluation and Analysis
Displays the accuracy reported by the implemented methods and per-sentence Extended Lesk evaluation results.

## 5. Project Structure

```text
ANLP-05_Word-Sense-Disambiguation/
│
├── app.py
├── app_backup.py
├── wsd_core.py
├── wsd_project.py
├── requirements.txt
├── .gitignore
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── output/
│   ├── analysis_report.txt
│   ├── sense_inventory.csv
│   ├── wsd_all_words.csv
│   └── wsd_results.csv
│
└── screenshots/
    ├── 01_sense_analyzer.png
    ├── 02_Supervised WSD.png
    ├── 03_lexical_relations.png
    └── 04_ Evaluation and Analysis.png
```

## 6. Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- Visual Studio Code (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/LAASYA-SD30/ANLP-05_Word-Sense-Disambiguation.git
cd ANLP-05_Word-Sense-Disambiguation
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If required by your NLTK setup, download the WordNet resources using Python:

```python
import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")
```

## 7. Running the Application

From the project directory, activate the virtual environment and run:

```bash
python app.py
```

Open the local URL displayed in the terminal. For a typical Flask development configuration, this is:

```text
http://127.0.0.1:5000
```

The exact URL depends on the host and port configured in the application.

## 8. Sample Input

**Input sentence:**

```text
The farmer sat on the bank of the river.
He opened a savings account at the bank to deposit his money.
```

**Target word:** `bank`

The system retrieves candidate WordNet senses and uses the selected WSD method to predict a meaning based on the sentence context.

## 9. Evaluation

The application displays the following results for its demonstration dataset:

- **Extended Lesk accuracy:** 25%
- **Supervised WSD accuracy for bank:** 100%
- **Supervised WSD accuracy for bat:** 100%
- **Reported overall supervised accuracy:** 100%

These values are specific to the available demonstration examples. The supervised results should not be interpreted as general-purpose accuracy without testing on a larger, independent labelled dataset.

## 10. Output Files

The `output/` directory contains generated results:

- `analysis_report.txt` — analysis and evaluation report.
- `sense_inventory.csv` — retrieved WordNet sense information.
- `wsd_all_words.csv` — word-level WSD results.
- `wsd_results.csv` — WSD prediction results.

The `screenshots/` directory contains visual evidence of the application's interface and outputs.

## 11. Limitations and Future Enhancements

- Expand the labelled dataset with more words and contextual examples.
- Improve contextual scoring and handling of function words.
- Evaluate performance on a larger, independently labelled dataset.
- Compare additional WSD approaches using precision, recall, and F1-score.
- Explore contextual language models for improved sense prediction.

## 12. Author

**GitHub:** [LAASYA-SD30](https://github.com/LAASYA-SD30)

## 13. References

- [WordNet](https://wordnet.princeton.edu/)
- [NLTK Documentation](https://www.nltk.org/)
- [NLTK WordNet Guide](https://www.nltk.org/howto/wordnet.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Flask Documentation](https://flask.palletsprojects.com/)
