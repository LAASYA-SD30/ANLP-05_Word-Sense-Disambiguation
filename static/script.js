
const $ = (id) => document.getElementById(id);

function make(tag, text, cls) {
  const element = document.createElement(tag);

  if (text !== undefined && text !== null) {
    element.textContent = text;
  }

  if (cls) {
    element.className = cls;
  }

  return element;
}

function buildTable(headers, rows) {
  const wrap = make("div", null, "table-wrap");
  const table = make("table");
  const thead = make("thead");
  const head = make("tr");

  headers.forEach((header) => {
    head.appendChild(make("th", header));
  });

  thead.appendChild(head);
  table.appendChild(thead);

  const tbody = make("tbody");

  rows.forEach((row) => {
    const tr = make("tr");

    row.forEach((value) => {
      tr.appendChild(make("td", value ?? ""));
    });

    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  wrap.appendChild(table);

  return wrap;
}

async function callApi(url, payload) {
  const options = payload
    ? {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }
    : {};

  const response = await fetch(url, options);
  const data = await response.json();

  if (!response.ok || data.success === false) {
    throw new Error(data.error || "Request failed.");
  }

  return data;
}

function showError(box, error) {
  box.replaceChildren(make("p", error.message, "error"));
}

function loading(box) {
  box.replaceChildren(make("p", "Loading...", "muted"));
}

// 1. Sense inventory and lexical relations.
async function loadSenses() {
  const box = $("senses-output");
  const word = $("word-input").value.trim();

  if (!word) {
    showError(box, new Error("Enter a word."));
    return;
  }

  loading(box);

  try {
    const data = await callApi(
      "/api/senses?word=" + encodeURIComponent(word)
    );

    box.replaceChildren();

    box.appendChild(
      make(
        "p",
        `${data.total} senses found for "${data.word}".`,
        "muted"
      )
    );

    if (!data.senses.length) {
      box.appendChild(
        make("p", "No WordNet senses were found.", "muted")
      );
      return;
    }

    data.senses.forEach((sense) => {
      const card = make("div", null, "sense");

      card.appendChild(
        make("h3", `${sense.synset} (${sense.pos})`)
      );

      card.appendChild(make("p", sense.definition));

      if (sense.examples.length) {
        card.appendChild(
          make("p", "Example: " + sense.examples[0], "muted")
        );
      }

      const relations = [
        ["Synonyms", sense.synonyms],
        ["Antonyms", sense.antonyms],
        ["Hypernyms", sense.hypernyms],
        ["Hyponyms", sense.hyponyms],
        ["Meronyms", sense.meronyms],
        ["Holonyms", sense.holonyms]
      ];

      relations.forEach(([label, items]) => {
        const line = make("div", null, "rel");

        line.appendChild(make("b", label + ":"));

        const values = (items || []).map((item) =>
          typeof item === "string" ? item : item.name
        );

        line.appendChild(
          document.createTextNode(
            " " + (values.length ? values.join(", ") : "None found")
          )
        );

        card.appendChild(line);
      });

      box.appendChild(card);
    });
  } catch (error) {
    showError(box, error);
  }
}


// 2. All-words WSD.
async function runWsd() {
  const box = $("wsd-output");
  const sentence = $("sentence-input").value.trim();

  if (!sentence) {
    showError(box, new Error("Enter sample text."));
    return;
  }

  loading(box);

  try {
    const data = await callApi("/api/disambiguate", {
      sentence,
      method: $("method-select").value
    });

    box.replaceChildren();

    if (!data.results.length) {
      box.appendChild(
        make("p", "No content words found.", "muted")
      );
      return;
    }

    box.appendChild(
      buildTable(
        [
          "Word",
          "POS",
          "Predicted synset",
          "Definition",
          "Overlap"
        ],
        data.results.map((result) => [
          result.word,
          result.pos,
          result.predicted_synset,
          result.definition,
          result.overlap_score
        ])
      )
    );
  } catch (error) {
    showError(box, error);
  }
}


// 3. Supervised WSD.
async function runSupervised() {
  const box = $("sup-output");
  const word = $("sup-word").value;
  const sentence = $("sup-sentence").value.trim();

  if (!word || !sentence) {
    showError(
      box,
      new Error("Enter a target word and sentence.")
    );
    return;
  }

  loading(box);

  try {
    const data = await callApi("/api/supervised", {
      word,
      sentence
    });

    const result = make("div", null, "result-box");

    result.appendChild(
      make("h3", "Supervised WSD Prediction")
    );

    result.appendChild(
      make("p", "Target word: " + data.word)
    );

    result.appendChild(
      make("p", "Predicted sense: " + data.sense)
    );

    result.appendChild(
      make("p", "Definition: " + data.definition)
    );

    result.appendChild(
      make("p", "Model confidence: " + data.confidence + "%")
    );

    result.appendChild(
      make("p", "Method: " + data.method)
    );

    box.replaceChildren(result);
  } catch (error) {
    showError(box, error);
  }
}


// 4. Evaluation and analysis.
async function runEval() {
  const box = $("eval-output");

  loading(box);

  try {
    const data = await callApi("/api/evaluate");

    box.replaceChildren();

    box.appendChild(
      make("h3", "Lesk methods: accuracy on sample sentences")
    );

    box.appendChild(
      buildTable(
        ["Method", "Accuracy (%)"],
        Object.entries(data.lesk_accuracy)
      )
    );

    box.appendChild(
      make("h3", "Supervised WSD: sample accuracy by word")
    );

    const supervisedRows = Object.entries(
      data.supervised_per_word
    );

    supervisedRows.push([
      "Overall",
      data.supervised_overall
    ]);

    box.appendChild(
      buildTable(
        ["Word", "Accuracy (%)"],
        supervisedRows
      )
    );

    box.appendChild(
      make("h3", "Per-sentence results")
    );

    box.appendChild(
      buildTable(
        [
          "Sentence",
          "Target",
          "Method",
          "Predicted synset",
          "Correct"
        ],
        data.samples.map((sample) => [
          sample.sentence,
          sample.target,
          sample.method,
          sample.predicted_synset,
          sample.correct ? "Yes" : "No"
        ])
      )
    );

    if (data.note) {
      box.appendChild(make("p", data.note, "muted"));
    }
  } catch (error) {
    showError(box, error);
  }
}


// Connect all interactive controls.
$("senses-btn").addEventListener("click", loadSenses);
$("wsd-btn").addEventListener("click", runWsd);
$("sup-btn").addEventListener("click", runSupervised);
$("eval-btn").addEventListener("click", runEval);

$("word-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    loadSenses();
  }
});

// Load the initial sense inventory.
loadSenses();
