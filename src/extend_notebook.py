"""Extend the interim notebook with the final-modelling chapters.

The interim notebook ends at the Phase 3i baseline. The brief requires a single,
well-annotated technical deliverable covering every stage, so this script appends
Part II (N = 5 embedding matrix, Phase 3iv refinement, comparison, business
insights and conclusions) and writes the final notebook.

The appended cells read the checked-in artefacts, so they are executable without
retraining any model. The scripts that produce those artefacts are listed in the
final reproducibility cell.

Run: python src/extend_notebook.py
Writes: DSAI4205_Final_Group8_Twitter_Sentiment.ipynb
"""
from __future__ import annotations

import csv
import json
import re
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_NB = ROOT / "DSAI4205_Interim_Group8_Twitter_Sentiment.ipynb"
OUT_NB = ROOT / "DSAI4205_Final_Group8_Twitter_Sentiment.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": text}


CELLS = [
    md("""---
# Part II - Final modelling (Weeks 11-13)

The first part of this notebook covered EDA, the cleaning pipeline and the
required TF-IDF + Logistic Regression baseline. This part covers the advanced
modelling required by Phase 3(ii)-(iv):

1. **N = 5 embedding x analytics-model matrix.** Five analytics models, each
   paired with a different word embedding, scored under one protocol.
2. **Baseline refinement (Phase 3iv).** Train-only CV sweeps, then one test
   evaluation of the selected configuration.
3. **Comparison and interpretation.** Advanced models vs both baselines, plus
   the class-weight trade-off.
4. **Business insights and conclusions.**

All numbers in this part are read from the checked-in artefacts, so the
notebook cannot drift from the report.
"""),
    md("""## 8. The N = 5 embedding x analytics-model matrix

### 8.1 Design constraints

The brief requires **N analytics models**, where N is the number of group
members (five), each paired with a different **word embedding**.

- TF-IDF is **not** a word embedding, so it cannot occupy one of the five slots.
  It is kept as a reference row only.
- Logistic Regression is reserved for the baseline, so it cannot be reused.
- The five analytics models must be **different algorithms**.
- A single model may not serve as both embedding and classifier. DistilBERT is
  therefore frozen and paired with an independently fitted MLP.
- The consultation asked us to avoid near-identical tree models: only one
  Random Forest, no Extra Trees, and a Neural Network included.

The final five analytics models are XGBoost, Random Forest, SVM, MLP and k-NN -
five distinct algorithms.
"""),
    md("""### 8.2 Protocol

`src/embeddings_common.py` holds one protocol so every model is comparable:

- identical rows and identical folds (`StratifiedKFold(5, shuffle, seed=42)`);
- cross-validation on the **training file only**;
- the test file scored **exactly once** per model;
- the same metric block (accuracy, macro F1, weighted F1, kappa, within-1-class
  accuracy).

Two adapters are needed. Dense document vectors (GloVe, Word2Vec, FastText,
Sentence-BERT) are mean-pooled into one vector per tweet and passed to a
classical estimator. Frozen DistilBERT takes token ids, produces one 768-d
masked-mean vector per tweet, and hands that matrix to an independently fitted
MLP; the encoder is never trained. Mean pooling is a deliberate baseline choice;
it is also a reason the non-transformer embeddings underperform, because it
discards word order.
"""),
    code("""import json
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)

try:
    matrix = pd.read_csv("reports/tables/embedding_model_matrix.csv")
except FileNotFoundError:
    matrix = pd.DataFrame(PART2_DATA["matrix"])
cols = ["id", "embedding", "analytics_model", "cv_accuracy", "cv_sd",
        "test_accuracy", "test_macro_f1", "kappa", "within1", "role"]
display(matrix.sort_values(["eligible_for_n5", "test_accuracy"],
                           ascending=[False, False])[cols]
        .round(4)
        .reset_index(drop=True))"""),
    md("""### 8.3 Reading the matrix

Three observations matter:

- **Frozen DistilBERT + MLP is the strongest eligible model**, but it does not
  beat the deployed TF-IDF reference. The end-to-end fine-tuned reference does,
  but it is not counted because it trains encoder and classifier jointly. One
  test row is 0.25 pp.
- **The CV-to-test column exposes rank reversals.** Word2Vec + RF has strong CV
  accuracy but the lowest test accuracy; Sentence-BERT falls from CV to test.
  With 592 training rows, CV folds share one fitted embedding space, so CV is
  not an independent estimate of generalisation.
- **The from-scratch embeddings are weak** because 592 tweets cannot estimate a
  stable vector space. GloVe, pretrained on billions of tokens, scores better
  under the same kind of pipeline.

The representation, not the classifier, is the lever.
"""),
    md("""## 9. Baseline refinement (Phase 3iv)

Every candidate was scored by 5-fold CV on the **training file only**; the
winning configuration was frozen before the test file was scored once. The full
sweep is `reports/tables/improved_cv_sweep.csv`.
"""),
    code("""try:
    sweep = pd.read_csv("reports/tables/improved_cv_sweep.csv")
except FileNotFoundError:
    sweep = pd.DataFrame(PART2_DATA["sweep"])
display(sweep.sort_values("cv_macro_f1", ascending=False).head(8).round(4))"""),
    code("""try:
    final = json.load(open("models/final_metrics.json", encoding="utf-8"))
    improved = json.load(open("models/improved_metrics.json", encoding="utf-8"))
except FileNotFoundError:
    final = PART2_DATA["final_metrics"]
    improved = PART2_DATA["improved_metrics"]

rows = []
for r in final["cv_train_only"]:
    test = next(t for t in final["test_results"] if t["model"] == r["model"])
    rows.append(["M1-M3: " + r["model"], r["accuracy"], r["macro_f1"],
                 test["accuracy"], test["macro_f1"]])
rows.append(["B4: phrase+lemma, balanced",
             improved["cv"]["accuracy"], improved["cv"]["macro_f1"],
             improved["test"]["accuracy"], improved["test"]["macro_f1"]])

display(pd.DataFrame(rows, columns=["model", "CV acc", "CV macro F1",
                                    "Test acc", "Test macro F1"]).round(4))"""),
    md("""### 9.1 The class-weight trade-off

Balanced class weights raise macro F1 in cross-validation by lifting the
minority class, at the cost of some accuracy on the largest class. The table
below makes the per-class shift explicit, which is the point the consultation
asked us to explain.
"""),
    code("""try:
    dep = pd.read_csv("reports/tables/final_per_class.csv")
    b4 = pd.read_csv("reports/tables/improved_per_class.csv")
except FileNotFoundError:
    dep = pd.DataFrame(PART2_DATA["final_per_class"])
    b4 = pd.DataFrame(PART2_DATA["improved_per_class"])
comp = dep.merge(b4, on="class", suffixes=("_deployed", "_b4"))
display(comp[["class", "recall_deployed", "recall_b4",
              "f1_deployed", "f1_b4"]].round(3))"""),
    md("""The CV-winning balanced model does **not** beat the deployed model on the
fixed test file. It is therefore retained as an experiment, not deployed. That
is the honest outcome, and it is consistent with the instructor's point that
macro F1 can rise while accuracy falls without the model becoming "worse".
"""),
    md("""## 10. Comparison: advanced models vs both baselines

Against the Phase 3i baseline, only DistilBERT wins on accuracy. Against the
deployed M2, the picture is similar: DistilBERT gains about 1.3 pp, every other
embedding is below M2, and the TF-IDF model remains best on macro F1.

For a decision-support system the practical choice is therefore M2 unless the
small transformer edge justifies pretraining and CPU time.
"""),
    md("""## 11. Business insights

The model is useful as a **triage tool**, not an oracle:

- **Route complaints.** Negative has high recall, so a complaints queue can be
  built reliably; the cost is lower precision, which is acceptable with human
  review.
- **Separate chatter.** Irrelevant has high precision but low recall because
  the entity column is missing. Send low-confidence rows to review rather than
  auto-classifying them.
- **Aggregate by brand.** Brand-level sentiment is the level at which managers
  act; row-level explanations matter less than the aggregate trend.

Any deployment agreement should state the measured ceiling: roughly half the
rows get the exact four-way label right, and about three in four get the
polarity direction right.
"""),
    md("""## 12. Limitations and conclusions

- The dataset is missing its `entity` column, so "Irrelevant" is partly
  unlearnable. This is the largest single limitation.
- The corpus has 989 rows in total; one test row is 0.25 pp, so small gaps are
  noise.
- Label noise is severe: highly similar tweets share a label only about one
  time in three, and a lexicon audit agrees with the label only 41-53% of the
  time on clear sentiment.
- The required baseline reaches about 50% accuracy; the deployed refinement
  reaches about 54%; the best N = 5 model reaches about 55%. A realistic
  ceiling is 60-65%.

The project delivered every phase of the brief. The important result is not a
score but the measured explanation of why the score plateaus.
"""),
    md("""## 13. Reproducing Part II

```bash
python src/embed_1_glove_mlp.py       # GloVe -> Neural Network (MLP)
python src/embed_2_word2vec_rf.py     # Word2Vec -> Random Forest
python src/embed_3_fasttext_svm.py    # FastText -> SVM
python src/embed_4_bert_head.py       # DistilBERT -> neural head
python src/embed_5_sbert_knn.py       # Sentence-BERT -> kNN
python src/experiments_embeddings.py  # assemble matrix, fig15/fig16
python src/explore_improved.py        # Phase 3iv train-only CV sweep
python src/train_improved.py          # selected config, scored once on test
python src/make_final_report.py       # rebuild the final .docx
```

---
*End of final technical notebook - Group 8 ChillGuy - DSAI4205 Big Data Analytics (2026/27 S1)*
"""),
]


IMPL_FILES = [
    "src/embeddings_common.py",
    "src/embed_1_glove_xgb.py",
    "src/embed_2_word2vec_rf.py",
    "src/embed_3_fasttext_svm.py",
    "src/embed_4_bert_frozen_mlp.py",
    "src/embed_5_sbert_knn.py",
    "src/experiments_embeddings.py",
    "src/train_improved.py",
]


def strip_main(src: str) -> str:
    """Remove the `if __name__ == '__main__': main()` block.

    In a notebook `__name__` is '__main__', so leaving it in would execute the
    training routine the moment the cell is run. The functions stay defined and
    can be called explicitly.
    """
    return re.split(r"\nif __name__ == [\"']__main__[\"']:", src)[0].rstrip()


SETUP_CELL = code("""# --- Run-All setup: dependencies + repository root ---
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

# 1) Small packages the notebook needs, installed into the SAME interpreter
#    that runs this kernel (so it works even if Jupyter uses another Python).
for module, package in [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("nltk", "nltk"),
    ("ftfy", "ftfy"),
    ("wordcloud", "wordcloud"),
    ("seaborn", "seaborn"),
]:
    if importlib.util.find_spec(module) is None:
        print(f"Installing {package} ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--user", package])
importlib.invalidate_caches()

# 2) NLTK corpora used by the cleaning and lexicon steps.
import nltk
for corpus, folder in [
    ("vader_lexicon", "sentiment"),
    ("wordnet", "corpora"),
    ("omw-1.4", "corpora"),
    ("stopwords", "corpora"),
]:
    try:
        nltk.data.find(f"{folder}/{corpus}")
    except LookupError:
        nltk.download(corpus, quiet=True)

# 3) Locate the repository root so relative paths work regardless of cwd.
def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "src" / "embeddings_common.py").exists():
            return p
    return start


ROOT = _find_repo_root(Path.cwd().resolve())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
print("Dependencies checked. Repository root located.")""")

SETUP_NOTE = md("""> **Submission note: this notebook is fully self-contained.**
> All preprocessing and model code is inside this file, the dataset
> (`traindata.csv`, `testdata.csv`) is embedded, and the final result tables
> are embedded as a fallback. The first cell installs the small dependencies
> and the NLTK data into the kernel you are using, then reconstructs the
> dataset beside the notebook. **Run All does not retrain the five models**;
> it runs the EDA/baseline sections live and displays the recorded results.""")


def _csv_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    with open(SRC_NB, encoding="utf-8") as f:
        nb = json.load(f)

    # Recorded results, embedded so Part II can run without the repository files.
    part2 = {
        "matrix": _csv_rows(ROOT / "reports" / "tables" / "embedding_model_matrix.csv"),
        "sweep": _csv_rows(ROOT / "reports" / "tables" / "improved_cv_sweep.csv"),
        "final_metrics": json.load(open(ROOT / "models" / "final_metrics.json", encoding="utf-8")),
        "improved_metrics": json.load(open(ROOT / "models" / "improved_metrics.json", encoding="utf-8")),
        "final_per_class": _csv_rows(ROOT / "reports" / "tables" / "final_per_class.csv"),
        "improved_per_class": _csv_rows(ROOT / "reports" / "tables" / "improved_per_class.csv"),
    }
    part2_literal = repr(json.loads(json.dumps(part2)))

    raw_dir = ROOT / "Data 5 - Twitter" / "Data 5 - Twitter"
    raw_train = (raw_dir / "traindata.csv").read_text(encoding="utf-8")
    raw_test = (raw_dir / "testdata.csv").read_text(encoding="utf-8")

    setup_source = (
        SETUP_CELL["source"]
        + "\n\n# --- dataset embedded so the notebook needs no external files ---\n"
        + "RAW_TRAIN_CSV = " + repr(raw_train) + "\n"
        + "RAW_TEST_CSV = " + repr(raw_test) + "\n"
        + "_has_original = (\n"
        + "    (ROOT / 'Data 5 - Twitter' / 'traindata.csv').exists()\n"
        + "    or (ROOT / 'Data 5 - Twitter' / 'Data 5 - Twitter' / 'traindata.csv').exists()\n"
        + ")\n"
        + "if not _has_original:\n"
        + "    for _name, _text in (('traindata.csv', RAW_TRAIN_CSV), ('testdata.csv', RAW_TEST_CSV)):\n"
        + "        _p = ROOT / _name\n"
        + "        if not _p.exists():\n"
        + "            _p.write_text(_text, encoding='utf-8')\n"
        + "print('Dataset ready next to the notebook.')\n"
        + "\n# --- recorded results (fallback when repository files are absent) ---\n"
        + "PART2_DATA = " + part2_literal
    )

    # setup must run before the first code cell
    nb["cells"].insert(1, SETUP_NOTE)
    nb["cells"].insert(1, code(setup_source))

    # inline text_clean so Part I does not depend on src/
    tc_source = (ROOT / "src" / "text_clean.py").read_text(encoding="utf-8")
    nb["cells"].insert(3, code(
        "# --- src/text_clean.py (inlined so the notebook is self-contained) ---\n"
        + tc_source
        + "\n\n# register the inlined definitions as the `text_clean` module\n"
          "import sys as _sys, types as _types\n"
          "_tc = _types.ModuleType('text_clean')\n"
          "_tc.__dict__.update({k: v for k, v in globals().items() if not k.startswith('__')})\n"
          "_sys.modules['text_clean'] = _tc\n"
          "print('text_clean loaded from the notebook.')"
    ))

    # everything except the closing markdown cell
    nb["cells"].extend(CELLS[:-1])

    nb["cells"].append(md("""## 14. Full model implementation

This section embeds the exact source code that produced Part II. The cells are
deliberately side-effect-free: their `main()` entry points are removed so that
running a cell only defines functions. Call `main()` explicitly to retrain a
row. The shared protocol lives in `embeddings_common.py`; each model script then
declares its embedding and estimator and calls `cv_and_test`.
"""))

    nb["cells"].append(code("""import sys
from pathlib import Path

ROOT = Path.cwd()
if (ROOT / "src").is_dir():
    sys.path.insert(0, str(ROOT / "src"))
print("Repository root located.")"""))

    for rel in IMPL_FILES:
        path = ROOT / rel
        if not path.exists():
            continue
        source = strip_main(path.read_text(encoding="utf-8"))
        if rel.endswith("embeddings_common.py"):
            source += (
                "\n\n# register the inlined definitions as the `embeddings_common` module\n"
                "import sys as _sys, types as _types\n"
                "_ec = _types.ModuleType('embeddings_common')\n"
                "_ec.__dict__.update({k: v for k, v in globals().items() if not k.startswith('__')})\n"
                "_sys.modules['embeddings_common'] = _ec\n"
                "print('embeddings_common loaded from the notebook.')"
            )
        nb["cells"].append(md(f"### `{rel}`"))
        nb["cells"].append(code(source))

    nb["cells"].append(CELLS[-1])  # closing markdown

    with open(OUT_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"written -> {OUT_NB}")
    print(f"cells   -> {len(nb['cells'])}")


if __name__ == "__main__":
    main()
