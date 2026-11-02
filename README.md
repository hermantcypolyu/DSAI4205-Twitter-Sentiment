# DSAI4205 Group Project - Dataset 5: Twitter Sentiment

**Group 8 - ChillGuy** | Interim submission (Week 10)

---

## Which file do I need?

| I want to... | Open this |
|---|---|
| **Submit / show my professor** | `reports/Interim_Report_Group8_Twitter_Sentiment.docx` (22 pages) |
| **Understand the project in Cantonese** | `reports/白話解釋手冊_Group8_Twitter情緒分析.docx` (19 pages) |
| **Submit the technical deliverable** | `DSAI4205_Interim_Group8_Twitter_Sentiment.ipynb` |
| **Prepare for the consultation Q&A** | 白話解釋手冊 → Chapter 7 (Q&A cheat sheet) |

---

## Key results at a glance

**Deployed model: word + char TF-IDF union → Logistic Regression (C=1.0) on raw `phrase`**

| Metric | Baseline (v1) | **Final (v2)** |
|---|---|---|
| Test accuracy | 49.6% | **54.2%** |
| Test macro F1 | 44.0% | **50.4%** |
| Test balanced accuracy | 47.1% | **50.5%** |
| Cohen's κ (4-class) | 0.30 | **0.37** |
| within-1-class accuracy | 70.1% | **71.5%** |
| Majority-class reference | 30.2% | 30.2% |
| Training rows after cleaning | 592 of 596 | 592 of 596 |

**Best encoder (optional, needs `torch`):** DistilBERT fine-tuned → **56.9%** test
accuracy / 48.6% macro-F1, at ~15 min CPU per run.

### Model comparison (held-out test, scored once)

| Model | 5-fold CV acc (train) | Test acc | Test macro-F1 |
|---|---|---|---|
| Majority-class guess | — | 30.2% | 11.6% |
| M1 word TF-IDF + LR *(v1 baseline)* | 47.5% | 50.9% | 44.3% |
| M3 word+char union + LinearSVC | 48.2% | 52.6% | 50.0% |
| **M2 word+char union + LR** ← deployed | 49.2% | **54.2%** | **50.4%** |
| DeBERTa-v3-small fine-tuned | 50.5% ± 3.9% | 53.2% | 43.1% |
| DistilBERT fine-tuned | 51.7% ± 0.8% | 56.9% | 48.6% |
| **RoBERTa-base fine-tuned** ← best | 54.4% ± 2.0% | **57.4%** | **49.9%** |
| M5 ensemble (0.5·M2 + 0.5·M4) | — | 55.7% | — |

> **Note the rank reversal.** On train CV, RoBERTa leads DistilBERT by 2.7 pp and
> LinearSVC *loses* to the union-LR by 1.0 pp — but on test, DistilBERT gains
> 5.2 pp over its CV score while DeBERTa gains only 2.7 pp. With 397 test rows
> one row = 0.25 pp, so gaps under ~2 pp are inside the noise band. The
> defensible claims are: (a) the transformer family beats the linear family,
> (b) character n-grams beat word-only, (c) everything beats majority by 20+ pp.

> **Protocol:** model *selection* uses 5-fold stratified CV on **train only**;
> every test score above is a single evaluation of an already-fixed model.
> Seed = 42 everywhere. See `src/train_final.py`.

---

## The four findings worth discussing

1. **The dataset is missing its `entity` column.** The documentation defines an entity-level task, but the CSVs contain only `Phrase` and `Sentiment`. Label 2 ("Irrelevant") is therefore defined relative to information that was never shipped — the model catches only 9 of 66 such rows.
2. **Imbalance is mild (1.65:1); label overlap is the real problem.** Class weights still lift macro F1 by 6.6 pp in CV, which shows the minority class needs help even at mild skew.
3. **15.8% of rows had double-encoded UTF-8 (mojibake).** Repaired 92 rows with `ftfy`. Invisible in Excel, but it splits every affected word out of the vocabulary.
4. **Cleaning does not raise accuracy — and we say so.** Cleaning is justified by correctness, test validity and BERT-compatibility in Weeks 11–13, not by score.

---

## Why the accuracy plateaus near 55% — measured, not guessed

This was the central question of the project, so we ran four controlled
experiments (`src/experiments_headroom.py`) rather than speculating.

### E1 — Permutation control (is the model learning anything real?)

| Condition | CV accuracy |
|---|---|
| Real labels | **47.3%** |
| Labels shuffled (mean of 5 runs) | 28.1% ± 1.8% (max 31.4%) |
| Majority-class guess | 30.2% |

The model sits **19 points above** the shuffled-label ceiling. It is learning
real signal — the limit is not a bug or a broken feature space.

### E2 — Nearest-neighbour label agreement (how noisy are the labels?)

For each training tweet, find the *k* nearest other training tweets by TF-IDF
cosine similarity and check whether they share its label. Median neighbour
cosine similarity is **0.90** — these really are near-duplicates.

| Text view | k=5 | k=10 | k=25 |
|---|---|---|---|
| `phrase` (raw) | 35.1% | **33.6%** | 31.4% |
| `text` (cleaned) | 34.6% | 33.1% | 30.7% |
| `text_lemma` | 34.7% | 33.6% | 31.5% |

**Tweets that are 90% textually identical share a label only one time in three.**
No model can exceed the consistency of its own labels, so labels — not features —
set the ceiling.

### E3 — Cross-view agreement (is our cleaning destroying signal?)

| Pair | Prediction agreement |
|---|---|
| `phrase` ↔ `text` | 91.4% |
| `phrase` ↔ `text_lemma` | 73.5% |
| `text` ↔ `text_lemma` | 74.7% |

Light cleaning is **lossless**. Lemmatisation + stop-word removal throws away
**26%** of the view — which is exactly why the final model reads `phrase` and
gained ~4.5 pp by doing so.

### E4 — One-vs-rest AUC (which class is genuinely unlearnable?)

| Class | Model AUC | Best single-feature rule | Always-say-no baseline |
|---|---|---|---|
| Negative | 0.78 | 0.76 | 0.70 |
| Positive | 0.69 | 0.57 | 0.72 |
| **Irrelevant** | **0.65** | **0.61** | **0.82** |
| Neutral | 0.69 | 0.55 | 0.75 |

For Irrelevant, **every learned model loses to the trivial "never predict
Irrelevant" rule (0.82)**. That is the signature of a class defined by
information the dataset does not contain — confirmed by the missing `entity`
column.

### Correcting a common mis-estimate

It is tempting to say "fixing Irrelevant alone would give +25 pp". It would not.
**50** of the 397 test rows are Irrelevant tweets predicted as something else:

> 50 / 397 = **12.6 pp** — even *perfect* Irrelevant recall caps accuracy at **87.4%**
> for that class alone, but Irrelevant contributes only **27.5% of the 182 errors**.

### Where the 182 errors actually go (397 test rows)

| Error type | count | pp of accuracy |
|---|---|---|
| Correct | 215 | 54.2% |
| **Positive → Negative** (polarity flip) | **37** | **−9.3** |
| Neutral → Negative | 32 | −8.1 |
| Irrelevant → Negative | 23 | −5.8 |
| Positive → Neutral | 20 | −5.0 |
| Irrelevant → Positive | 17 | −4.3 |
| Negative → Positive | 13 | −3.3 |
| Neutral → Positive | 20 | −5.0 |
| Irrelevant → Neutral | 10 | −2.5 |
| Other (→ Irrelevant) | 10 | −2.5 |

**The single largest error source is Positive → Negative: 37 rows (9.3 pp).**
These are polarity flips — tweets the model reads with the *opposite* sentiment.
Combined with Neutral → Negative (32), **69 of 182 errors (38%) are "negative"
acting as a gravitational well**, absorbing Neutral, Positive and Irrelevant
alike. That is a class-prior artefact, not a language-understanding failure: the
model has learned that "strong words ⇒ Negative" because Negative is the
second-largest class and the easiest to trigger.

**Verdict on 80%:** not reachable with this dataset. Three independent hard
limits — (1) label noise: 90%-similar tweets agree only 33.6% of the time;
(2) the missing `entity` column: Irrelevant is unlearnable without it;
(3) sample size: 989 total rows, 397 test rows, so ±2–3 pp is noise.
A realistic ceiling is **60–65%**, and 54.2% (or 56.9% with BERT) sits close to it.

---

## Reproducing everything

```bash
# --- pipeline -----------------------------------------------------------
python src/run_pipeline.py          # clean data, EDA figures, cleaning log
python src/train_baseline.py        # v1 baseline, model figures, metrics
python src/train_final.py           # v2 final model  -> models/final_metrics.json
python src/train_final.py --with-transformer   # adds DistilBERT (~15 min CPU)

# --- the headroom investigation (produces the 4 experiments above) ------
python src/explore_models.py        # 28-config sweep
python src/audit_labels.py          # lexicon label-noise audit
python src/analyse_ceiling.py       # neighbour-consistency ceiling
python src/experiments_headroom.py  # E1 permutation, E2/E3/E4
python src/explore_transformer.py distilbert-base-uncased 4
python src/explore_transformer.py roberta-base 4
python src/explore_transformer.py microsoft/deberta-v3-small 4

# --- documents ----------------------------------------------------------
python src/make_interim_report.py   # rebuild the formal .docx
python src/make_layman_report.py    # rebuild the Cantonese guide
python src/make_notebook.py         # rebuild the notebook JSON
python src/verify_consistency.py    # cross-document number check
```

The report generators read every number from JSON artefacts
(`models/baseline_metrics.json`, `models/final_metrics.json`,
`data/processed/eda_summary.json`, `data/processed/cleaning_log.json`,
`data/processed/headroom_summary.json`) rather than from hand-typed text, so the
documents cannot drift out of sync with the code.

---

## Project structure

```
DSAI_project/
├── DSAI4205_Interim_Group8_Twitter_Sentiment.ipynb   technical deliverable
├── Data 5 - Twitter/Data 5 - Twitter/                raw CSVs (untouched)
│   ├── traindata.csv
│   └── testdata.csv
├── data/processed/
│   ├── train_clean.csv                               cleaned + raw text
│   ├── test_clean.csv
│   ├── cleaning_log.json                             row-level audit trail
│   └── eda_summary.json                              all EDA statistics
├── models/
│   ├── baseline_tfidf_logreg.joblib                  v1 fitted pipeline
│   ├── baseline_metrics.json                         v1 reported scores
│   ├── final_tfidf.joblib                            *** deployed model ***
│   └── final_metrics.json                            *** all v2 numbers ***
├── reports/
│   ├── Interim_Report_Group8_Twitter_Sentiment.docx  *** submit this ***
│   ├── 白話解釋手冊_Group8_Twitter情緒分析.docx          *** read this ***
│   ├── figures/                                      12 charts
│   └── tables/                                       CSV metric exports
└── src/
    ├── text_clean.py                                 shared cleaning functions
    ├── run_pipeline.py                               cleaning + EDA
    ├── train_baseline.py                             v1 baseline model
    ├── train_final.py                                *** v2 final pipeline ***
    ├── explore_models.py                             28-config linear sweep
    ├── audit_labels.py                               lexicon label-noise audit
    ├── analyse_ceiling.py                            neighbour-consistency ceiling
    ├── experiments_headroom.py                       E1-E4 headroom experiments
    ├── explore_transformer.py                        BERT/RoBERTa/DeBERTa sweep
    ├── make_interim_report.py                        report generator
    ├── make_layman_report.py                         Cantonese guide generator
    ├── make_notebook.py                              notebook generator
    ├── inspect_irrelevant.py                         Irrelevant-class investigation
    └── verify_consistency.py                         cross-document number check
```

---

## Baseline model

- **Features:** TF-IDF, word 1–2 grams, `sublinear_tf`, 7,780 features
- **Classifier:** multinomial logistic regression, C=1.0, lbfgs, max_iter=2000
- **Protocol:** stratified 5-fold CV on train (model selection) + single held-out test evaluation
- **Seed:** 42 everywhere

## Final model (v2)

- **Text view:** raw `phrase` — beats `text_lemma` by ~4.5 pp (E3 evidence)
- **Features:** `FeatureUnion` of
  - word 1–2 grams, `min_df=1`, `max_df=0.95`, `sublinear_tf`, unicode accents stripped
  - char_wb 2–5 grams, `min_df=2`, `max_df=0.95`, `sublinear_tf`
- **Classifier:** multinomial logistic regression, C=1.0, max_iter=4000
- **Why the union:** character n-grams absorb hashtags, `@mentions`, slang and
  misspellings that word tokens drop — Irrelevant recall rises from 13.6% → 24.2%
- **Protocol:** identical to v1; selection on train CV only, one test evaluation

---

## Cleaning pipeline (9 documented steps)

| # | Step | Rows affected |
|---|---|---|
| 1 | Schema standardisation (float→Int64 label) | 596 |
| 2 | Missing value handling | 0 |
| 3 | Blank text removal | 0 |
| 4 | Exact duplicate removal | 3 |
| 5 | Conflicting-label resolution (majority vote) | 1 |
| 6 | Invalid label correction | 0 |
| 7 | Encoding repair (ftfy + NFKC) | 92 |
| 8 | Normalisation + tokenisation + stop-words | 592 |
| 8.1 | Lemmatisation (vocabulary −13.0%) | 592 |
| 9 | Length outlier flagging (not deletion) | 0 (2 flagged) |

---

## Next steps (Weeks 11–13)

Five analytics models, each paired with a different embedding (N = 5 group members):

| # | Embedding | Model |
|---|---|---|
| 1 | GloVe | Gradient Boosting (XGBoost / LightGBM) |
| 2 | Word2Vec (from scratch) | Random Forest |
| 3 | FastText (subword) | SVM |
| 4 | BERT (fine-tuned) | Neural classifier head |
| 5 | Sentence-BERT | kNN / MLP |

Chosen across different model families because the brief penalises near-identical
tree models for lack of diversity.
