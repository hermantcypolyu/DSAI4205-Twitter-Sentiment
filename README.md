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

## Weeks 11–13: the N = 5 embedding × analytics-model matrix

The brief requires **N = 5 analytics models**, where N is the number of group
members, each paired with a different **word embedding**. Two constraints come
from the brief itself:

- **TF-IDF is not a word embedding.** It is a sparse count vector, so it cannot
  occupy one of the five slots.
- **Logistic Regression is reserved for the baseline.** The deployed model
  (M2) therefore cannot be re-used as one of the five.

`src/embeddings_common.py` holds the protocol so all five are comparable:
identical rows, identical folds (`StratifiedKFold(5, shuffle, seed=42)`),
identical metric block, CV on **train only**, and the test file scored **exactly
once** per model.

| # | Embedding | Analytics model | Script |
|---|---|---|---|
| 1 | GloVe `glove-wiki-gigaword-100` (frozen, 100-d) | Gradient Boosting (XGBoost) | `embed_1_glove_xgb.py` |
| 2 | Word2Vec skip-gram (**trained from scratch**) | Random Forest (500 trees) | `embed_2_word2vec_rf.py` |
| 3 | FastText char 3–6-grams (**from scratch**) | SVM (RBF kernel) | `embed_3_fasttext_svm.py` |
| 4 | DistilBERT (**fine-tuned**) | Neural classifier head | `embed_4_bert_head.py` |
| 5 | Sentence-BERT `all-MiniLM-L6-v2` (frozen, 384-d) | k-Nearest-Neighbours (k=25, cosine) | `embed_5_sbert_knn.py` |

The five are deliberately drawn from **different model families** — boosted
trees, bagged trees, kernel SVM, a neural head, and a non-parametric neighbour
vote — because the brief penalises near-identical tree models for lack of
diversity. Report with `src/experiments_embeddings.py`.

### Results

Produced by `python src/experiments_embeddings.py`; full table in
`reports/tables/embedding_model_matrix.csv`, figures `fig15` / `fig16`.

| # | Embedding | Analytics model | CV acc | Test acc | Test macro-F1 | κ |
|---|---|---|---|---|---|---|
| 4 | DistilBERT (fine-tuned) | Neural classifier head | 53.7% ± 3.2 | **55.4%** | 0.4485 | 0.381 |
| 5 | Sentence-BERT (all-MiniLM-L6-v2) | k-NN (k=25, cosine) | 49.7% ± 2.1 | **48.9%** | 0.4611 | 0.300 |
| 1 | GloVe (glove-wiki-gigaword-100) | Gradient Boosting (XGBoost) | 36.8% ± 4.6 | **46.1%** | 0.4176 | 0.261 |
| 3 | FastText (char 3–6-grams, from scratch) | SVM (RBF) | 39.2% ± 4.9 | **40.6%** | 0.3249 | 0.169 |
| 2 | Word2Vec (from scratch) | Random Forest (500) | 43.1% ± 3.5 | **37.8%** | 0.3205 | 0.134 |
| — | *TF-IDF (word + char union)* | *Logistic Regression (deployed)* | *49.2%* | *54.2%* | *0.5042* | *0.367* |

*Last row is the reference: TF-IDF is a sparse count vector, not a word
embedding, so it does **not** count toward N = 5. Majority-class baseline is
30.2%. One test row = 0.25 pp, so gaps under ~1.5 pp are noise.*

Two readings matter here.

**Best eligible combination: DistilBERT + neural head, 55.4%.** That is 1.3 pp
above the deployed TF-IDF model and 25.2 pp above the majority baseline. Because
one test row is worth 0.252 pp, the DistilBERT-vs-TF-IDF gap is about **5 rows**
— statistically real (test is fixed, seed is fixed) but substantively small.

**Rank reversal between CV and test.** Word2Vec + RF has the *second-highest*
CV accuracy (43.1%) but the *lowest* test accuracy (37.8%) — a 5.3 pp collapse,
the largest in the matrix. FastText moves the same direction but mildly
inverts (CV 39.2%, test 40.6%), so the ordering by CV does not predict the
ordering by test either way. This is the reason the project reports a single
held-out test score instead of trusting CV: with 592 training rows, CV folds
share one fitted embedding space and so are not independent estimates of
generalisation.

### Why does TF-IDF still beat every embedding?

Because on this dataset the classifier was never the bottleneck — the
**representation was arithmetically lossy** in a way that hurts exactly the
classes we cannot predict.

**1. Mean pooling destroys the word order that carries sentiment.** Every
non-transformer row reduces a tweet to one averaged vector. For *"not good"*
that vector is `½(vec(not) + vec(good))`, which sits almost on top of *"good"*.
A negation flips the label but barely moves the embedding — and negation cues
are common here: **345 of 989 rows (34.9%) contain an explicit negation word.**
TF-IDF keeps the bigram `not good` as its own independent feature, so a linear
model can learn "this feature ⇒ Negative" directly. GloVe and Word2Vec are
*structurally incapable* of representing this, no matter which classifier sits
on top.

Only DistilBERT avoids the problem, because its contextual encoder represents
each token *given its sentence* rather than by a fixed lookup — which is
precisely why it tops the eligible table.

**2. A 592-tweet corpus cannot learn a good space from scratch.** Word2Vec and
FastText produced a **4,280-vector vocabulary** from 12,565 training tokens —
an average of only **2.9 occurrences per word**, with **73.2% of the vocabulary
appearing exactly once**. Distributional methods cannot estimate a stable vector
from a single occurrence, so most of the learned space is noise. GloVe has
400,000 vectors from 6 billion tokens and scores much better (46.1%), which
isolates the effect: the *same* XGBoost pipeline gains **8.3 pp** purely by
swapping a badly-trained space for a well-trained one.

**3. Coverage is not the problem — quality is.** GloVe's out-of-vocabulary rate
on training tokens is only **17.9%**, and **zero** training rows had no
in-vocabulary token at all, so coverage cannot explain its 46.1%. FastText's
subword handling was genuinely exercised — **372 of 397 test rows contained at
least one word never seen in training**, and every one of them was still
embedded — yet it scored **third of five**. So the story is not "missing words";
it is "present-but-uninformative vectors": a single 100-d average carries far
less usable signal than thousands of sparse lexical features, each of which can
be learned independently.

**4. The evidence that this is a ceiling, not a tuning failure.** A logistic
regression on the *identical* GloVe vectors scored CV 38.3% / test 41.1%,
i.e. *worse* than XGBoost's 46.1% on the same vectors. Replacing the classifier
moved accuracy by less than 5 pp, while replacing the embedding moved it by
17.6 pp (Word2Vec 37.8% → DistilBERT 55.4%). The variance lives in the
representation.

### What this means for the ceiling claim

The matrix strengthens the report's original conclusion rather than overturning
it. DistilBERT is contextual — it is the only row whose representation can
represent *"not good"* differently from *"good"* — and it lands at 55.4%, close
to the 54.2% already reached by cheap TF-IDF. Adding an order-sensitive,
pretrained, fine-tuned encoder bought roughly **one percentage point**. When the
best representation available on this dataset is worth about 1 pp over a
generation-, parameter- and cost-free baseline, the remaining error is not a
modeling deficiency; it is consistent with the label-noise measurements in
`exp_label_self_agreement.csv` (nearest-neighbour label agreement 0.336) and
`exp_ovr_auc.csv` (Irrelevant OVR-AUC 0.648, *below* the 0.821 of always
predicting "not Irrelevant").

One finding is worth carrying forward: **Sentence-BERT + k-NN reached 48.9%,
beating an MLP head on the exact same vectors (45.3% test, 47.1% CV).** k-NN is
a pure nearest-neighbour vote with no fitted parameters, so its accuracy is a
*direct* measurement of how often a tweet's nearest labelled neighbours share
its label — the same quantity experiment E2 estimated at **0.3363** for k=10 on
the raw `phrase` view (and 0.3144 at k=25, the k used here). The two independent
measurements agree: E2 says roughly a third of nearest neighbours share a label,
and k-NN — which predicts exactly that — scores 48.9%, well below what a
well-separated four-class problem would allow. That agreement is the strongest
single piece of evidence that a large part of this test set is not recoverable
from the text.


