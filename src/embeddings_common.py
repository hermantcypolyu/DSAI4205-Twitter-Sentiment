"""Shared helpers for the N=5 embedding x analytics-model matrix (Weeks 11-13).

The brief requires N = number of group members = 5 analytics models, each paired
with a different *word embedding*. Two hard constraints from the professor:

  * TF-IDF is NOT an embedding, so it cannot be one of the five.
  * Logistic Regression is reserved for the Phase 3i baseline, so it cannot be
    reused either.

This module gives every one of the five scripts the same protocol so the
comparison is fair:

  * identical rows, identical folds (StratifiedKFold, shuffle, seed 42)
  * CV on TRAIN only -- test is scored exactly once per model
  * the same metric block as src/train_final.py (score_block)

Two embedding sources exist, and they need different adapters:

  1. **Dense document vectors** (GloVe / Word2Vec / FastText / Sentence-BERT).
     Every row becomes one fixed-length vector, so a classic sklearn estimator
     can consume it directly. `DocumentVectorizer` wraps the pooler as a
     sklearn transformer so it can sit inside a Pipeline.

  2. **Fine-tuned encoders** (DistilBERT). The model takes token ids, not a
     precomputed matrix, so those scripts use
     `train_final.transformer_fit_predict` directly instead.

Pooling note: mean pooling is used rather than max or SIF. With tweets of ~20
tokens, mean pooling is the standard baseline and needs no extra
hyper-parameters, which keeps the five models comparable.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TAB = ROOT / "reports" / "tables"
FIG = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
for _d in (TAB, FIG, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

LABELS = ["Negative", "Positive", "Irrelevant", "Neutral"]
PALETTE = ["#D62728", "#2CA02C", "#7F7F7F", "#1F77B4"]
SEED = 42
N_SPLITS = 5

# The five eligible combinations, in the order the report declares them.
MATRIX_ROWS = [
    (1, "GloVe (glove-wiki-gigaword-100)", "Gradient Boosting (XGBoost)"),
    (2, "Word2Vec (trained from scratch)", "Random Forest"),
    (3, "FastText (subword)", "Support Vector Machine"),
    (4, "BERT (fine-tuned)", "Neural classifier head"),
    (5, "Sentence-BERT (all-MiniLM-L6-v2)", "k-Nearest-Neighbours"),
]


# --------------------------------------------------------------------- data
def load_data(text_col: str = "phrase"):
    """Load the processed splits. `phrase` is raw text and is the canonical view."""
    train = pd.read_csv(PROC / "train_clean.csv")
    test = pd.read_csv(PROC / "test_clean.csv")
    X_tr = train[text_col].fillna("").astype(str)
    X_te = test[text_col].fillna("").astype(str)
    y_tr = train["sentiment"].astype(int).values
    y_te = test["sentiment"].astype(int).values
    return X_tr, X_te, y_tr, y_te


def tokens_of(texts) -> list[list[str]]:
    """Lower-case whitespace split.

    Deliberately simple: the point of these models is the *embedding*, so the
    tokenisation must not advantage one embedding over another. Char-ngram
    handling is left to FastText, which does it internally.
    """
    return [[t for t in str(s).lower().split() if t.strip()] for s in texts]


# ------------------------------------------------------------------ pooling
def mean_pool(tokens: list[str], kv, dim: int) -> np.ndarray:
    """Average the in-vocabulary word vectors.

    Rows whose tokens are all out-of-vocabulary get an all-zero vector. Such a
    row carries no information, and a zero vector is the honest representation
    of "nothing known" -- much better than dropping the row, which would silently
    change the class balance between train and test.
    """
    vecs = [kv[t] for t in tokens if t in kv]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


def embed_corpus(texts, kv, dim: int) -> np.ndarray:
    return np.vstack([mean_pool(tok, kv, dim) for tok in tokens_of(texts)])


class DocumentVectorizer(BaseEstimator, TransformerMixin):
    """sklearn adapter: raw strings -> mean-pooled embedding matrix.

    Fitting is a no-op *by design*. The embedding must be fitted on train text
    only (Word2Vec/FastText) and that is done by the caller before this object
    is constructed. Keeping `fit` a no-op guarantees the wrapper can never
    accidentally re-fit on validation or test tokens inside a CV loop.
    """

    def __init__(self, kv, dim: int):
        self.kv = kv
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return embed_corpus(X, self.kv, self.dim)


# ------------------------------------------------------------------ metrics
def score_block(y_true, proba) -> dict:
    """Same metric block as train_final.py, so numbers are directly comparable."""
    from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                                 precision_recall_fscore_support)
    pred = np.asarray(proba).argmax(1)
    p, r, f, s = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2, 3],
                                                 zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted")),
        "balanced_accuracy": float(np.mean(r)),
        "cohen_kappa": float(cohen_kappa_score(y_true, pred)),
        "within1_accuracy": float((np.abs(y_true - pred) <= 1).mean()),
        "per_class": [{"class": LABELS[i], "precision": float(p[i]), "recall": float(r[i]),
                       "f1": float(f[i]), "support": int(s[i])} for i in range(4)],
    }


def proba_of(estimator, X) -> np.ndarray:
    """4-column probabilities, softmaxing decision_function when needed (SVM)."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)
    m = estimator.decision_function(X)
    m = np.atleast_2d(m)
    e = np.exp(m - m.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cv_and_test(make_estimator, X_tr, y_tr, X_te, y_te, n_splits: int = N_SPLITS):
    """TRAIN-only CV, then one test evaluation. Returns (cv_proba, test_proba, meta).

    `make_estimator` must return a *fresh unfitted* estimator each call.
    """
    import copy
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    cv_proba = np.zeros((len(y_tr), 4))
    fold_acc = []
    for tr, va in skf.split(X_tr, y_tr):
        est = copy.deepcopy(make_estimator())
        est.fit(X_tr.iloc[tr] if hasattr(X_tr, "iloc") else X_tr[tr], y_tr[tr])
        p = proba_of(est, X_tr.iloc[va] if hasattr(X_tr, "iloc") else X_tr[va])
        cv_proba[va] = p
        fold_acc.append(float((p.argmax(1) == y_tr[va]).mean()))

    final = copy.deepcopy(make_estimator())
    final.fit(X_tr, y_tr)
    test_proba = proba_of(final, X_te)

    meta = {"cv_accuracy_sd": float(np.std(fold_acc)),
            "cv_fold_accuracies": [round(a, 6) for a in fold_acc]}
    return cv_proba, test_proba, meta, final


# ------------------------------------------------------------------ results
def save_result(name: str, embedding: str, model: str, cv: dict, test: dict,
                extra: dict, out_dir: Path | None = None) -> Path:
    """Persist one row of the matrix as its own JSON, keyed by script name."""
    out_dir = out_dir or PROC
    payload = {"id": None, "embedding": embedding, "analytics_model": model,
               "cv": cv, "test": test, **extra}
    # fill in the declared matrix id where the pairing matches
    for idx, emb, mod in MATRIX_ROWS:
        if emb.split(" (")[0].lower() in embedding.lower():
            payload["id"] = idx
            break
    path = out_dir / f"embedding_result_{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def report(train_meta: dict, cv: dict, test: dict, label: str) -> None:
    """Uniform console output: CV first, then the single test evaluation."""
    print(f"\n  [{label}]")
    print(f"    CV    acc={cv['accuracy']:.4f} +/-{train_meta['cv_accuracy_sd']:.4f}  "
          f"macroF1={cv['macro_f1']:.4f}  kappa={cv['cohen_kappa']:.4f}")
    print(f"    TEST  acc={test['accuracy']:.4f}  macroF1={test['macro_f1']:.4f}  "
          f"kappa={test['cohen_kappa']:.4f}")


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0
        print(f"    runtime {self.seconds:.1f}s")
