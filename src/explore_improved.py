"""Train-only CV sweeps for Phase 3iv (baseline refinement).

Every configuration is scored by 5-fold stratified CV on the TRAIN file only.
The test file is never read here. The winning configuration is then evaluated
exactly once by `train_improved.py`.

Two families are explored:
  A. TF-IDF shape: n-gram ranges, min_df, regularisation strength, class weights
  B. Ensembles: LR + calibrated LinearSVC + ComplementNB, and a two-column
     phrase + lemma view (the configuration finally selected)

Run: python src/explore_improved.py
"""
from __future__ import annotations

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from embeddings_common import PROC, SEED, TAB

tr = pd.read_csv(PROC / "train_clean.csv")
X = tr["phrase"].fillna("").astype(str)
y = tr["sentiment"].astype(int).values
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)


def word(n1=1, n2=2, min_df=1, max_df=0.95):
    return TfidfVectorizer(ngram_range=(n1, n2), min_df=min_df, max_df=max_df,
                           sublinear_tf=True, strip_accents="unicode")


def char(n1=2, n2=5, min_df=2, max_df=0.95):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(n1, n2), min_df=min_df,
                           max_df=max_df, sublinear_tf=True, strip_accents="unicode")


def lr(c=1.0, balanced=True):
    return LogisticRegression(C=c, max_iter=4000, solver="lbfgs",
                              class_weight="balanced" if balanced else None,
                              random_state=SEED)


def union():
    return FeatureUnion([("word", word()), ("char", char())])


CASES = {
    "w12+cwb25 LR C1 (M2)": (union(), lr(1.0, balanced=False)),
    "w12+cwb26 df1 LR C1": (FeatureUnion([("w", word(1, 2, 1)), ("c", char(2, 6, 1))]), lr(1.0, False)),
    "w13+cwb26 df1 LR C1": (FeatureUnion([("w", word(1, 3, 1)), ("c", char(2, 6, 1))]), lr(1.0, False)),
    "w12+cwb25 LR C2": (union(), lr(2.0, False)),
    "w12+cwb25 LR C0.5": (union(), lr(0.5, False)),
    "w12+cwb25 LR balanced": (union(), lr(1.0, True)),
    "balanced LR C0.5": (union(), lr(0.5, True)),
    "balanced LR C2": (union(), lr(2.0, True)),
    "balanced LR C4": (union(), lr(4.0, True)),
    "Calibrated LinearSVC": (union(), CalibratedClassifierCV(
        LinearSVC(C=1.0, random_state=SEED), cv=3)),
    "SGD log_loss": (union(), SGDClassifier(loss="log_loss", alpha=1e-4,
                                            max_iter=2000, random_state=SEED)),
    "vote LR+SVC": (union(), VotingClassifier([
        ("lr", lr(1.0)), ("svc", CalibratedClassifierCV(
            LinearSVC(C=1.0, random_state=SEED), cv=3))], voting="soft")),
    "vote LR+NB": (union(), VotingClassifier([
        ("lr", lr(1.0)), ("nb", ComplementNB(alpha=0.5))], voting="soft")),
    "vote LR+SVC+NB": (union(), VotingClassifier([
        ("lr", lr(1.0)),
        ("svc", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=SEED), cv=3)),
        ("nb", ComplementNB(alpha=0.5))], voting="soft")),
}

rows = []
for name, (vec, clf) in CASES.items():
    pipe = Pipeline([("vec", vec), ("clf", clf)])
    acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy").mean()
    f1 = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro").mean()
    rows.append((name, acc, f1))
    print(f"  {name:28} acc={acc:.4f}  macroF1={f1:.4f}", flush=True)

# two-column view: phrase (word+char) + lemmatised word features
Xdf = tr[["phrase", "text_lemma"]].fillna("")
two_col = ColumnTransformer([
    ("phrase", union(), "phrase"),
    ("lemma", word(), "text_lemma"),
])
pipe = Pipeline([("vec", two_col), ("clf", lr(1.0, True))])
acc = cross_val_score(pipe, Xdf, y, cv=skf, scoring="accuracy").mean()
f1 = cross_val_score(pipe, Xdf, y, cv=skf, scoring="f1_macro").mean()
rows.append(("phrase+lemma union (SELECTED)", acc, f1))
print(f"  {'phrase+lemma union (SELECTED)':28} acc={acc:.4f}  macroF1={f1:.4f}", flush=True)

with open(TAB / "improved_cv_sweep.csv", "w", newline="", encoding="utf-8") as f:
    import csv
    w = csv.writer(f)
    w.writerow(["configuration", "cv_accuracy", "cv_macro_f1"])
    w.writerows(rows)
print(f"  saved -> {TAB / 'improved_cv_sweep.csv'}")

print("\nBest by accuracy:")
for name, acc, f1 in sorted(rows, key=lambda r: -r[1])[:5]:
    print(f"  {name:28} acc={acc:.4f}  macroF1={f1:.4f}")
print("Best by macro F1:")
for name, acc, f1 in sorted(rows, key=lambda r: -r[2])[:5]:
    print(f"  {name:28} acc={acc:.4f}  macroF1={f1:.4f}")
