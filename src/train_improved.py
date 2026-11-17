"""Phase 3iv - improved baseline selected by train-only CV.

Configuration chosen by the two train-only CV sweeps (`_sweep_linear.py`,
`_sweep2.py`):

  * view 1 ("phrase"): word 1-2 grams + char_wb 2-5 grams TF-IDF union
  * view 2 ("text_lemma"): word 1-2 grams TF-IDF
  * classifier: multinomial Logistic Regression, C=1.0, class_weight='balanced'

On train-only 5-fold CV this reaches 50.0% accuracy / 45.7% macro F1, versus
49.2% / 42.7% for the deployed M2. The test file is scored exactly once here,
after the configuration was fixed.

Run: python src/train_improved.py
Writes:
  models/improved_metrics.json
  reports/tables/improved_per_class.csv
  reports/tables/improved_predictions.csv
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from embeddings_common import MODELS, PROC, SEED, TAB, cv_and_test, score_block

WORD = dict(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True,
            strip_accents="unicode")
CHAR = dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_df=0.95,
            sublinear_tf=True, strip_accents="unicode")


def make_model() -> Pipeline:
    return Pipeline([
        ("vec", ColumnTransformer([
            ("phrase", FeatureUnion([
                ("word", TfidfVectorizer(**WORD)),
                ("char", TfidfVectorizer(**CHAR)),
            ]), "phrase"),
            ("lemma", TfidfVectorizer(**WORD), "text_lemma"),
        ])),
        ("clf", LogisticRegression(C=1.0, max_iter=4000, solver="lbfgs",
                                   class_weight="balanced", random_state=SEED)),
    ])


def main() -> None:
    train = pd.read_csv(PROC / "train_clean.csv")
    test = pd.read_csv(PROC / "test_clean.csv")
    X_tr = train[["phrase", "text_lemma"]].fillna("")
    X_te = test[["phrase", "text_lemma"]].fillna("")
    y_tr = train["sentiment"].astype(int).values
    y_te = test["sentiment"].astype(int).values

    print("=" * 88)
    print("Phase 3iv improved baseline: phrase+lemma views + LR (balanced)")
    print("=" * 88)

    cv_proba, test_proba, meta, final = cv_and_test(
        make_model, X_tr, y_tr, X_te, y_te)
    cv = score_block(y_tr, cv_proba)
    test = score_block(y_te, test_proba)

    print(f"  CV    acc={cv['accuracy']:.4f} +/-{meta['cv_accuracy_sd']:.4f}  "
          f"macroF1={cv['macro_f1']:.4f}  kappa={cv['cohen_kappa']:.4f}")
    print(f"  TEST  acc={test['accuracy']:.4f}  macroF1={test['macro_f1']:.4f}  "
          f"kappa={test['cohen_kappa']:.4f}")

    ref_path = MODELS / "final_metrics.json"
    ref = json.load(open(ref_path, encoding="utf-8")) if ref_path.exists() else None
    comparison = None
    if ref:
        dep = ref["deployed_metrics"]
        comparison = {
            "deployed_model": ref.get("deployed_model"),
            "accuracy_delta": round(test["accuracy"] - dep["accuracy"], 6),
            "macro_f1_delta": round(test["macro_f1"] - dep["macro_f1"], 6),
            "weighted_f1_delta": round(test["weighted_f1"] - dep["weighted_f1"], 6),
        }
        print(f"  delta vs deployed M2: acc {comparison['accuracy_delta']*100:+.1f} pp, "
              f"macroF1 {comparison['macro_f1_delta']*100:+.1f} pp")

    payload = {
        "model": "phrase (word+char TF-IDF) + text_lemma (word TF-IDF) + LR (C=1.0, balanced)",
        "protocol": "configuration fixed by 5-fold CV on TRAIN only; test scored once",
        "cv": cv,
        "cv_accuracy_sd": meta["cv_accuracy_sd"],
        "cv_fold_accuracies": meta["cv_fold_accuracies"],
        "test": test,
        "comparison_vs_deployed": comparison,
    }
    with open(MODELS / "improved_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("  saved -> models/improved_metrics.json")

    pd.DataFrame(test["per_class"]).to_csv(TAB / "improved_per_class.csv", index=False)
    labels = ["Negative", "Positive", "Irrelevant", "Neutral"]
    pred = test_proba.argmax(1)
    out = pd.DataFrame({
        "phrase": X_te["phrase"].values,
        "actual": [labels[i] for i in y_te],
        "predicted": [labels[i] for i in pred],
        "confidence": test_proba.max(1),
        "correct": pred == y_te,
    })
    out.to_csv(TAB / "improved_predictions.csv", index=False)
    print("  saved -> reports/tables/improved_per_class.csv, improved_predictions.csv")


if __name__ == "__main__":
    main()
