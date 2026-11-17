"""Combination 1 of 5: GloVe embeddings + Gradient Boosting (XGBoost).

Embedding : glove-wiki-gigaword-100 (100-d, pretrained on 6B tokens, frozen)
Pooling   : mean over in-vocabulary tokens
Model     : XGBClassifier (gradient-boosted trees)

Why this pairing: boosted trees are the strongest classical learner on dense,
low-dimensional numerical input, and 100-d GloVe is exactly that. Trees also
handle the non-linear interaction between a handful of opinion-bearing
dimensions that a linear model cannot.

The embedding is FROZEN - no fine-tuning - so this row measures what a
pre-trained vector space alone can do. Download is ~66 MB on first run.

Run: python src/embed_1_glove_xgb.py
Writes: data/processed/embedding_result_glove_xgb.json
"""
from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from embeddings_common import (SEED, DocumentVectorizer, Timer, cv_and_test,
                               load_data, report, save_result, score_block,
                               tokens_of)


def main() -> None:
    print("=" * 88)
    print("1/5  GloVe glove-wiki-gigaword-100  ->  XGBoost")
    print("=" * 88)

    # imported here so the notebook cell can define this function without
    # requiring xgboost at cell-execution time
    from xgboost import XGBClassifier
    import gensim.downloader as api
    print("  loading GloVe (cached after first download, ~66 MB) ...")
    kv = api.load("glove-wiki-gigaword-100")
    dim = kv.vector_size
    print(f"  vocabulary {len(kv):,} words | dim {dim}")

    X_tr, X_te, y_tr, y_te = load_data()
    print(f"  train {len(X_tr)} | test {len(X_te)}")

    # sanity: how much of our vocabulary does GloVe actually cover?
    toks = [t for row in tokens_of(X_tr) for t in row]
    oov = sum(1 for t in toks if t not in kv)
    print(f"  OOV rate on train tokens: {oov/len(toks)*100:.1f}% ({oov:,}/{len(toks):,})")
    empty = sum(1 for row in tokens_of(X_tr) if not any(t in kv for t in row))
    print(f"  train rows with zero in-vocab tokens: {empty}")

    def make():
        return Pipeline([
            ("vec", DocumentVectorizer(kv, dim)),
            ("scale", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.8,
                reg_lambda=1.0, min_child_weight=1,
                objective="multi:softprob", num_class=4,
                tree_method="hist", n_jobs=16, random_state=SEED,
                verbosity=0,
            )),
        ])

    with Timer() as t:
        cv_proba, test_proba, meta, _ = cv_and_test(make, X_tr, y_tr, X_te, y_te)
        cv = score_block(y_tr, cv_proba)
        test = score_block(y_te, test_proba)
        report(meta, cv, test, "GloVe + XGBoost")

    save_result("glove_xgb", "GloVe (glove-wiki-gigaword-100)",
                "Gradient Boosting (XGBoost)", cv, test,
                {"cv_accuracy_sd": meta["cv_accuracy_sd"],
                 "embedding_dim": int(dim),
                 "embedding_source": "pretrained (frozen)",
                 "pooling": "mean",
                 "runtime_s": round(t.seconds, 1),
                 "notes": f"OOV {oov/len(toks)*100:.1f}% on train tokens"})
    print("  saved -> data/processed/embedding_result_glove_xgb.json")


if __name__ == "__main__":
    main()
