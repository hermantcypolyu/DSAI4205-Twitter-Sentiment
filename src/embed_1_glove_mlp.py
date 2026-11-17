"""Combination 1 of 5: GloVe embeddings + Neural Network (multilayer perceptron).

Embedding : glove-wiki-gigaword-100 (100-d, pretrained on 6B tokens, frozen)
Pooling   : mean over in-vocabulary tokens
Model     : MLPClassifier (2 hidden layers, 256 -> 128, ReLU, early stopping)

Why this pairing replaces the earlier GloVe + XGBoost row
---------------------------------------------------------
The consultation feedback asked us to keep only one tree-ensemble model and
replace the other with a Neural Network. The final N = 5 matrix therefore keeps
Random Forest (Word2Vec + RF) as its single tree model and uses this MLP as the
neural analytics model on the GloVe space. That gives five genuinely distinct
families: shallow neural network, bagged trees, kernel SVM, fine-tuned
transformer, and non-parametric nearest neighbours.

The embedding stays FROZEN - no fine-tuning - so this row measures what a
pre-trained vector space plus a learned non-linear boundary can do. The GloVe
archive is read directly from the local cache (no gensim dependency); the
download is ~66 MB compressed on first run.

Run: python src/embed_1_glove_mlp.py
Writes: data/processed/embedding_result_glove_mlp.json
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from embeddings_common import (ROOT, SEED, DocumentVectorizer, Timer, cv_and_test,
                               load_data, report, save_result, score_block,
                               tokens_of)

GLOVE_GZ = ROOT / ".cache" / "glove-wiki-gigaword-100.gz"
GLOVE_URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/download/"
             "glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz")


def load_glove(path: Path = GLOVE_GZ) -> tuple[dict[str, np.ndarray], int]:
    """Parse the gzipped GloVe text file into {word: float32 vector}.

    A plain dict is enough because `mean_pool` only needs `word in kv` and
    `kv[word]`, so we avoid the heavy gensim build on Python 3.14.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it with:\n  curl -L -o {path} {GLOVE_URL}")
    kv: dict[str, np.ndarray] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 3:
                continue
            # Some archives carry a "400000 100" header line; skip it.
            if len(parts) == 2 and parts[0].isdigit():
                continue
            kv[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    dim = len(next(iter(kv.values())))
    return kv, dim


def main() -> None:
    print("=" * 88)
    print("1/5  GloVe glove-wiki-gigaword-100  ->  Neural Network (MLP)")
    print("=" * 88)

    print(f"  loading GloVe from {GLOVE_GZ} ...")
    kv, dim = load_glove()
    print(f"  vocabulary {len(kv):,} words | dim {dim}")

    X_tr, X_te, y_tr, y_te = load_data()
    print(f"  train {len(X_tr)} | test {len(X_te)}")

    # sanity: how much of our vocabulary does GloVe actually cover?
    toks = [t for row in tokens_of(X_tr) for t in row]
    oov = sum(1 for t in toks if t not in kv)
    print(f"  OOV rate on train tokens: {oov/len(toks)*100:.1f}% ({oov:,}/{len(toks):,})")

    def make():
        return Pipeline([
            ("vec", DocumentVectorizer(kv, dim)),
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(512, 256), activation="relu", solver="adam",
                alpha=1e-3, batch_size=64, learning_rate_init=1e-3,
                max_iter=1200, early_stopping=True, validation_fraction=0.12,
                n_iter_no_change=25, random_state=SEED,
            )),
        ])

    with Timer() as t:
        cv_proba, test_proba, meta, final = cv_and_test(make, X_tr, y_tr, X_te, y_te)
        cv = score_block(y_tr, cv_proba)
        test = score_block(y_te, test_proba)
        report(meta, cv, test, "GloVe + MLP")

    save_result("glove_mlp", "GloVe (glove-wiki-gigaword-100)",
                "Neural Network (MLP)", cv, test,
                {"cv_accuracy_sd": meta["cv_accuracy_sd"],
                 "embedding_dim": int(dim),
                 "embedding_source": "pretrained (frozen)",
                 "pooling": "mean",
                 "runtime_s": round(t.seconds, 1),
                 "notes": f"MLP 256-128; OOV {oov/len(toks)*100:.1f}% on train tokens"})
    print("  saved -> data/processed/embedding_result_glove_mlp.json")


if __name__ == "__main__":
    main()
