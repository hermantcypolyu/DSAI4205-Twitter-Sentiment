"""Combination 3 of 5: FastText (subword) + Support Vector Machine.

Embedding : FastText with character n-grams (min_n=3, max_n=6), 100-d, trained
            ONLY on the 592 training tweets.
Model     : SVC(kernel="rbf") on standardised vectors, probabilities via Platt.

Why this pairing: FastText represents each word as the sum of its character
3-6-grams, so it can build a vector for a word it never saw - exactly the
situation with tweets ("luvvv", "sooo", "#awkward"). It is the single most
appropriate embedding for this dataset's spelling noise, and it is worth
checking whether that subword robustness beats GloVe's larger but exact-match
vocabulary.

An RBF SVM is chosen over a linear one because the subword space is noisier and
not linearly separable; the kernel can carve curved decision regions around the
messy middle class ("Irrelevant"). probability=True adds internal CV for Platt
scaling - that CV is nested strictly inside TRAIN.

Run: python src/embed_3_fasttext_svm.py
Writes: data/processed/embedding_result_fasttext_svm.json
"""
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from embeddings_common import (SEED, DocumentVectorizer, Timer, cv_and_test,
                               load_data, report, save_result, score_block,
                               tokens_of)


def main() -> None:
    print("=" * 88)
    print("3/5  FastText (char 3-6-grams, trained from scratch)  ->  SVM (RBF)")
    print("=" * 88)

    # imported here so the notebook cell can define this function without
    # requiring gensim to be installed (it has no Python 3.14 wheel)
    from gensim.models import FastText

    X_tr, X_te, y_tr, y_te = load_data()
    print(f"  train {len(X_tr)} | test {len(X_te)}")

    train_tokens = tokens_of(X_tr)          # <-- train only, leakage guard
    print(f"  training corpus: {len(train_tokens)} tweets")

    ft = FastText(
        sentences=train_tokens,
        vector_size=100,
        window=5,
        min_count=1,
        min_n=3, max_n=6,                   # subword range: captures stems/suffixes
        sg=1,
        negative=10,
        epochs=30,
        workers=16,
        seed=SEED,
    )
    dim = ft.wv.vector_size
    print(f"  learned vocabulary: {len(ft.wv):,} vectors | dim {dim}")

    def make():
        return Pipeline([
            ("vec", DocumentVectorizer(ft.wv, dim)),
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale",
                        probability=True, random_state=SEED)),
        ])

    with Timer() as t:
        cv_proba, test_proba, meta, _ = cv_and_test(make, X_tr, y_tr, X_te, y_te)
        cv = score_block(y_tr, cv_proba)
        test = score_block(y_te, test_proba)
        report(meta, cv, test, "FastText + SVM(RBF)")

    # subword generalisation: how many test rows contain a never-seen word?
    seen = set(ft.wv.key_to_index)
    test_rows_oov = sum(1 for row in tokens_of(X_te) if any(t not in seen for t in row))
    print(f"  test rows containing an unseen word: {test_rows_oov}/{len(X_te)} "
          f"(FastText still embeds these via subwords)")

    save_result("fasttext_svm", "FastText (subword)", "Support Vector Machine",
                cv, test,
                {"cv_accuracy_sd": meta["cv_accuracy_sd"],
                 "embedding_dim": int(dim),
                 "embedding_source": "self-trained on 592 train tweets, char 3-6-grams",
                 "pooling": "mean",
                 "runtime_s": round(t.seconds, 1),
                 "notes": f"{len(ft.wv):,} learned vectors; subword handles OOV"})
    print("  saved -> data/processed/embedding_result_fasttext_svm.json")


if __name__ == "__main__":
    main()
