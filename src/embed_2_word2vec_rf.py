"""Combination 2 of 5: Word2Vec (trained from scratch) + Random Forest.

Embedding : Word2Vec skip-gram, 100-d, trained ONLY on the 592 training tweets.
            Nothing is loaded from the internet - this is the "train your own
            embedding" bonus criterion.
Model     : RandomForestClassifier (500 trees)

Why this pairing: an embedding learned from 592 tweets captures the domain
vocabulary (slang, handles, misspellings) that GloVe has never seen. A forest
is a high-variance learner, and the classic reason to pair it with a weak,
noisy feature space is that bagging over 500 trees averages that noise out. The
pairing therefore tests a specific claim: *does a domain-specific but tiny
corpus beat a large generic one?*

Leakage guard: Word2Vec is fitted on `tokens_of(train)` only, before CV, and
`DocumentVectorizer.fit` is a no-op, so test tokens can never influence the
vector space.

Run: python src/embed_2_word2vec_rf.py
Writes: data/processed/embedding_result_word2vec_rf.json
"""
from __future__ import annotations

from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from embeddings_common import (SEED, DocumentVectorizer, Timer, cv_and_test,
                               load_data, report, save_result, score_block,
                               tokens_of)


def main() -> None:
    print("=" * 88)
    print("2/5  Word2Vec (trained from scratch on 592 tweets)  ->  Random Forest")
    print("=" * 88)

    X_tr, X_te, y_tr, y_te = load_data()
    print(f"  train {len(X_tr)} | test {len(X_te)}")

    # ---- embedding fitted on TRAIN tokens only (leakage guard) --------------
    train_tokens = tokens_of(X_tr)
    vocab = {t for row in train_tokens for t in row}
    print(f"  training corpus: {len(train_tokens)} tweets, {len(vocab):,} distinct tokens")

    w2v = Word2Vec(
        sentences=train_tokens,        # <-- train only, never X_te
        vector_size=100,
        window=5,
        min_count=1,
        sg=1,                          # skip-gram: better for rare words
        negative=10,
        epochs=30,
        workers=16,
        seed=SEED,
        sample=1e-4,
    )
    dim = w2v.wv.vector_size
    print(f"  learned vocabulary: {len(w2v.wv):,} vectors | dim {dim}")

    def make():
        return Pipeline([
            ("vec", DocumentVectorizer(w2v.wv, dim)),
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=500, max_features="sqrt", min_samples_leaf=1,
                n_jobs=16, random_state=SEED,
            )),
        ])

    with Timer() as t:
        cv_proba, test_proba, meta, _ = cv_and_test(make, X_tr, y_tr, X_te, y_te)
        cv = score_block(y_tr, cv_proba)
        test = score_block(y_te, test_proba)
        report(meta, cv, test, "Word2Vec + Random Forest")

    test_cov = sum(1 for row in tokens_of(X_te) if any(t in w2v.wv for t in row))
    print(f"  test rows with >=1 in-vocab token: {test_cov}/{len(X_te)}")

    save_result("word2vec_rf", "Word2Vec (trained from scratch)", "Random Forest",
                cv, test,
                {"cv_accuracy_sd": meta["cv_accuracy_sd"],
                 "embedding_dim": int(dim),
                 "embedding_source": "self-trained on 592 train tweets",
                 "pooling": "mean",
                 "runtime_s": round(t.seconds, 1),
                 "notes": f"{len(w2v.wv):,} learned vectors; {test_cov}/{len(X_te)} test rows covered"})
    print("  saved -> data/processed/embedding_result_word2vec_rf.json")


if __name__ == "__main__":
    main()
