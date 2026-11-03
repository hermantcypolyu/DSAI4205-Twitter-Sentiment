"""Combination 5 of 5: Sentence-BERT + k-Nearest-Neighbours.

Embedding : all-MiniLM-L6-v2 (384-d), a sentence-transformer. It is used FROZEN -
            it was trained on 1B sentence pairs to place similar *sentences* near
            each other, which is a different objective from word vectors.
Model     : KNeighborsClassifier with cosine distance, k=25.

Why this pairing: kNN is the only one of the five that makes no assumption about
the shape of the decision boundary. It predicts by majority vote among the
nearest labelled tweets, so its accuracy is a direct measurement of a claim we
already made with experiment E2: *if neighbours in embedding space share labels,
kNN must succeed; if the same-label fraction is only ~0.33, no amount of model
cleverness can help.* This row is therefore the cleanest test of the label-noise
ceiling, not just another classifier.

A cosine metric is used because SBERT vectors are normalised for cosine
similarity, and k=25 (~4% of train) smooths the vote without over-smoothing.

Also logged: an MLP variant on the same vectors, to separate "the embedding is
bad" from "the neighbour vote is bad".

Run: python src/embed_5_sbert_knn.py
Writes: data/processed/embedding_result_sbert_knn.json
"""
from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from embeddings_common import (SEED, Timer, cv_and_test, load_data, report,
                               save_result, score_block)

MODEL_NAME = "all-MiniLM-L6-v2"
K = 25


def main() -> None:
    print("=" * 88)
    print(f"5/5  Sentence-BERT {MODEL_NAME} (384-d, frozen)  ->  kNN (k={K}, cosine)")
    print("=" * 88)

    from sentence_transformers import SentenceTransformer

    X_tr, X_te, y_tr, y_te = load_data()
    print(f"  train {len(X_tr)} | test {len(X_te)}")

    print(f"  encoding with {MODEL_NAME} ...")
    enc = SentenceTransformer(MODEL_NAME, device="cpu")
    dim = enc.get_sentence_embedding_dimension()
    E_tr = enc.encode(list(X_tr), batch_size=64, show_progress_bar=False,
                      convert_to_numpy=True, normalize_embeddings=True)
    E_te = enc.encode(list(X_te), batch_size=64, show_progress_bar=False,
                      convert_to_numpy=True, normalize_embeddings=True)
    print(f"  embeddings train {E_tr.shape} | test {E_te.shape}")

    # ---- kNN: pure neighbour vote, no fitted parameters ---------------------
    def make_knn():
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=K, metric="cosine",
                                         weights="distance", n_jobs=16)),
        ])

    with Timer() as t1:
        cv_p, te_p, meta, _ = cv_and_test(make_knn, E_tr, y_tr, E_te, y_te)
        cv = score_block(y_tr, cv_p)
        test = score_block(y_te, te_p)
        report(meta, cv, test, f"SBERT + kNN(k={K})")

    save_result("sbert_knn", "Sentence-BERT (all-MiniLM-L6-v2)",
                "k-Nearest-Neighbours", cv, test,
                {"cv_accuracy_sd": meta["cv_accuracy_sd"],
                 "embedding_dim": int(dim),
                 "embedding_source": f"{MODEL_NAME}, frozen",
                 "pooling": "SBERT sentence encoder",
                 "runtime_s": round(t1.seconds, 1),
                 "notes": f"k={K}, cosine, distance-weighted vote"})
    print("  saved -> data/processed/embedding_result_sbert_knn.json")

    # ---- MLP variant: same vectors, learned boundary -----------------------
    print("\n  -- variant: same embeddings, MLP head --")
    from sklearn.model_selection import cross_val_score
    mlp = Pipeline([
        ("scale", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3,
                              max_iter=600, early_stopping=True,
                              random_state=SEED)),
    ])
    mlp_cv = cross_val_score(mlp, E_tr, y_tr, cv=5, scoring="accuracy").mean()
    mlp.fit(E_tr, y_tr)
    mlp_test = float((mlp.predict(E_te) == y_te).mean())
    print(f"    MLP   CV acc={mlp_cv:.4f}   TEST acc={mlp_test:.4f}")


if __name__ == "__main__":
    main()
