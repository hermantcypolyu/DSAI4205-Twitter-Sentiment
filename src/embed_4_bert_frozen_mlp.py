"""Combination 4 of 5: frozen DistilBERT embeddings + independent MLP classifier.

Embedding : distilbert-base-uncased (768-d), FROZEN - no fine-tuning.
Pooling   : masked mean over the last hidden state.
Model     : MLPClassifier (same 512 -> 256 architecture as the GloVe row).

Why this replaces the end-to-end fine-tuned row
-----------------------------------------------
The brief's integration requirement says a single model may not serve as both
the embedding model and the analytics classifier ("for instance, employing BERT
for both embedding and classification"). End-to-end fine-tuning trains the
encoder and the head jointly, so it does not satisfy that rule. Here the encoder
is frozen and used purely as an embedding extractor; the MLP is a separately
fitted analytics classifier. Only the embedding changes relative to row 1, so
the pair isolates the value of a contextual vector space.

The earlier end-to-end result is retained in
`data/processed/embedding_result_bert_head.json` as an upper-bound reference
and is explicitly NOT counted toward N = 5.

Run: python src/embed_4_bert_frozen_mlp.py
Writes: data/processed/embedding_result_bert_frozen_mlp.json
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from embeddings_common import (SEED, Timer, cv_and_test, load_data, report,
                               save_result, score_block)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64


def main() -> None:
    print("=" * 88)
    print(f"4/5  {MODEL_NAME} (frozen)  ->  Neural Network (MLP)")
    print("=" * 88)

    # imported here so the notebook cell can define this function without
    # requiring torch/transformers at cell-execution time
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    X_tr, X_te, y_tr, y_te = load_data()
    print(f"  train {len(X_tr)} | test {len(X_te)}")

    @torch.no_grad()
    def encode(texts, batch_size: int = 64) -> np.ndarray:
        chunks = []
        texts = list(texts)
        for i in range(0, len(texts), batch_size):
            batch = tok(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            chunks.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(chunks)

    with Timer() as t_enc:
        E_tr = encode(X_tr)
        E_te = encode(X_te)
    print(f"  embeddings train {E_tr.shape} | test {E_te.shape}")

    def make():
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(512, 256), activation="relu", solver="adam",
                alpha=1e-3, batch_size=64, learning_rate_init=1e-3,
                max_iter=1200, early_stopping=True, validation_fraction=0.12,
                n_iter_no_change=25, random_state=SEED,
            )),
        ])

    with Timer() as t:
        cv_proba, test_proba, meta, _ = cv_and_test(make, E_tr, y_tr, E_te, y_te)
        cv = score_block(y_tr, cv_proba)
        test = score_block(y_te, test_proba)
        report(meta, cv, test, "DistilBERT frozen + MLP")

    save_result("bert_frozen_mlp", "DistilBERT (frozen)", "Neural Network (MLP)",
                cv, test,
                {"cv_accuracy_sd": meta["cv_accuracy_sd"],
                 "embedding_dim": 768,
                 "embedding_source": f"{MODEL_NAME}, frozen",
                 "pooling": "masked mean of last hidden state",
                 "runtime_s": round(t_enc.seconds + t.seconds, 1),
                 "notes": "encoder frozen; MLP 512-256 trained separately"})
    print("  saved -> data/processed/embedding_result_bert_frozen_mlp.json")


if __name__ == "__main__":
    main()
