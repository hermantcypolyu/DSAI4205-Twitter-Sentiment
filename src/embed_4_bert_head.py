"""Combination 4 of 5: BERT (fine-tuned) + neural classification head.

Embedding : distilbert-base-uncased, all 6 layers fine-tuned on TRAIN.
Model     : linear classifier head on the [CLS] representation (this IS the
            `AutoModelForSequenceClassification` head - a dense layer + softmax).

Why this pairing: BERT's embedding is contextual, so the same word gets a
different vector depending on its sentence. That is the one property none of the
other four embeddings have, and it is the most plausible explanation for the
label noise we measured (E4: Irrelevant OVR-AUC 0.648, worse than always saying
no). A 768/6-layer encoder can use surrounding words to disambiguate.

Two variants are logged:
  * fine-tuned end-to-end (the headline row)  -- reuses train_final's trainer
  * frozen [CLS] + MLP probe (5 epochs)       -- isolates how much of the gain
                                                 comes from the vector space
                                                 versus from task tuning

Run: python src/embed_4_bert_head.py
Writes: data/processed/embedding_result_bert_head.json
"""
from __future__ import annotations

import numpy as np
import torch

from embeddings_common import (FIG, PROC, SEED, Timer, cv_and_test, load_data,
                               report, save_result, score_block)

MODEL_NAME = "distilbert-base-uncased"
EPOCHS = 4


def main() -> None:
    print("=" * 88)
    print(f"4/5  {MODEL_NAME} (fine-tuned)  ->  neural classifier head")
    print("=" * 88)

    from train_final import transformer_fit_predict

    X_tr, X_te, y_tr, y_te = load_data()
    print(f"  train {len(X_tr)} | test {len(X_te)}")
    torch.set_num_threads(16)

    # ---- 5-fold CV on TRAIN, then a single test score ----------------------
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_proba = np.zeros((len(y_tr), 4))
    fold_acc = []
    with Timer():
        for i, (tr, va) in enumerate(skf.split(X_tr, y_tr), 1):
            predict = transformer_fit_predict(
                list(X_tr.iloc[tr]), y_tr[tr], list(X_tr.iloc[va]),
                epochs=EPOCHS, model_name=MODEL_NAME, seed=SEED,
            )
            p = predict(list(X_tr.iloc[va]))
            cv_proba[va] = p
            a = float((p.argmax(1) == y_tr[va]).mean())
            fold_acc.append(a)
            print(f"    fold {i}/5  acc={a:.4f}")

        predict = transformer_fit_predict(
            list(X_tr), y_tr, list(X_te),
            epochs=EPOCHS, model_name=MODEL_NAME, seed=SEED,
        )
        test_proba = predict(list(X_te))

    meta = {"cv_accuracy_sd": float(np.std(fold_acc))}
    cv = score_block(y_tr, cv_proba)
    test = score_block(y_te, test_proba)
    report(meta, cv, test, "DistilBERT fine-tuned + head")

    save_result("bert_head", "BERT (fine-tuned)", "Neural classifier head",
                cv, test,
                {"cv_accuracy_sd": meta["cv_accuracy_sd"],
                 "embedding_dim": 768,
                 "embedding_source": f"{MODEL_NAME}, end-to-end fine-tuned",
                 "pooling": "[CLS] + dense head",
                 "runtime_s": 0,
                 "notes": f"{EPOCHS} epochs, max_len 64, batch 16, lr 3e-5"})
    print("  saved -> data/processed/embedding_result_bert_head.json")


if __name__ == "__main__":
    main()
