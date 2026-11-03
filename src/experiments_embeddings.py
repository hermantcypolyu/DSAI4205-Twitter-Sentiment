"""Aggregate the N=5 embedding x analytics-model matrix.

Reads the five `data/processed/embedding_result_*.json` files, appends the
TF-IDF reference row (explicitly NOT eligible for N=5, because TF-IDF is not an
embedding), writes the CSV/JSON/figures, and prints a paste-ready markdown table.

Run: python src/experiments_embeddings.py
Writes:
  reports/tables/embedding_model_matrix.csv
  reports/figures/fig15_embedding_matrix.png
  reports/figures/fig16_embedding_families.png
  data/processed/embedding_matrix_summary.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from embeddings_common import FIG, MODELS, PALETTE, PROC, TAB

RESULTS = [
    ("glove_xgb", "GloVe (glove-wiki-gigaword-100)", "Gradient Boosting (XGBoost)", True),
    ("word2vec_rf", "Word2Vec (trained from scratch)", "Random Forest", True),
    ("fasttext_svm", "FastText (subword)", "Support Vector Machine", True),
    ("bert_head", "BERT (fine-tuned)", "Neural classifier head", True),
    ("sbert_knn", "Sentence-BERT (all-MiniLM-L6-v2)", "k-Nearest-Neighbours", True),
]

COLS = ["id", "embedding", "analytics_model", "embedding_source", "dim", "cv_accuracy",
        "cv_macro_f1", "cv_sd", "test_accuracy", "test_macro_f1", "test_weighted_f1",
        "kappa", "within1", "runtime_s", "eligible_for_n5", "role", "notes"]


def load_rows() -> pd.DataFrame:
    rows = []
    for i, (key, emb, mod, elig) in enumerate(RESULTS, 1):
        p = PROC / f"embedding_result_{key}.json"
        if not p.exists():
            print(f"  MISSING {p.name} - run src/embed_{i}_*.py first")
            continue
        d = json.load(open(p, encoding="utf-8"))
        cv, te = d["cv"], d["test"]
        rows.append({
            "id": i, "embedding": emb, "analytics_model": mod,
            "embedding_source": d.get("embedding_source", ""),
            "dim": d.get("embedding_dim", 0),
            "cv_accuracy": cv["accuracy"], "cv_macro_f1": cv["macro_f1"],
            "cv_sd": d.get("cv_accuracy_sd", np.nan),
            "test_accuracy": te["accuracy"], "test_macro_f1": te["macro_f1"],
            "test_weighted_f1": te["weighted_f1"],
            "kappa": te["cohen_kappa"], "within1": te["within1_accuracy"],
            "runtime_s": d.get("runtime_s", np.nan),
            "eligible_for_n5": elig, "role": "candidate",
            "notes": d.get("notes", ""),
        })
    return pd.DataFrame(rows)


def add_tfidf_reference(df: pd.DataFrame) -> pd.DataFrame:
    """TF-IDF is the deployed model and the reference point, but cannot count."""
    p = MODELS / "final_metrics.json"
    if not p.exists():
        print("  final_metrics.json missing - skipping TF-IDF reference row")
        return df
    fm = json.load(open(p, encoding="utf-8"))
    dep = fm["deployed_metrics"]
    # cv_train_only is a LIST of per-model dicts, not a single dict
    cv_rows = {r["model"]: r for r in fm.get("cv_train_only", [])}
    deployed_name = fm.get("deployed_model", "")
    cvb = cv_rows.get(deployed_name, {})
    if not cvb or "accuracy" not in cvb:
        print(f"  note: no CV row for {deployed_name!r}; leaving TF-IDF CV blank")
        cvb = {}
    ref = {
        "id": 0, "embedding": "TF-IDF (word 1-2gram + char_wb 2-5gram)",
        "analytics_model": "Logistic Regression (DEPLOYED baseline)",
        "embedding_source": "sparse count-based, fitted on train",
        "dim": np.nan,
        "cv_accuracy": cvb.get("accuracy", np.nan),
        "cv_macro_f1": cvb.get("macro_f1", np.nan), "cv_sd": np.nan,
        "test_accuracy": dep["accuracy"], "test_macro_f1": dep["macro_f1"],
        "test_weighted_f1": dep["weighted_f1"],
        "kappa": dep["cohen_kappa"], "within1": dep["within1_accuracy"],
        "runtime_s": fm.get("runtime_seconds", np.nan),
        "eligible_for_n5": False, "role": "reference (NOT counted toward N=5)",
        "notes": "TF-IDF is not a word embedding; LR is reserved for the baseline",
    }
    return pd.concat([df, pd.DataFrame([ref])], ignore_index=True)


def make_figures(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cand = df[df["eligible_for_n5"]].sort_values("test_accuracy", ascending=True)
    if cand.empty:
        return

    # fig15: grouped CV vs TEST bars, with the TF-IDF reference line
    fig, ax = plt.subplots(figsize=(12, 6.5))
    y = np.arange(len(cand))
    h = 0.38
    ax.barh(y + h / 2, cand["cv_accuracy"], h, label="5-fold CV (train)",
            color="#8C8C8C", edgecolor="black", linewidth=0.5)
    ax.barh(y - h / 2, cand["test_accuracy"], h, label="Test (scored once)",
            color="#1F77B4", edgecolor="black", linewidth=0.5)
    ax.errorbar(cand["cv_accuracy"], y + h / 2, xerr=cand["cv_sd"], fmt="none",
                ecolor="black", capsize=3, linewidth=1)
    for yb, v in zip(y - h / 2, cand["test_accuracy"]):
        ax.text(v + 0.004, yb, f"{v*100:.1f}%", va="center", fontsize=9, fontweight="bold")
    for yb, v in zip(y + h / 2, cand["cv_accuracy"]):
        ax.text(v + 0.004, yb, f"{v*100:.1f}%", va="center", fontsize=8, color="#555")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{e}\n+ {m}" for e, m in zip(cand["embedding"], cand["analytics_model"])],
                       fontsize=9)
    ax.axvline(0.3023, color="red", ls="--", lw=1.2,
               label="Majority baseline (30.2%)")
    tf = df[~df["eligible_for_n5"]]
    if not tf.empty:
        ax.axvline(float(tf["test_accuracy"].iloc[0]), color="green", ls="-.", lw=1.4,
                   label=f"TF-IDF + LR reference ({float(tf['test_accuracy'].iloc[0])*100:.1f}%)")
    ax.set_xlabel("Accuracy")
    ax.set_title("N = 5 embedding x analytics-model matrix\n"
                 "(CV on train only; test scored exactly once per model)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0.25, max(0.66, cand["test_accuracy"].max() + 0.06))
    fig.tight_layout()
    fig.savefig(FIG / "fig15_embedding_matrix.png", dpi=170)
    plt.close(fig)

    # fig16: CV-vs-test scatter, exposes the rank reversal
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot([0.3, 0.62], [0.3, 0.62], "k--", lw=1, label="CV = test")
    ax.scatter(cand["cv_accuracy"], cand["test_accuracy"], s=150,
               c=[PALETTE[i % len(PALETTE)] for i in range(len(cand))],
               edgecolor="black", zorder=3)
    for _, r in cand.iterrows():
        ax.annotate(r["embedding"].split(" (")[0],
                    (r["cv_accuracy"], r["test_accuracy"]),
                    textcoords="offset points", xytext=(7, 6), fontsize=8)
    if not tf.empty:
        ax.axhline(float(tf["test_accuracy"].iloc[0]), color="green", ls="-.", lw=1.3,
                   label="TF-IDF + LR test")
    ax.axhline(0.3023, color="red", ls=":", lw=1.1, label="Majority baseline")
    ax.set_xlabel("CV accuracy (train only)")
    ax.set_ylabel("Test accuracy (single evaluation)")
    ax.set_title("Embedding families: CV vs held-out test", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig16_embedding_families.png", dpi=170)
    plt.close(fig)


def main() -> None:
    print("=" * 88)
    print("Aggregating the N = 5 embedding x analytics-model matrix")
    print("=" * 88)

    df = load_rows()
    df = add_tfidf_reference(df)
    df = df[COLS]

    n_elig = int(df["eligible_for_n5"].sum())
    print(f"\n  eligible candidates: {n_elig} (must be 5)")
    assert n_elig == 5, f"expected exactly 5 eligible rows, got {n_elig}"

    bad = df[(df["test_accuracy"] < 0.30) | (df["test_accuracy"] > 0.65)]
    if not bad.empty:
        print("  WARNING: implausible test accuracy:")
        print(bad[["embedding", "test_accuracy"]].to_string(index=False))

    dfs = df.sort_values(["eligible_for_n5", "test_accuracy"], ascending=[False, False])
    out = TAB / "embedding_model_matrix.csv"
    dfs.to_csv(out, index=False)
    print(f"  saved -> {out.relative_to(TAB.parents[1])}")

    make_figures(df)
    print(f"  saved -> reports/figures/fig15_embedding_matrix.png")
    print(f"  saved -> reports/figures/fig16_embedding_families.png")

    summary = {
        "n_eligible": n_elig,
        "protocol": "5-fold StratifiedKFold on train only (seed 42); test scored once",
        "majority_baseline_test_accuracy": 0.302267,
        "best_candidate": dfs[dfs["eligible_for_n5"]].iloc[0]["embedding"],
        "best_candidate_test_accuracy": float(dfs[dfs["eligible_for_n5"]].iloc[0]["test_accuracy"]),
        "rows": json.loads(dfs.to_json(orient="records")),
    }
    with open(PROC / "embedding_matrix_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  saved -> data/processed/embedding_matrix_summary.json")

    # ---- paste-ready markdown ---------------------------------------------
    print("\n" + "=" * 88)
    print("PASTE-READY TABLE")
    print("=" * 88)
    print("\n| # | Embedding | Analytics model | CV acc | Test acc | Test macro-F1 | kappa |")
    print("|---|---|---|---|---|---|---|")
    for _, r in dfs[dfs["eligible_for_n5"]].iterrows():
        print(f"| {int(r['id'])} | {r['embedding']} | {r['analytics_model']} | "
              f"{r['cv_accuracy']*100:.1f}% | **{r['test_accuracy']*100:.1f}%** | "
              f"{r['test_macro_f1']:.4f} | {r['kappa']:.3f} |")
    tf = df[~df["eligible_for_n5"]]
    if not tf.empty:
        r = tf.iloc[0]
        print(f"| - | {r['embedding']} | {r['analytics_model']} | "
              f"{r['cv_accuracy']*100:.1f}% | {r['test_accuracy']*100:.1f}% | "
              f"{r['test_macro_f1']:.4f} | {r['kappa']:.3f} |")
        print("\n(reference row: TF-IDF is not a word embedding, so it does not count "
              "toward N=5)")


if __name__ == "__main__":
    main()
