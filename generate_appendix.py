"""
Generate thesis appendix tables and figures.

Reads saved artefacts from results/ and writes publication-ready outputs to
appendix/.  Run from the project root after all three datasets have been
evaluated in both classify and retrieve modes.

Usage
-----
    python generate_appendix.py

Outputs
-------
appendix/
├── tables/
│   ├── A1_classification_summary.csv / .tex
│   ├── A2_classification_detail_{dataset}.csv / .tex
│   ├── A3_rf_vs_xgboost_{dataset}.csv / .tex
│   ├── B1_retrieval_summary.csv / .tex
│   └── B2_retrieval_detail.csv / .tex
└── figures/
    ├── C1_feature_importance_walmart_amazon.pdf / .png
    ├── C2_feature_importance_dblp_googlescholar.pdf / .png
    └── C3_feature_importance_dblp_acm.pdf / .png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

DATASETS = [
    ("walmart_amazon",     "Walmart-Amazon"),
    ("dblp_googlescholar", "DBLP-GoogleScholar"),
    ("dblp_acm",           "DBLP-ACM"),
]

RESULTS = Path("results")
OUT     = Path("appendix")
TABLES  = OUT / "tables"
FIGS    = OUT / "figures"

THESIS_RC = {
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"  ⚠  Missing: {path}")
        return None
    with open(path) as fh:
        return json.load(fh)


def _save_table(df: pd.DataFrame, stem: str, caption: str = "") -> None:
    csv_path = TABLES / f"{stem}.csv"
    tex_path = TABLES / f"{stem}.tex"
    df.to_csv(csv_path)
    print(f"  Saved {csv_path.name}")
    try:
        tex = df.to_latex(
            float_format="%.4f",
            caption=caption,
            label=f"tab:{stem}",
            na_rep="—",
        )
        tex_path.write_text(tex)
        print(f"  Saved {tex_path.name}")
    except Exception as exc:
        print(f"  (LaTeX export skipped: {exc})")


def _pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.1%}"


def _fmt4(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.4f}"


# ── A. Classification Results ─────────────────────────────────────────────────

def make_classification_summary() -> None:
    """Table A1 — one row per dataset, key classification metrics."""
    print("\n── A1: Classification summary ──")
    rows = []
    for key, label in DATASETS:
        m = _load_json(RESULTS / key / "metrics.json")
        if m is None:
            continue
        t = m.get("thresholds", {})
        rows.append({
            "Dataset":           label,
            "Precision":         round(m["precision"], 4),
            "Recall":            round(m["recall"],    4),
            "F1":                round(m["f1"],        4),
            "PR-AUC":            round(m["pr_auc"],    4),
            "ROC-AUC":           round(m["roc_auc"],   4),
            "% Automated":       m["pct_automated"],
            "Review queue":      m["review_queue_count"],
            "Missed matches":    m["missed_matches"],
            "Approve threshold": round(t.get("high", float("nan")), 4),
            "Reject threshold":  round(t.get("low",  float("nan")), 4),
        })
    df = pd.DataFrame(rows).set_index("Dataset")
    _save_table(df, "A1_classification_summary",
                "Classification metrics across all three benchmark datasets.")


def make_classification_detail() -> None:
    """Tables A2a-A2c — per-dataset detailed metrics including confusion matrix."""
    print("\n── A2: Classification detail (per dataset) ──")
    for i, (key, label) in enumerate(DATASETS, start=1):
        m = _load_json(RESULTS / key / "metrics.json")
        if m is None:
            continue
        c = m.get("confusion", {})
        t = m.get("thresholds", {})
        rows = {
            "Metric": [
                "Test pairs (n)",
                "Positive pairs (matches)",
                "Match rate",
                "Precision",
                "Recall",
                "F1",
                "PR-AUC",
                "ROC-AUC",
                "Accuracy",
                "Auto-approved",
                "Flagged for review",
                "Auto-rejected",
                "% Automated",
                "Missed matches (false rejects)",
                "Missed matches %",
                "True positives",
                "False positives",
                "True negatives",
                "False negatives",
                "Approve threshold (high)",
                "Reject threshold (low)",
                "Threshold case",
                "Model type",
                "n_features",
            ],
            "Value": [
                m.get("n_pairs"),
                m.get("n_matches"),
                f"{m.get('n_matches', 0) / m.get('n_pairs', 1):.1%}",
                round(m["precision"], 4),
                round(m["recall"],    4),
                round(m["f1"],        4),
                round(m["pr_auc"],    4),
                round(m["roc_auc"],   4),
                round(m["accuracy"],  4),
                m.get("auto_approve"),
                m.get("flag_review"),
                m.get("auto_reject"),
                f"{m.get('pct_automated')} %",
                m.get("missed_matches"),
                f"{m.get('missed_matches_pct')} %",
                c.get("tp"),
                c.get("fp"),
                c.get("tn"),
                c.get("fn"),
                round(t.get("high", float("nan")), 4),
                round(t.get("low",  float("nan")), 4),
                t.get("case", "—"),
                m.get("model_type"),
                m.get("n_features"),
            ],
        }
        df = pd.DataFrame(rows).set_index("Metric")
        _save_table(df, f"A2{chr(96+i)}_classification_detail_{key}",
                    f"Detailed classification metrics — {label}.")


def make_rf_vs_xgboost() -> None:
    """Tables A3a-A3c — RF vs XGBoost benchmark comparison per dataset."""
    print("\n── A3: RF vs XGBoost benchmark ──")
    for i, (key, label) in enumerate(DATASETS, start=1):
        rf_path  = RESULTS / key / "metrics_rf.json"
        xgb_path = RESULTS / key / "metrics_xgboost.json"
        if not rf_path.exists() or not xgb_path.exists():
            print(f"  ⚠  Benchmark results missing for {label} — run --benchmark first")
            continue
        rf  = _load_json(rf_path)
        xgb = _load_json(xgb_path)
        rows = []
        for model_label, m in [("Random Forest", rf), ("XGBoost", xgb)]:
            t = m.get("thresholds", {})
            rows.append({
                "Model":             model_label,
                "Precision":         round(m["precision"], 4),
                "Recall":            round(m["recall"],    4),
                "F1":                round(m["f1"],        4),
                "PR-AUC":            round(m["pr_auc"],    4),
                "ROC-AUC":           round(m["roc_auc"],   4),
                "% Automated":       m["pct_automated"],
                "Review queue":      m["review_queue_count"],
                "Approve threshold": round(t.get("high", float("nan")), 4),
            })
        df = pd.DataFrame(rows).set_index("Model")
        _save_table(df, f"A3{chr(96+i)}_rf_vs_xgboost_{key}",
                    f"RF vs XGBoost benchmark — {label}.")


# ── B. Retrieval Results ──────────────────────────────────────────────────────

def make_retrieval_summary() -> None:
    """Table B1 — cross-dataset retrieval summary (key ranking metrics)."""
    print("\n── B1: Retrieval summary ──")
    rows = []
    for key, label in DATASETS:
        m = _load_json(RESULTS / key / "retrieval_metrics.json")
        if m is None:
            continue
        rows.append({
            "Dataset":                 label,
            "Invoices evaluated":      m.get("n_invoices"),
            "Match accuracy":          _pct(m.get("match_accuracy")),
            "False approval rate":     _pct(m.get("false_approval_rate")),
            "Review rate":             _pct(m.get("review_rate")),
            "Rejection rate":          _pct(m.get("rejection_rate")),
            "MRR":                     _fmt4(m.get("mrr")),
            "Top-3 accuracy":          _pct(m.get("top_3_accuracy")),
            "Top-5 accuracy":          _pct(m.get("top_5_accuracy")),
            "Mean conf. (correct)":    _fmt4(m.get("mean_confidence_correct")),
            "Mean conf. (false appr.)":_fmt4(m.get("mean_confidence_false")),
        })
    df = pd.DataFrame(rows).set_index("Dataset")
    _save_table(df, "B1_retrieval_summary",
                "Retrieval evaluation metrics across all three benchmark datasets.")


def make_retrieval_detail() -> None:
    """Table B2 — raw counts per dataset."""
    print("\n── B2: Retrieval detail (counts) ──")
    rows = []
    for key, label in DATASETS:
        m = _load_json(RESULTS / key / "retrieval_metrics.json")
        if m is None:
            continue
        rows.append({
            "Dataset":            label,
            "n_invoices":         m.get("n_invoices"),
            "n_approved":         m.get("n_approved"),
            "n_correct_approved": m.get("n_correct_approved"),
            "n_false_approved":   m.get("n_false_approved"),
            "n_review":           m.get("n_review"),
            "n_rejected":         m.get("n_rejected"),
            "Skipped (≠1 match)": m.get("no_ground_truth"),
        })
    df = pd.DataFrame(rows).set_index("Dataset")
    _save_table(df, "B2_retrieval_detail",
                "Retrieval decision counts across all three benchmark datasets.")


# ── C. Feature Importance Plots ───────────────────────────────────────────────

def make_feature_importance_plots() -> None:
    """Figures C1-C3 — top-15 feature importance horizontal bar charts."""
    print("\n── C: Feature importance plots ──")
    palette = {
        "walmart_amazon":     "#2563EB",
        "dblp_googlescholar": "#7c3aed",
        "dblp_acm":           "#059669",
    }
    for i, (key, label) in enumerate(DATASETS, start=1):
        fi_path = RESULTS / key / "feature_importance.csv"
        if not fi_path.exists():
            print(f"  ⚠  Feature importance missing for {label}")
            continue
        fi = pd.read_csv(fi_path)
        top = fi.nlargest(15, "importance").sort_values("importance")

        with plt.rc_context(THESIS_RC):
            fig, ax = plt.subplots(figsize=(7, 5))
            bars = ax.barh(
                top["feature"], top["importance"],
                color=palette.get(key, "#2563EB"),
                edgecolor="white", linewidth=0.4,
            )
            ax.set_xlabel("Feature importance (mean decrease in impurity)")
            ax.set_title(f"Top-15 feature importances — {label}")
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            ax.spines[["top", "right"]].set_visible(False)
            ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
            ax.set_xlim(0, top["importance"].max() * 1.18)
            fig.tight_layout()

            stem = f"C{i}_feature_importance_{key}"
            for ext in ("pdf", "png"):
                path = FIGS / f"{stem}.{ext}"
                fig.savefig(path)
                print(f"  Saved {path.name}")
            plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    print("Generating thesis appendix …")

    make_classification_summary()
    make_classification_detail()
    make_rf_vs_xgboost()

    make_retrieval_summary()
    make_retrieval_detail()

    make_feature_importance_plots()

    print(f"\nDone.  All outputs written to {OUT}/")
