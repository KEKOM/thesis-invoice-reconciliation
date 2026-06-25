"""
Evaluation: per-dataset metrics and cross-dataset comparison.

Design notes
------------
* compute_metrics() is the single point of contact with test-set labels.
  It must only be called AFTER thresholds have been chosen on the validation
  split (see src.thresholds.find_thresholds).

* compare_datasets() produces the cross-domain comparison table required for
  the thesis.  It deliberately includes review_queue_count and
  review_queue_pct alongside the standard metrics.

  Rationale: identical threshold targets produce different automation rates
  across domains.  Reporting review-queue size makes this visible and supports
  an honest cross-domain robustness claim — the same threshold procedure yields
  different automation behaviour on Walmart-Amazon vs DBLP-ACM, rather than
  assuming the two datasets are directly comparable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model import Thresholds, decide


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_metrics(
    model,
    X_test:     pd.DataFrame,
    y_test:     pd.Series,
    thresholds: Thresholds,
) -> dict[str, Any]:
    """
    Evaluate a fitted model on the held-out test split.

    This function must only be called once per pipeline run, on test data.
    Do not pass validation data here.

    Parameters
    ----------
    model : fitted CalibratedClassifierCV (or any predict_proba estimator)
    X_test : feature matrix (test split)
    y_test : labels (test split)
    thresholds : Thresholds
        Chosen on the validation split by src.thresholds.find_thresholds.

    Returns
    -------
    dict containing scalar metrics, confusion matrix, per-action counts,
    and raw arrays (proba, actions, pred) for building result DataFrames.
    """
    proba   = model.predict_proba(X_test)[:, 1]
    pred    = (proba >= 0.5).astype(int)
    actions = decide(proba, thresholds)

    y_arr = np.asarray(y_test)
    tn, fp, fn, tp = confusion_matrix(y_arr, pred).ravel()
    n = len(y_arr)

    n_approve = int((actions == "auto-approve").sum())
    n_reject  = int((actions == "auto-reject").sum())
    n_flag    = int((actions == "flag-review").sum())

    # Missed matches: true positives that were auto-rejected (safety metric)
    ar_mask       = actions == "auto-reject"
    missed_matches = int(y_arr[ar_mask].sum())
    missed_pct     = round(100 * missed_matches / max(1, int(y_arr.sum())), 1)

    return {
        # ── Dataset shape ─────────────────────────────────────────────────────
        "n_pairs":       n,
        "n_matches":     int(y_arr.sum()),
        "n_non_matches": int((y_arr == 0).sum()),

        # ── Core classification metrics ───────────────────────────────────────
        # Lead with precision/recall/F1 — raw accuracy is misleading on
        # class-imbalanced datasets (Walmart-Amazon ~10 % match rate).
        "precision":  round(float(precision_score(y_arr, pred, zero_division=0)), 4),
        "recall":     round(float(recall_score(y_arr,    pred, zero_division=0)), 4),
        "f1":         round(float(f1_score(y_arr,        pred, zero_division=0)), 4),
        "pr_auc":     round(float(average_precision_score(y_arr, proba)),         4),
        "roc_auc":    round(float(roc_auc_score(y_arr, proba)),                   4),
        "accuracy":   round(float((tp + tn) / n),                                 4),

        # ── Confusion matrix ──────────────────────────────────────────────────
        "confusion":  {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},

        # ── Routing counts ────────────────────────────────────────────────────
        "auto_approve":       n_approve,
        "flag_review":        n_flag,
        "auto_reject":        n_reject,
        "pct_automated":      round(100 * (n_approve + n_reject) / n, 1),
        "review_queue_count": n_flag,
        "review_queue_pct":   round(100 * n_flag / n, 1),

        # ── Safety: missed matches in the auto-reject zone ────────────────────
        "missed_matches":     missed_matches,
        "missed_matches_pct": missed_pct,

        # ── Raw arrays (for building result DataFrames) ───────────────────────
        "proba":    proba,
        "actions":  actions,
        "pred":     pred,
    }


# ── Cross-dataset comparison ──────────────────────────────────────────────────

def compare_datasets(
    metrics_by_dataset: dict[str, dict],
) -> pd.DataFrame:
    """
    Build a side-by-side comparison table across datasets.

    Parameters
    ----------
    metrics_by_dataset : dict
        Keys are dataset names (e.g. "walmart_amazon", "dblp_acm").
        Values are dicts returned by compute_metrics().

    Returns
    -------
    pd.DataFrame with one row per dataset and columns for the key metrics
    plus review-queue size/proportion.

    Why review-queue size is included
    ----------------------------------
    Identical threshold targets (e.g. 90 % approve precision, 90 % reject
    recall) produce different automation rates across domains.  Reporting
    review_queue_count and review_queue_pct makes this visible and supports
    an honest cross-domain robustness claim: the same procedure yields
    different automation behaviour on Walmart-Amazon vs DBLP-ACM.
    """
    rows = []
    for name, m in metrics_by_dataset.items():
        rows.append({
            "dataset":            name,
            "n_test_pairs":       m["n_pairs"],
            "match_rate_pct":     round(100 * m["n_matches"] / max(1, m["n_pairs"]), 1),
            "precision":          m["precision"],
            "recall":             m["recall"],
            "f1":                 m["f1"],
            "pr_auc":             m["pr_auc"],
            "roc_auc":            m["roc_auc"],
            "pct_automated":      m["pct_automated"],
            "review_queue_count": m["review_queue_count"],
            "review_queue_pct":   m["review_queue_pct"],
            "missed_matches":     m["missed_matches"],
            "missed_matches_pct": m["missed_matches_pct"],
        })
    return pd.DataFrame(rows).set_index("dataset")


# ── RF vs XGBoost model comparison ───────────────────────────────────────────

def compare_models(
    rf_metrics: dict,
    xgb_metrics: dict,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Side-by-side RF vs XGBoost comparison table for one dataset.

    Parameters
    ----------
    rf_metrics, xgb_metrics : dicts returned by compute_metrics() or load_metrics().
        Both must have been evaluated on the same test split so metrics are
        directly comparable (same n_pairs, same label distribution).
    dataset_name : str
        Included in the DataFrame index label for identification in thesis tables.

    Returns
    -------
    pd.DataFrame with one row per model and columns for every key metric plus
    review-queue size.  RF is listed first as the primary model.

    Why review-queue size is included
    ----------------------------------
    A model with higher F1 but a larger review queue may not be preferable in
    a financial audit context where analyst time is the binding constraint.
    Reporting both makes the trade-off explicit.
    """
    rows = []
    for label, m in [
        ("Random Forest (primary)", rf_metrics),
        ("XGBoost (benchmark)",     xgb_metrics),
    ]:
        rows.append({
            "model":              label,
            "precision":          m["precision"],
            "recall":             m["recall"],
            "f1":                 m["f1"],
            "pr_auc":             m["pr_auc"],
            "roc_auc":            m["roc_auc"],
            "pct_automated":      m["pct_automated"],
            "review_queue_count": m["review_queue_count"],
            "review_queue_pct":   m["review_queue_pct"],
        })
    df = pd.DataFrame(rows).set_index("model")
    df.index.name = f"model [{dataset_name}]"
    return df


# ── Serialise / deserialise ───────────────────────────────────────────────────

def _serialisable(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_metrics(metrics: dict, path: str | Path) -> None:
    """Write scalar metrics (not raw arrays) to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Strip non-serialisable raw arrays before writing
    slim = {k: v for k, v in metrics.items() if k not in {"proba", "actions", "pred"}}
    with open(path, "w") as fh:
        json.dump(_serialisable(slim), fh, indent=2)


def load_metrics(path: str | Path) -> dict:
    """Load previously saved metrics from JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path) as fh:
        return json.load(fh)
