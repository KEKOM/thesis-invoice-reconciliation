"""
Matching model, calibration, and decision logic.

Stage 2 of the pipeline:
  - a classifier (Random Forest or XGBoost) predicts match probability
  - probabilities are calibrated on a held-out validation slice so that
    p = 0.80 means ≈80 % empirical true-positive rate
  - a confidence threshold turns calibrated probabilities into one of three
    actions, mirroring the "only flag exceptions" requirement:

        auto-approve   : p(match) >= high_threshold
        flag-review    : low_threshold < p(match) < high_threshold
        auto-reject    : p(match) <= low_threshold

Calibration note
----------------
An uncalibrated RF's output of p = 0.80 reflects tree vote ratios, not
empirical match frequency.  After calibration, threshold values transfer
across datasets without re-tuning — p = 0.90 really means ~90 % of approved
pairs are genuine matches on ANY balanced dataset.

Model selection
---------------
Set cfg.model.type = "rf" (default) or "xgboost" in the dataset config.
Both classifiers expose the same fit/predict_proba interface, so everything
downstream (calibration, thresholds, evaluation, dashboard) is identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import PredefinedSplit, train_test_split

from src.config import ModelConfig, CalibrationConfig, PipelineConfig


# ── Decision types ────────────────────────────────────────────────────────────

@dataclass
class Thresholds:
    high: float = 0.80   # p >= this  → auto-approve
    low:  float = 0.20   # p <= this  → auto-reject


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(
    cfg: ModelConfig,
    y: pd.Series | None = None,
) -> Union[RandomForestClassifier, "XGBClassifier"]:  # noqa: F821
    """
    Instantiate the base classifier from config (not yet fitted).

    Parameters
    ----------
    cfg : ModelConfig
        Specifies type ("rf" | "xgboost"), hyperparameters, random_state.
    y : pd.Series, optional
        Training labels — only used for XGBoost's ``scale_pos_weight``
        (negative/positive ratio for class-imbalance handling).
        Ignored for Random Forest (uses class_weight="balanced" instead).
    """
    if cfg.type == "rf":
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            class_weight=cfg.class_weight,
            n_jobs=-1,
            random_state=cfg.random_state,
        )

    if cfg.type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError(
                "XGBoost is not installed. Run: pip install 'xgboost>=2.0'"
            ) from None

        spw = 1.0
        if y is not None:
            n_neg = int((y == 0).sum())
            n_pos = int((y == 1).sum())
            spw   = n_neg / max(n_pos, 1)

        return XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            subsample=cfg.xgb_subsample,
            scale_pos_weight=spw,
            random_state=cfg.random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model type: {cfg.type!r}")


# ── Legacy helper (backward compat, used by root run.py) ─────────────────────

def train_model(X: pd.DataFrame, y: pd.Series, **rf_kwargs) -> RandomForestClassifier:
    """Train a plain (uncalibrated) Random Forest.  Kept for backward compat."""
    params = dict(n_estimators=300, max_depth=None, class_weight="balanced",
                  n_jobs=-1, random_state=42)
    params.update(rf_kwargs)
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model


# ── Calibrated training ───────────────────────────────────────────────────────

def train_calibrated_model(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: PipelineConfig | None = None,
    val_fraction: float = 0.20,
    min_isotonic_samples: int = 1000,
    **rf_kwargs,
) -> tuple[CalibratedClassifierCV, Union[RandomForestClassifier, "XGBClassifier"]]:  # noqa: F821
    """
    Train a classifier and calibrate its probabilities on a held-out split.

    Parameters
    ----------
    X, y : feature matrix and labels (training data only — no validation or
           test rows should be passed here).
    cfg : PipelineConfig, optional
        When provided, model type and calibration settings come from cfg.
        Explicit kwargs are used as fallback when cfg is None.
    val_fraction : float
        Share of rows withheld for calibration fitting (cfg overrides this).
    min_isotonic_samples : int
        Threshold for choosing isotonic vs sigmoid calibration (cfg overrides).
    **rf_kwargs
        Extra keyword arguments forwarded to RandomForestClassifier when
        cfg is None (backward-compat path).

    Returns
    -------
    (calibrated, base_estimator)
        calibrated  — use for all probability estimates and decisions.
        base_estimator — retains feature_importances_ for interpretability.

    Calibration strategy
    --------------------
    cv='prefit' was removed in scikit-learn 1.6.  We replicate its semantics
    via PredefinedSplit: CalibratedClassifierCV clones the base estimator and
    fits the clone on fold=-1 rows; fold=0 rows are withheld exclusively for
    fitting the calibration layer.  This ensures the calibration layer never
    sees the same rows the base model was trained on.  The clone fitted by
    calibration is returned as ``base_estimator`` (exposes feature_importances_).
    """
    # Resolve settings
    if cfg is not None:
        _val_frac    = cfg.calibration.val_fraction
        _min_iso     = cfg.calibration.min_isotonic_samples
        _method_cfg  = cfg.calibration.method   # "auto" | "isotonic" | "sigmoid"
    else:
        _val_frac    = val_fraction
        _min_iso     = min_isotonic_samples
        _method_cfg  = "auto"

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=_val_frac, random_state=cfg.seed if cfg else 42, stratify=y
    )

    # Build base estimator WITHOUT fitting.
    # CalibratedClassifierCV with PredefinedSplit always clones the estimator
    # and fits that clone on the fold=-1 rows — pre-fitting base here would
    # just train the model a second time on identical data with no effect on
    # the calibrated output.
    if cfg is not None:
        base = build_model(cfg.model, y=y_tr)
    else:
        params = dict(n_estimators=300, max_depth=None, class_weight="balanced",
                      n_jobs=-1, random_state=42)
        params.update(rf_kwargs)
        base = RandomForestClassifier(**params)

    # Choose calibration method
    if _method_cfg == "auto":
        method = "isotonic" if len(X_tr) >= _min_iso else "sigmoid"
    else:
        method = _method_cfg

    # PredefinedSplit: fold=-1 → train (not used by calibration), fold=0 → calibrate
    test_fold = np.concatenate([
        np.full(len(X_tr),  -1, dtype=int),
        np.zeros(len(X_val), dtype=int),
    ])
    ps = PredefinedSplit(test_fold)
    X_combined = pd.concat([X_tr, X_val], ignore_index=True)
    y_combined = pd.concat([y_tr, y_val], ignore_index=True)

    calibrated = CalibratedClassifierCV(base, cv=ps, method=method)
    calibrated.fit(X_combined, y_combined)
    # Return the clone that calibration actually fitted — more correct than
    # returning the unfitted `base`, and exposes feature_importances_ directly.
    fitted_base = calibrated.calibrated_classifiers_[0].estimator
    return calibrated, fitted_base


# ── Decision and evaluation helpers ──────────────────────────────────────────

def decide(proba: np.ndarray, t: Thresholds) -> np.ndarray:
    """Map calibrated probabilities to action labels."""
    out = np.full(len(proba), "flag-review", dtype=object)
    out[proba >= t.high] = "auto-approve"
    out[proba <= t.low]  = "auto-reject"
    return out


def evaluate(model, X: pd.DataFrame, y: pd.Series, t: Thresholds) -> dict:
    """
    Compute metrics + per-pair predictions for a fitted model.

    Returns a dict containing scalar metrics, confusion matrix, per-action
    counts, and arrays (proba, actions, pred) for downstream use.
    """
    proba   = model.predict_proba(X)[:, 1]
    pred    = (proba >= 0.5).astype(int)
    actions = decide(proba, t)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    n       = len(y)
    n_flag  = int((actions == "flag-review").sum())
    return {
        "n_pairs":       n,
        "precision":     round(precision_score(y, pred, zero_division=0), 4),
        "recall":        round(recall_score(y, pred, zero_division=0), 4),
        "f1":            round(f1_score(y, pred, zero_division=0), 4),
        "roc_auc":       round(roc_auc_score(y, proba), 4),
        "pr_auc":        round(average_precision_score(y, proba), 4),
        "confusion":     {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "auto_approve":  int((actions == "auto-approve").sum()),
        "auto_reject":   int((actions == "auto-reject").sum()),
        "flag_review":   n_flag,
        "pct_automated": round(100 * (n - n_flag) / n, 1),
        "proba":         proba,
        "actions":       actions,
        "pred":          pred,
    }


def feature_importances(model, names: list[str], top: int = 15) -> pd.DataFrame:
    """Return top-N feature importances, unwrapping CalibratedClassifierCV."""
    inner = (
        model.calibrated_classifiers_[0].estimator
        if isinstance(model, CalibratedClassifierCV)
        else model
    )
    imp = pd.DataFrame({"feature": names, "importance": inner.feature_importances_})
    return (
        imp.sort_values("importance", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


# ── Curve / sweep helpers (used by dashboard) ─────────────────────────────────

def roc_curve_data(model, X: pd.DataFrame, y: pd.Series) -> dict:
    proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "auc": round(float(roc_auc_score(y, proba)), 4)}


def pr_curve_data(model, X: pd.DataFrame, y: pd.Series) -> dict:
    proba = model.predict_proba(X)[:, 1]
    prec, rec, _ = precision_recall_curve(y, proba)
    return {"precision": prec.tolist(), "recall": rec.tolist(),
            "auc": round(float(average_precision_score(y, proba)), 4)}


def threshold_sweep(
    model, X: pd.DataFrame, y: pd.Series, n_points: int = 80
) -> pd.DataFrame:
    """Precision, recall, F1 at evenly-spaced thresholds (for sensitivity chart)."""
    proba = model.predict_proba(X)[:, 1]
    rows  = []
    for t in np.linspace(0.01, 0.99, n_points):
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 3),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y, pred,    zero_division=0)), 4),
            "f1":        round(float(f1_score(y, pred,        zero_division=0)), 4),
        })
    return pd.DataFrame(rows)
