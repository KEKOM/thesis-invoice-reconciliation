"""
Retrieval-mode invoice matching.

For each left-side record (incoming invoice) the pipeline:
  1. Constructs a per-query candidate pool: the one correct match plus all
     non-matching right-side records from the test split (pool_size controls
     the number of distractors; None = all).
  2. Optionally applies blocking to prune obvious non-matches first.
  3. Scores every candidate with the already-trained calibrated classifier.
  4. Applies the three-way threshold decision and returns ranked results.

This is the end-to-end demonstration: one invoice in, ranked matches out,
decision applied — without any retraining.

Pool construction guarantee
---------------------------
Every query processed by retrieve_all() has EXACTLY ONE correct match in its
pool.  Queries with zero or more-than-one labelled matches in the test split
are skipped (rare on ER-Magellan; reported in the summary).  This makes the
ranking metrics (MRR, top-k accuracy, mean_confidence_correct) unambiguous.

Entry points
------------
retrieve()               — score one invoice against a pool, return RetrievalResult
retrieve_all()           — run over every eligible left record in the test split
save_retrieval_metrics() — write RetrievalMetrics to JSON
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.block import block_pairs
from src.config import PipelineConfig
from src.features import build_features
from src.model import Thresholds


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Outcome for a single invoice query."""
    decision:     str          # "auto-approve" | "flag-review" | "auto-reject"
    top_score:    float
    candidates:   list[dict]   # top-N: {"score": float, right_* fields …}
    n_candidates: int          # candidates scored after blocking
    is_correct:   Optional[bool] = None   # top-ranked candidate is the correct match
    correct_match_rank:  Optional[int]   = None  # 1-indexed among all n_candidates
    correct_match_score: Optional[float] = None  # model score given to correct match


@dataclass
class RetrievalMetrics:
    """Aggregate retrieval metrics over the full test set."""
    n_invoices:          int
    match_accuracy:      float   # correct approvals / total approvals
    review_rate:         float   # invoices flagged / total invoices
    false_approval_rate: float   # wrong approvals / total approvals
    rejection_rate:      float   # invoices rejected / total invoices
    n_approved:          int
    n_review:            int
    n_rejected:          int
    n_correct_approved:  int
    n_false_approved:    int
    no_ground_truth:     int     = 0       # queries skipped (≠1 correct match)
    # Ranking metrics
    top_3_accuracy:          float = float("nan")  # P(correct match in top 3)
    top_5_accuracy:          float = float("nan")  # P(correct match in top 5)
    mrr:                     float = float("nan")  # mean reciprocal rank
    mean_confidence_correct: float = float("nan")  # avg score given to correct match
    mean_confidence_false:   float = float("nan")  # avg top score on false approvals


# ── Single-invoice retrieval ──────────────────────────────────────────────────

def retrieve(
    invoice: pd.Series,
    pool: pd.DataFrame,
    model,
    thresholds: Thresholds,
    cfg: PipelineConfig,
    feature_names: list[str],
    top_n: int = 5,
    use_blocking: bool = True,
    correct_right_key: Optional[tuple] = None,
) -> RetrievalResult:
    """
    Score one invoice against the candidate pool and return ranked results.

    Parameters
    ----------
    invoice : pd.Series
        A single left-side record with left_* prefixed columns.
    pool : pd.DataFrame
        Right-side candidate records with right_* prefixed columns.
    model : fitted CalibratedClassifierCV
        The already-trained pipeline model — NOT retrained here.
    thresholds : Thresholds
        high / low values chosen on the validation split.
    cfg : PipelineConfig
        Used for feature engineering and blocking configuration.
    feature_names : list[str]
        Column order from the training feature matrix — ensures alignment.
    top_n : int
        Maximum number of ranked candidates to return.
    use_blocking : bool
        Apply blocking before scoring.  Falls back to full pool if nothing passes.
    correct_right_key : tuple, optional
        Tuple of right-side column values identifying the correct match.
        When provided, correct_match_rank and correct_match_score are computed
        over ALL scored candidates (not just top_n).

    Returns
    -------
    RetrievalResult
    """
    invoice_df = invoice.to_frame().T.reset_index(drop=True)

    pairs: pd.DataFrame
    if use_blocking:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pairs = block_pairs(invoice_df, pool, cfg)
    else:
        pairs = pool.copy().reset_index(drop=True)
        for col, val in invoice.items():
            pairs[col] = val

    if pairs.empty:
        return RetrievalResult(
            decision="auto-reject", top_score=0.0, candidates=[], n_candidates=0,
        )

    # Feature engineering + scoring
    X, _ = build_features(pairs, cfg)
    X = X.reindex(columns=feature_names, fill_value=0.0)
    proba = model.predict_proba(X)[:, 1]

    # Rank by score descending
    order        = np.argsort(proba)[::-1]
    pairs_ranked = pairs.iloc[order].reset_index(drop=True)
    proba_ranked = proba[order]
    top_score    = float(proba_ranked[0])

    # Three-way decision based on the top candidate only
    if top_score >= thresholds.high:
        decision = "auto-approve"
    elif top_score > thresholds.low:
        decision = "flag-review"
    else:
        decision = "auto-reject"

    # Locate the correct match in the full ranked list (for MRR / top-k)
    right_cols_in_pairs = [c for c in pairs_ranked.columns if c.startswith("right_")]
    correct_match_rank:  Optional[int]   = None
    correct_match_score: Optional[float] = None

    if correct_right_key is not None:
        for rank_idx in range(len(pairs_ranked)):
            rk = tuple(None if (v != v) else v
                       for v in (pairs_ranked.iloc[rank_idx][c]
                                 for c in right_cols_in_pairs))
            if rk == correct_right_key:
                correct_match_rank  = rank_idx + 1
                correct_match_score = round(float(proba_ranked[rank_idx]), 4)
                break

    # Assemble top-N candidates
    right_cols = [c for c in pairs.columns if c.startswith("right_")]
    candidates = []
    for i in range(min(top_n, len(pairs_ranked))):
        entry: dict = {col: pairs_ranked.iloc[i][col] for col in right_cols}
        entry["score"] = round(float(proba_ranked[i]), 4)
        candidates.append(entry)

    return RetrievalResult(
        decision=decision,
        top_score=round(top_score, 4),
        candidates=candidates,
        n_candidates=len(pairs),
        correct_match_rank=correct_match_rank,
        correct_match_score=correct_match_score,
    )


# ── Batch retrieval over the full test split ──────────────────────────────────

def retrieve_all(
    test_df: pd.DataFrame,
    model,
    thresholds: Thresholds,
    cfg: PipelineConfig,
    feature_names: list[str],
    top_n: int = 5,
    use_blocking: bool = True,
    verbose: bool = False,
) -> tuple[list[RetrievalResult], RetrievalMetrics]:
    """
    Run retrieval over every eligible left-side record in the test split.

    Pool construction (per query)
    -----------------------------
    Each query gets a pool containing:
      - Its single correct right-side match (guaranteed by the skip logic).
      - All non-matching right-side records from the test split
        (or a random subsample of cfg.retrieval.pool_size - 1 distractors).

    Queries with zero or more-than-one labelled matches are skipped; the count
    is reported in RetrievalMetrics.no_ground_truth.

    Parameters
    ----------
    test_df : pd.DataFrame
        Held-out test split with left_*, right_*, and label columns.
    model, thresholds, cfg, feature_names
        From the standard pipeline — no retraining.
    top_n : int
        Number of candidates stored per result.
    use_blocking : bool
        Apply blocking before scoring each invoice.
    verbose : bool
        Print per-invoice progress every 50 invoices.

    Returns
    -------
    (results, metrics)
        results : list[RetrievalResult], one per evaluated invoice
        metrics : RetrievalMetrics aggregated over all results
    """
    left_cols  = [c for c in test_df.columns if c.startswith("left_")]
    right_cols = [c for c in test_df.columns if c.startswith("right_")]

    def _row_key(row: pd.Series, cols: list[str]) -> tuple:
        # Normalise NaN → None so that tuples used as dict keys hash and
        # compare consistently across separate iterrows() passes.
        # (float('nan') != float('nan'), so two nan objects from different
        #  iterations would never match in a dict lookup.)
        return tuple(None if (v != v) else v for v in (row[c] for c in cols))

    # Ground truth: lk → set of rks with label=1
    gt: dict[tuple, set[tuple]] = {}
    for _, row in test_df.iterrows():
        if row["label"] == 1:
            lk = _row_key(row, left_cols)
            rk = _row_key(row, right_cols)
            gt.setdefault(lk, set()).add(rk)

    # Full right-side record index: rk → row dict
    right_index: dict[tuple, dict] = {}
    for _, row in test_df[right_cols].drop_duplicates().iterrows():
        rk = _row_key(row, right_cols)
        right_index[rk] = row.to_dict()

    all_right_keys: list[tuple] = list(right_index.keys())
    pool_size: Optional[int] = cfg.retrieval.pool_size
    rng = np.random.default_rng(cfg.seed)

    invoices = test_df[left_cols].drop_duplicates().reset_index(drop=True)
    n_total   = len(invoices)

    results: list[RetrievalResult] = []
    n_skipped = 0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="All-NaN slice encountered")

        for idx in range(n_total):
            invoice  = invoices.iloc[idx]
            lk       = _row_key(invoice, left_cols)
            true_rks = gt.get(lk, set())

            # Enforce exactly one correct match per query
            if len(true_rks) != 1:
                n_skipped += 1
                continue

            correct_rk  = next(iter(true_rks))
            distractors = [rk for rk in all_right_keys if rk not in true_rks]

            # Optionally subsample distractors
            if pool_size is not None and pool_size > 1:
                n_dist = pool_size - 1
                if len(distractors) > n_dist:
                    chosen = rng.choice(len(distractors), n_dist, replace=False)
                    distractors = [distractors[int(i)] for i in sorted(chosen)]

            # Build per-query pool DataFrame
            pool_keys = [correct_rk] + distractors
            pool_rows = [right_index[rk] for rk in pool_keys]
            pool_df   = pd.DataFrame(pool_rows).reset_index(drop=True)

            result = retrieve(
                invoice, pool_df, model, thresholds, cfg, feature_names,
                top_n=top_n, use_blocking=use_blocking,
                correct_right_key=correct_rk,
            )

            # is_correct = top-ranked candidate is the correct match
            if result.candidates:
                top_rk = _row_key(
                    pd.Series({c: result.candidates[0].get(c) for c in right_cols}),
                    right_cols,
                )
                result.is_correct = top_rk in true_rks
            else:
                result.is_correct = False

            results.append(result)

            if verbose and (idx + 1) % 50 == 0:
                print(f"  {idx + 1}/{n_total} invoices …", flush=True)

    if verbose and n_skipped:
        print(f"  Skipped {n_skipped} queries (≠1 correct match in test split).")

    return results, _compute_metrics(results, n_skipped)


# ── Metric aggregation ────────────────────────────────────────────────────────

def _compute_metrics(
    results: list[RetrievalResult],
    n_skipped: int = 0,
) -> RetrievalMetrics:
    n = len(results)

    approved = [r for r in results if r.decision == "auto-approve"]
    review   = [r for r in results if r.decision == "flag-review"]
    rejected = [r for r in results if r.decision == "auto-reject"]

    n_approved = len(approved)
    n_review   = len(review)
    n_rejected = len(rejected)

    # Approval accuracy (top-ranked candidate = correct match)
    correct = [r for r in approved if r.is_correct is True]
    wrong   = [r for r in approved if r.is_correct is False]

    match_accuracy      = round(len(correct) / n_approved, 4) if n_approved else float("nan")
    false_approval_rate = round(len(wrong)   / n_approved, 4) if n_approved else 0.0

    # ── Ranking metrics (use correct_match_rank which covers all n_candidates) ──
    ranked = [r for r in results if r.correct_match_rank is not None]

    if ranked:
        mrr = float(np.mean([1.0 / r.correct_match_rank for r in ranked]))
        top3 = sum(1 for r in ranked if r.correct_match_rank <= 3)
        top5 = sum(1 for r in ranked if r.correct_match_rank <= 5)
        top_3_accuracy = round(top3 / len(ranked), 4)
        top_5_accuracy = round(top5 / len(ranked), 4)
    else:
        mrr = top_3_accuracy = top_5_accuracy = float("nan")

    # ── Confidence metrics ────────────────────────────────────────────────────
    correct_scores = [r.correct_match_score for r in results
                      if r.correct_match_score is not None]
    false_scores   = [r.top_score for r in approved if r.is_correct is False]

    mean_confidence_correct = (round(float(np.mean(correct_scores)), 4)
                               if correct_scores else float("nan"))
    mean_confidence_false   = (round(float(np.mean(false_scores)), 4)
                               if false_scores else float("nan"))

    return RetrievalMetrics(
        n_invoices          = n,
        match_accuracy      = match_accuracy,
        review_rate         = round(n_review   / n, 4) if n else 0.0,
        false_approval_rate = false_approval_rate,
        rejection_rate      = round(n_rejected / n, 4) if n else 0.0,
        n_approved          = n_approved,
        n_review            = n_review,
        n_rejected          = n_rejected,
        n_correct_approved  = len(correct),
        n_false_approved    = len(wrong),
        no_ground_truth     = n_skipped,
        top_3_accuracy      = top_3_accuracy,
        top_5_accuracy      = top_5_accuracy,
        mrr                 = round(mrr, 4) if not np.isnan(mrr) else float("nan"),
        mean_confidence_correct = mean_confidence_correct,
        mean_confidence_false   = mean_confidence_false,
    )


# ── Serialisation ─────────────────────────────────────────────────────────────

def save_retrieval_metrics(metrics: RetrievalMetrics, path: str | Path) -> None:
    """Write retrieval metrics to JSON alongside the standard metrics files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _safe(v):
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    d = {
        "n_invoices":             metrics.n_invoices,
        "match_accuracy":         _safe(metrics.match_accuracy),
        "review_rate":            _safe(metrics.review_rate),
        "false_approval_rate":    _safe(metrics.false_approval_rate),
        "rejection_rate":         _safe(metrics.rejection_rate),
        "n_approved":             metrics.n_approved,
        "n_review":               metrics.n_review,
        "n_rejected":             metrics.n_rejected,
        "n_correct_approved":     metrics.n_correct_approved,
        "n_false_approved":       metrics.n_false_approved,
        "no_ground_truth":        metrics.no_ground_truth,
        "top_3_accuracy":         _safe(metrics.top_3_accuracy),
        "top_5_accuracy":         _safe(metrics.top_5_accuracy),
        "mrr":                    _safe(metrics.mrr),
        "mean_confidence_correct": _safe(metrics.mean_confidence_correct),
        "mean_confidence_false":   _safe(metrics.mean_confidence_false),
    }
    with open(path, "w") as fh:
        json.dump(d, fh, indent=2)
