"""
Validation-split threshold search.

The three-way routing decision (auto-approve / flag-review / auto-reject)
requires two confidence thresholds: high (approve gate) and low (reject gate).
This module searches for those thresholds on the VALIDATION split only —
test data is never passed to any function here.

Coherence guarantee
-------------------
The returned thresholds always satisfy  high >= low.  When they did not,
the pipeline's three-way routing would be undefined: a pair with probability
between low and high cannot simultaneously be auto-approved and auto-rejected.

Search strategy (precision_target)
-----------------------------------
  high_threshold — lowest t such that precision(approved) >= approve_target
      "Among all pairs the model is confident enough to auto-approve,
       at least approve_target fraction are genuine matches."
      Taking the LOWEST such t maximises the auto-approve zone while
      maintaining the precision guarantee.

  low_threshold  — lowest t such that recall_nonmatch(rejected) >= reject_target
      "Of all true non-matches, at least reject_target fraction fall below t
       and are auto-rejected."
      Taking the LOWEST such t is the most conservative choice: we only
      auto-reject pairs where the model is very confident they are non-matches,
      minimising the risk of accidentally rejecting real matches.

Why not the highest valid reject threshold?
-------------------------------------------
recall_nonmatch(t) is monotonically non-decreasing in t: once it reaches the
target at some t*, all t > t* also satisfy it.  Taking max(valid_low) would
therefore return a value close to best_high (or equal to it), collapsing the
review zone to zero and sending everything directly to auto-approve or
auto-reject with no human oversight.  Taking min(valid_low) creates a
meaningful grey band [low, high] where uncertain pairs go to review.

Coherence guarantee
-------------------
Because best_low is the SMALLEST valid reject threshold and best_high is the
SMALLEST valid approve threshold, and because lower probabilities are easier to
reject (low precision is avoided at low threshold only for approve side), on a
reasonably good model best_high > best_low will typically hold, giving a
genuine [low, high] review band.

If best_high < best_low (rare; occurs when the recall target requires a very
high reject threshold that overlaps the approve zone), the fallback widens the
review band.

Coherence-restoring fallback
-----------------------------
Two situations trigger the fallback (case = "fallback_triggered"):

  A. No threshold meets the approve-precision target
     → high defaults to 0.90, low to 0.0 (everything goes to review).
  B. best_high < best_low (targets met independently but incoherent)
     → Keep high = best_high; set low = 0.0 (no auto-rejects; widen review).

In both cases the review band is widened, prioritising caution over automation.

Rationale (thesis): when precision and recall targets conflict, the system
deliberately prioritises caution (more manual review) over forced automation.
This is a design choice, not an error state — the reviewer can inspect the
ThresholdResult.case field and report it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.model import Thresholds


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ThresholdResult:
    """
    Outcome of a validation-split threshold search.

    Attributes
    ----------
    high : float
        Auto-approve threshold: p >= high → auto-approve.
    low : float
        Auto-reject threshold: p <= low → auto-reject.
        Invariant: high >= low always holds.
    case : str
        "both_targets_met"  — approve precision AND reject recall targets were
                              simultaneously satisfied at a coherent (h >= l) pair.
        "fallback_triggered" — targets could not be met coherently; the review
                              band was widened to restore coherence.
    search_df : pd.DataFrame
        One row per grid point; includes per-point metrics and which targets
        were met.  Useful for the thesis appendix (threshold sensitivity table).
    """
    high:      float
    low:       float
    case:      str
    search_df: pd.DataFrame = field(repr=False)

    @property
    def thresholds(self) -> Thresholds:
        """Convert to the Thresholds dataclass used by the rest of the pipeline."""
        return Thresholds(high=self.high, low=self.low)


# ── Main search function ──────────────────────────────────────────────────────

def find_thresholds(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg:   PipelineConfig,
) -> ThresholdResult:
    """
    Search for coherent (high, low) thresholds on the VALIDATION split.

    Parameters
    ----------
    model : fitted CalibratedClassifierCV (or any predict_proba estimator)
        Must have been trained exclusively on training data — never on val or test.
    X_val : pd.DataFrame
        Feature matrix for the validation split.
    y_val : pd.Series
        Labels for the validation split (1 = match, 0 = non-match).
    cfg : PipelineConfig
        Uses cfg.thresholds.{strategy, approve_precision_target,
        reject_recall_target, grid_steps} and cfg.seed.

    Returns
    -------
    ThresholdResult
        Contains (high, low, case, search_df).  Coherence guaranteed: high >= low.
    """
    proba_val   = model.predict_proba(X_val)[:, 1]
    y_arr       = np.asarray(y_val)
    n_nonmatch  = int((y_arr == 0).sum())

    # Deterministic grid — seed is used for reproducibility in any future
    # extension that involves random sampling of the grid.
    rng  = np.random.default_rng(cfg.seed)   # noqa: F841  (reserved for future use)
    grid = np.linspace(0.01, 0.99, cfg.thresholds.grid_steps)

    rows: list[dict] = []

    for t in grid:
        t_f = float(t)

        # ── Approve-side metrics at this threshold ────────────────────────────
        approve_mask = proba_val >= t_f
        n_approved   = int(approve_mask.sum())
        if n_approved > 0:
            approve_prec = float(y_arr[approve_mask].mean())
        else:
            approve_prec = float("nan")

        # ── Reject-side metrics at this threshold ─────────────────────────────
        # recall_nonmatch = fraction of true non-matches that fall in reject zone
        reject_mask  = proba_val <= t_f
        n_rejected   = int(reject_mask.sum())
        if n_nonmatch > 0 and n_rejected > 0:
            reject_recall = float((y_arr[reject_mask] == 0).sum() / n_nonmatch)
        else:
            reject_recall = 0.0

        rows.append({
            "threshold":              round(t_f, 4),
            "approve_precision":      round(approve_prec, 4) if not np.isnan(approve_prec) else float("nan"),
            "approve_count":          n_approved,
            "reject_recall_nonmatch": round(reject_recall, 4),
            "reject_count":           n_rejected,
            "meets_precision_target": (
                not np.isnan(approve_prec)
                and approve_prec >= cfg.thresholds.approve_precision_target
            ),
            "meets_recall_target":    (
                reject_recall >= cfg.thresholds.reject_recall_target
            ),
        })

    search_df = pd.DataFrame(rows)

    # ── Step 1: find best_high ────────────────────────────────────────────────
    # Lowest t where precision(p >= t) >= approve_target.
    # Monotonically non-decreasing: taking min gives largest approve zone.
    valid_high_mask = search_df["meets_precision_target"]
    valid_high      = search_df.loc[valid_high_mask, "threshold"]
    best_high       = float(valid_high.min()) if not valid_high.empty else None

    # ── Step 2: find best_low ─────────────────────────────────────────────────
    # Lowest t where recall_nonmatch(p <= t) >= reject_target.
    # Taking min is conservative: only auto-reject when model is very confident
    # the pair is a non-match.  This avoids collapsing the review zone to zero
    # (which would happen if we took max, since recall_nonmatch is monotone).
    valid_low_mask = search_df["meets_recall_target"]
    valid_low      = search_df.loc[valid_low_mask, "threshold"]
    best_low       = float(valid_low.min()) if not valid_low.empty else None

    # ── Step 3: assign case and set final thresholds ──────────────────────────
    if best_high is not None and best_low is not None and best_high >= best_low:
        high = best_high
        low  = best_low
        case = "both_targets_met"
    else:
        case = "fallback_triggered"
        if best_high is not None and best_low is not None:
            # Both targets met independently but incoherent (high < low)
            # → keep precision guarantee; widen review band by dropping auto-rejects
            high = best_high
            low  = 0.0
        elif best_high is not None:
            # Recall target unreachable → no auto-rejects
            high = best_high
            low  = 0.0
        else:
            # Precision target itself unreachable (degenerate model or tiny data)
            high = 0.90
            low  = 0.0

    # Explicit sanity check — guaranteed by construction above
    assert high >= low, f"Coherence invariant violated: high={high} < low={low}"

    high = round(float(high), 4)
    low  = round(float(low),  4)

    search_df["selected_high"] = high
    search_df["selected_low"]  = low
    search_df["case"]          = case

    return ThresholdResult(
        high=high,
        low=low,
        case=case,
        search_df=search_df,
    )
