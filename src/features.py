"""
Pairwise feature engineering.

For each candidate pair (left record vs right record) interpretable similarity
features are computed.  These mirror the checks a finance approver makes when
reconciling an invoice line against a reference record.

Feature groups
--------------
text fields (description / vendor / category)
    fuzzy string similarity via rapidfuzz: ratio, token_sort_ratio, token_set_ratio,
    and an exact-match flag.

numeric fields (price / amount / hours)
    absolute difference, relative difference (clipped at 5×), and a
    ``both_present`` flag (1 when neither value is NaN).

amount fields (monetary amounts — a subset of numeric fields)
    three additional margin-aware features (see below).

Margin-aware amount features
----------------------------
For fields listed in ``amount_fields`` (or cfg.features.amount_fields) the
pipeline adds three extra features on top of the standard numeric differences:

  {field}__margin
      Implied spread: 1 - min(a, b) / max(a, b).
      0.0 = exact match, 0.10 = 10 % spread, 1.0 = one value is zero.

  {field}__min_margin_dev
      Minimum absolute distance from any value in ``expected_margins``.
      Near-zero means the implied spread is close to a known / expected margin
      (e.g. 0 % for identical amounts, 10 % for a standard mark-up).

  {field}__within_margin
      1 if min_margin_dev <= margin_tolerance, else 0.

These features coexist with (do not replace) abs_diff, rel_diff, and
both_present.

Active margin case — Walmart-Amazon vs real Xebia deployment
------------------------------------------------------------
With ``expected_margins: [0.0, 0.10]``:

* On **Walmart-Amazon** benchmark data, true matches have nearly identical
  prices (two retailers listing the same product).  The 0.0 (0 %) margin entry
  is therefore the active case — nearly all genuine pairs land near zero spread.
  The 0.10 (10 %) entry is present for configuration consistency but is
  effectively inert on this dataset.

* On **real Xebia deployment data**, purchase invoices from international sister
  companies are expected to equal ~90 % of the corresponding sales amount
  (the fixed 10 % intercompany margin rule).  Here the 0.10 entry is the active
  case and the feature directly encodes the domain rule.

Proxy framing — Walmart-Amazon fields
--------------------------------------
  title     → vendor / company name variation across invoices and ERP records
  price     → invoice amount; margin feature encodes the intercompany rule
  category  → project / cost-centre / period alignment
  brand     → secondary vendor identifier
  modelno   → internal reference / PO number analogue

Proxy framing — DBLP-ACM fields
---------------------------------
  title     → record description (paper title → invoice description analogue)
  authors   → secondary text identifier
  venue     → series / publication channel
  year      → temporal alignment (→ invoice period / fiscal year analogue)
  (no amount field: margin features are deliberately absent for this dataset)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.config import PipelineConfig

# ── Module-level defaults (override via cfg or explicit kwargs) ───────────────
AMOUNT_FIELDS: list[str]   = ["amount", "price", "total", "net", "gross"]
EXPECTED_MARGINS: list[float] = [0.0, 0.10]
MARGIN_TOLERANCE: float    = 0.005


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _is_numeric_field(left: pd.Series, right: pd.Series, thresh: float = 0.6) -> bool:
    """Heuristic: field is numeric if ≥ thresh fraction of values parse as float."""
    both = pd.concat([_to_float(left), _to_float(right)])
    return both.notna().mean() >= thresh


def shared_fields(df: pd.DataFrame) -> list[str]:
    """Return field names that appear in both left_* and right_* columns."""
    left  = {c[len("left_"):] for c in df.columns if c.startswith("left_")}
    right = {c[len("right_"):] for c in df.columns if c.startswith("right_")}
    return sorted(left & right)


# ── Margin feature computation ────────────────────────────────────────────────

def _margin_features(
    ln: pd.Series,
    rn: pd.Series,
    field: str,
    expected_margins: list[float],
    margin_tolerance: float,
) -> dict[str, pd.Series]:
    """
    Compute margin, min_margin_dev, and within_margin for a single amount field.

    Parameters
    ----------
    ln, rn : pd.Series
        Numeric values for the left and right records (NaN for missing).
    field : str
        Base field name (used as prefix in output keys).
    expected_margins : list[float]
        The set of margin values considered "normal" (e.g. [0.0, 0.10]).
    margin_tolerance : float
        A pair is within margin when min_margin_dev <= this value.

    Returns
    -------
    dict mapping feature name → pd.Series (same index as ln/rn).
    """
    a = ln.clip(lower=0)  # monetary amounts are non-negative
    b = rn.clip(lower=0)

    max_ab = pd.concat([a, b], axis=1).max(axis=1)
    min_ab = pd.concat([a, b], axis=1).min(axis=1)

    # margin = 1 - min/max
    # both-zero → exact match → margin = 0.0
    # either missing → NaN (filled with worst-case 1.0 below)
    margin = 1.0 - min_ab / max_ab.replace(0.0, np.nan)
    margin = margin.where(max_ab > 0, 0.0)
    margin = margin.where(ln.notna() & rn.notna())

    if expected_margins:
        devs = np.abs(
            np.stack([margin.values] * len(expected_margins))
            - np.asarray(expected_margins)[:, np.newaxis]
        )
        with np.errstate(all="ignore"):    # suppress nanmin-on-all-nan warning
            min_dev_vals = np.nanmin(devs, axis=0)
        min_dev_vals[np.isnan(margin.values)] = np.nan
        min_dev = pd.Series(min_dev_vals, index=margin.index)
    else:
        min_dev = pd.Series(np.full(len(margin), np.nan), index=margin.index)

    within = (min_dev <= margin_tolerance).fillna(False).astype(int)

    return {
        f"{field}__margin":         margin.fillna(1.0),
        f"{field}__min_margin_dev": min_dev.fillna(1.0),
        f"{field}__within_margin":  within,
    }


# ── Main feature builder ──────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    cfg: PipelineConfig | None = None,
    *,
    amount_fields:    list[str]   | None = None,
    expected_margins: list[float] | None = None,
    margin_tolerance: float = MARGIN_TOLERANCE,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the pairwise feature matrix for a set of candidate pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Candidate pair rows.  Must have left_* and right_* columns.
    cfg : PipelineConfig, optional
        When provided its features section takes precedence over the explicit
        kwargs below.  Pass ``cfg`` in all new pipeline code; the kwargs exist
        for backward compatibility with tests and predict.py.
    amount_fields : list[str], optional
        Fields that receive margin-aware features in addition to numeric diffs.
        Ignored when cfg is provided.
    expected_margins : list[float], optional
        Margins considered "normal" for margin feature computation.
        Ignored when cfg is provided.
    margin_tolerance : float
        within_margin = 1 when min_margin_dev <= this value.
        Ignored when cfg is provided.

    Returns
    -------
    (X, feature_names) : (pd.DataFrame, list[str])
        X has no NaN values (filled with 0.0 at the end).
    """
    # Resolve settings: cfg > explicit kwargs > module defaults
    if cfg is not None:
        _amount_fields    = cfg.features.amount_fields
        _expected_margins = cfg.features.expected_margins
        _margin_tolerance = cfg.features.margin_tolerance
        _text_override    = set(cfg.features.text_fields)
        _numeric_override = set(cfg.features.numeric_fields)
    else:
        _amount_fields    = amount_fields    if amount_fields    is not None else AMOUNT_FIELDS
        _expected_margins = expected_margins if expected_margins is not None else EXPECTED_MARGINS
        _margin_tolerance = margin_tolerance
        _text_override    = set()
        _numeric_override = set()

    feats = pd.DataFrame(index=df.index)

    for field in shared_fields(df):
        lcol, rcol = f"left_{field}", f"right_{field}"
        lvals = df[lcol].fillna("").astype(str)
        rvals = df[rcol].fillna("").astype(str)

        # Determine numeric vs text:
        # explicit config override > _is_numeric_field heuristic
        if field in _numeric_override:
            is_numeric = True
        elif field in _text_override:
            is_numeric = False
        else:
            is_numeric = _is_numeric_field(df[lcol], df[rcol])

        if is_numeric:
            ln, rn   = _to_float(df[lcol]), _to_float(df[rcol])
            abs_diff = (ln - rn).abs()
            denom    = pd.concat([ln.abs(), rn.abs()], axis=1).max(axis=1).replace(0, np.nan)
            rel_diff = (abs_diff / denom).fillna(0.0)

            feats[f"{field}__abs_diff"]     = abs_diff.fillna(abs_diff.max())
            feats[f"{field}__rel_diff"]     = rel_diff.clip(0, 5)
            feats[f"{field}__both_present"] = (ln.notna() & rn.notna()).astype(int)

            if field in _amount_fields:
                for fname, fseries in _margin_features(
                    ln, rn, field, _expected_margins, _margin_tolerance
                ).items():
                    feats[fname] = fseries
        else:
            feats[f"{field}__ratio"]      = [fuzz.ratio(a, b) / 100            for a, b in zip(lvals, rvals)]
            feats[f"{field}__token_sort"] = [fuzz.token_sort_ratio(a, b) / 100 for a, b in zip(lvals, rvals)]
            feats[f"{field}__token_set"]  = [fuzz.token_set_ratio(a, b) / 100  for a, b in zip(lvals, rvals)]
            feats[f"{field}__exact"]      = (lvals.str.lower() == rvals.str.lower()).astype(int)

    feats = feats.fillna(0.0)
    return feats, list(feats.columns)
