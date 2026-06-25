"""
Blocking: reduce candidate pairs before classification.

A full cross-join of N invoices × M reference records is O(N·M). Blocking
uses cheap, high-recall keys to discard pairs very unlikely to match before
running the slower feature-engineering + classifier step.

Blocking keys (any one hit → pair is retained):
  1. vendor_key   – first-3-char normalised vendor/title prefix
  2. amount_bucket – total/amount/price rounded to nearest bucket_width
  3. margin_band  – for each non-zero expected_margin m, the left amount
                    scaled by 1/(1-m) and (1-m) to catch both directions
  4. year_bucket  – year field (or any numeric temporal field) bucketed to
                    nearest bucket_width (default 1 = exact-year buckets)

Keys 1–3 serve the Walmart-Amazon / invoice domain.
Key 4 serves DBLP-ACM and other temporally-keyed datasets.

Both datasets are configured via config/[dataset].yaml so no dataset-specific
code lives here — the column search order comes from cfg.blocking.

Falls back to a full cross-join when no blocking key is found in the data
(acceptable only for small datasets; warns to stdout).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.config import PipelineConfig


# ── Key extractors ────────────────────────────────────────────────────────────

def _vendor_key(s: pd.Series) -> pd.Series:
    """First 3 lowercase alphanumeric characters of a text field."""
    clean = s.fillna("").str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return clean.str[:3]


def _amount_bucket(s: pd.Series, width: float = 10.0) -> pd.Series:
    """Alias kept for backward compatibility with existing tests."""
    return _numeric_bucket(s, width)


def _numeric_bucket(s: pd.Series, width: float) -> pd.Series:
    """Round a numeric series to the nearest bucket boundary; NaN if unparseable."""
    nums = pd.to_numeric(
        s.astype(str).str.replace(r"[^\d.]", "", regex=True),
        errors="coerce",
    )
    return np.floor(nums / width) * width   # NaN propagates → excluded from join


def _find_col(df: pd.DataFrame, prefix: str, candidates: list[str]) -> str | None:
    """Return the first column named prefix+candidate that exists in df."""
    for name in candidates:
        col = f"{prefix}{name}"
        if col in df.columns:
            return col
    return None


# ── Main blocking function ────────────────────────────────────────────────────

def block_pairs(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    cfg: PipelineConfig | None = None,
    *,
    amount_bucket_width:  float        = 10.0,
    expected_margins:     list[float] | None = None,
    vendor_key_fields:    list[str]   | None = None,
    amount_key_fields:    list[str]   | None = None,
    year_key_fields:      list[str]   | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of candidate pairs with left_* and right_* columns.

    Parameters
    ----------
    left_df, right_df : pd.DataFrame
        Must already carry their left_*/right_* prefix.  If a DataFrame has
        no prefixed columns the prefix is added automatically.
    cfg : PipelineConfig, optional
        When provided, all blocking settings come from cfg.blocking and
        cfg.features.  Explicit kwargs are ignored when cfg is set.
    amount_bucket_width : float
        Bucket width for amount and year blocking (ignored when cfg is set).
    expected_margins : list[float]
        Non-zero margins trigger margin-band blocking (ignored when cfg is set).
    vendor_key_fields, amount_key_fields, year_key_fields : list[str]
        Column search order for each blocking key (ignored when cfg is set).

    Returns
    -------
    pd.DataFrame
        Candidate pair rows with left_* and right_* columns interleaved.
        Tracking columns (left___inv_idx, right___ref_idx) pass through if
        present in the inputs.
    """
    # ── Resolve settings ──────────────────────────────────────────────────────
    if cfg is not None:
        _bucket_w   = cfg.blocking.amount_bucket_width
        _margins    = cfg.features.expected_margins
        _vk_fields  = cfg.blocking.vendor_key_fields
        _ak_fields  = cfg.blocking.amount_key_fields
        _yk_fields  = cfg.blocking.year_key_fields
    else:
        _bucket_w   = amount_bucket_width
        _margins    = expected_margins or []
        _vk_fields  = vendor_key_fields or ["vendor", "title", "name", "supplier"]
        _ak_fields  = amount_key_fields or ["total", "amount", "price", "unit_price"]
        _yk_fields  = year_key_fields   or []

    # ── Normalise input DataFrames ────────────────────────────────────────────
    left  = left_df.copy().reset_index(drop=True)
    right = right_df.copy().reset_index(drop=True)

    if not any(c.startswith("left_") for c in left.columns):
        left = left.rename(columns={c: f"left_{c}" for c in left.columns})
    if not any(c.startswith("right_") for c in right.columns):
        right = right.rename(columns={c: f"right_{c}" for c in right.columns})

    left["__li"]  = range(len(left))
    right["__ri"] = range(len(right))

    # ── Locate blocking columns in each DataFrame ─────────────────────────────
    lv_col = _find_col(left,  "left_",  _vk_fields)
    rv_col = _find_col(right, "right_", _vk_fields)
    la_col = _find_col(left,  "left_",  _ak_fields)
    ra_col = _find_col(right, "right_", _ak_fields)
    ly_col = _find_col(left,  "left_",  _yk_fields)
    ry_col = _find_col(right, "right_", _yk_fields)

    pair_frames: list[pd.DataFrame] = []

    # ── 1. Vendor/title prefix key ────────────────────────────────────────────
    if lv_col and rv_col:
        lk = pd.DataFrame({"__li": left["__li"],  "__vk": _vendor_key(left[lv_col])})
        rk = pd.DataFrame({"__ri": right["__ri"], "__vk": _vendor_key(right[rv_col])})
        lk = lk[lk["__vk"].str.len() > 0]
        rk = rk[rk["__vk"].str.len() > 0]
        if not lk.empty and not rk.empty:
            pair_frames.append(lk.merge(rk, on="__vk")[["__li", "__ri"]])

    # ── 2. Amount bucket ──────────────────────────────────────────────────────
    if la_col and ra_col:
        lk = pd.DataFrame({"__li": left["__li"],  "__ab": _numeric_bucket(left[la_col],  _bucket_w)})
        rk = pd.DataFrame({"__ri": right["__ri"], "__ab": _numeric_bucket(right[ra_col], _bucket_w)})
        lk = lk.dropna(subset=["__ab"])
        rk = rk.dropna(subset=["__ab"])
        if not lk.empty and not rk.empty:
            pair_frames.append(lk.merge(rk, on="__ab")[["__li", "__ri"]])

    # ── 3. Margin-band blocking ───────────────────────────────────────────────
    # For each non-zero expected margin m, scale the left amount in both
    # directions so a pair where one side is (1-m)× the other lands in the
    # same bucket.  m=0 is already covered by amount_bucket above.
    if la_col and ra_col and _margins:
        la_vals = pd.to_numeric(
            left[la_col].astype(str).str.replace(r"[^\d.]", "", regex=True),
            errors="coerce",
        )
        r_buckets = _numeric_bucket(right[ra_col], _bucket_w)
        rk = pd.DataFrame({"__ri": right["__ri"], "__mb": r_buckets}).dropna(subset=["__mb"])

        for m in _margins:
            if abs(m) < 1e-9:
                continue
            for scale in [1.0 / max(1.0 - m, 1e-6), 1.0 - m]:
                scaled = la_vals * scale
                lk = pd.DataFrame({
                    "__li": left["__li"],
                    "__mb": np.floor(scaled / _bucket_w) * _bucket_w,
                })
                lk = lk.dropna(subset=["__mb"])
                if not lk.empty and not rk.empty:
                    pair_frames.append(lk.merge(rk, on="__mb")[["__li", "__ri"]])

    # ── 4. Year / temporal bucket ─────────────────────────────────────────────
    if ly_col and ry_col:
        lk = pd.DataFrame({"__li": left["__li"],  "__yb": _numeric_bucket(left[ly_col],  _bucket_w)})
        rk = pd.DataFrame({"__ri": right["__ri"], "__yb": _numeric_bucket(right[ry_col], _bucket_w)})
        lk = lk.dropna(subset=["__yb"])
        rk = rk.dropna(subset=["__yb"])
        if not lk.empty and not rk.empty:
            pair_frames.append(lk.merge(rk, on="__yb")[["__li", "__ri"]])

    # ── Assemble candidate pairs ──────────────────────────────────────────────
    if not pair_frames:
        warnings.warn(
            f"block_pairs: no blocking key found — falling back to full cross-join "
            f"({len(left)}×{len(right)} = {len(left)*len(right)} pairs). "
            "This is only safe for small datasets.",
            RuntimeWarning,
            stacklevel=2,
        )
        l_idx = pd.Series(range(len(left)),  name="__li")
        r_idx = pd.Series(range(len(right)), name="__ri")
        pairs = l_idx.to_frame().merge(r_idx.to_frame(), how="cross")
    else:
        pairs = (
            pd.concat(pair_frames, ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )

    # ── Materialise column data ───────────────────────────────────────────────
    l_cols = [c for c in left.columns  if c.startswith("left_")]
    r_cols = [c for c in right.columns if c.startswith("right_")]

    left_part  = left.loc[pairs["__li"].to_numpy(),  l_cols].reset_index(drop=True)
    right_part = right.loc[pairs["__ri"].to_numpy(), r_cols].reset_index(drop=True)

    return pd.concat([left_part, right_part], axis=1).reset_index(drop=True)
