"""
Tests for margin-aware amount features in src/features.py.

Covers:
  - Exact match (margin = 0, within_margin = 1)
  - 10 % spread (margin ≈ 0.10, within_margin = 1 given tolerance)
  - Arbitrary non-margin spread (within_margin = 0)
  - Zero values (both-zero edge case, one-zero edge case)
  - NaN propagation (missing values → worst-case fill)
  - Coexistence: margin features sit alongside existing abs_diff / rel_diff / ratio features
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import pytest

from src.features import _margin_features, build_features, MARGIN_TOLERANCE, EXPECTED_MARGINS


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pair(left_val, right_val) -> tuple[pd.Series, pd.Series]:
    ln = pd.Series([float(left_val) if left_val is not None else np.nan])
    rn = pd.Series([float(right_val) if right_val is not None else np.nan])
    return ln, rn


def _pair_df(left_price, right_price) -> pd.DataFrame:
    """Single-row pair DataFrame with left_price / right_price columns."""
    return pd.DataFrame({"left_price": [left_price], "right_price": [right_price]})


# ── _margin_features unit tests ────────────────────────────────────────────────

class TestMarginFeaturesUnit:
    def test_exact_match(self):
        ln, rn = _pair(100, 100)
        out = _margin_features(ln, rn, "price", EXPECTED_MARGINS, MARGIN_TOLERANCE)
        assert out["price__margin"].iloc[0] == pytest.approx(0.0)
        assert out["price__min_margin_dev"].iloc[0] == pytest.approx(0.0)
        assert out["price__within_margin"].iloc[0] == 1

    def test_ten_percent_spread(self):
        # 100 vs 90 → margin = 1 - 90/100 = 0.10
        ln, rn = _pair(100, 90)
        out = _margin_features(ln, rn, "price", EXPECTED_MARGINS, MARGIN_TOLERANCE)
        assert out["price__margin"].iloc[0] == pytest.approx(0.10, abs=1e-9)
        assert out["price__min_margin_dev"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert out["price__within_margin"].iloc[0] == 1

    def test_arbitrary_spread_not_within_margin(self):
        # 100 vs 75 → margin = 0.25, nearest expected is 0.10, dev = 0.15
        ln, rn = _pair(100, 75)
        out = _margin_features(ln, rn, "price", EXPECTED_MARGINS, MARGIN_TOLERANCE)
        assert out["price__margin"].iloc[0] == pytest.approx(0.25, abs=1e-9)
        assert out["price__min_margin_dev"].iloc[0] == pytest.approx(0.15, abs=1e-4)
        assert out["price__within_margin"].iloc[0] == 0

    def test_both_zero(self):
        # Both zero → defined as exact match
        ln, rn = _pair(0, 0)
        out = _margin_features(ln, rn, "price", EXPECTED_MARGINS, MARGIN_TOLERANCE)
        assert out["price__margin"].iloc[0] == pytest.approx(0.0)
        assert out["price__within_margin"].iloc[0] == 1

    def test_one_zero(self):
        # 100 vs 0 → margin = 1.0 (max spread)
        ln, rn = _pair(100, 0)
        out = _margin_features(ln, rn, "price", EXPECTED_MARGINS, MARGIN_TOLERANCE)
        assert out["price__margin"].iloc[0] == pytest.approx(1.0)
        assert out["price__within_margin"].iloc[0] == 0

    def test_nan_left_fills_worst_case(self):
        ln, rn = _pair(None, 100)
        out = _margin_features(ln, rn, "price", EXPECTED_MARGINS, MARGIN_TOLERANCE)
        assert out["price__margin"].iloc[0] == pytest.approx(1.0)
        assert out["price__min_margin_dev"].iloc[0] == pytest.approx(1.0)
        assert out["price__within_margin"].iloc[0] == 0

    def test_nan_right_fills_worst_case(self):
        ln, rn = _pair(100, None)
        out = _margin_features(ln, rn, "price", EXPECTED_MARGINS, MARGIN_TOLERANCE)
        assert out["price__margin"].iloc[0] == pytest.approx(1.0)
        assert out["price__within_margin"].iloc[0] == 0

    def test_custom_tolerance(self):
        # margin = 0.05, default tolerance = 0.005, but with wider tolerance = 0.10 → within
        ln, rn = _pair(100, 95)
        out_narrow = _margin_features(ln, rn, "price", [0.0], margin_tolerance=0.005)
        out_wide   = _margin_features(ln, rn, "price", [0.0], margin_tolerance=0.10)
        assert out_narrow["price__within_margin"].iloc[0] == 0
        assert out_wide["price__within_margin"].iloc[0] == 1

    def test_symmetric_order(self):
        # Margin should be the same regardless of which side is bigger
        ln_a, rn_a = _pair(100, 90)
        ln_b, rn_b = _pair(90, 100)
        out_a = _margin_features(ln_a, rn_a, "price", EXPECTED_MARGINS, MARGIN_TOLERANCE)
        out_b = _margin_features(ln_b, rn_b, "price", EXPECTED_MARGINS, MARGIN_TOLERANCE)
        assert out_a["price__margin"].iloc[0] == pytest.approx(out_b["price__margin"].iloc[0])


# ── build_features integration tests ─────────────────────────────────────────

class TestBuildFeaturesMarginCoexistence:
    def test_margin_cols_present_for_amount_field(self):
        df = _pair_df(100, 90)
        X, names = build_features(df, amount_fields=["price"])
        assert "price__margin" in names
        assert "price__min_margin_dev" in names
        assert "price__within_margin" in names

    def test_existing_numeric_cols_still_present(self):
        df = _pair_df(100, 90)
        X, names = build_features(df, amount_fields=["price"])
        assert "price__abs_diff" in names
        assert "price__rel_diff" in names
        assert "price__both_present" in names

    def test_non_amount_field_has_no_margin_cols(self):
        df = pd.DataFrame({
            "left_description": ["widget A"],
            "right_description": ["widget B"],
        })
        X, names = build_features(df, amount_fields=["price"])
        assert not any("__margin" in n for n in names)

    def test_exact_match_pipeline(self):
        df = _pair_df(50.0, 50.0)
        X, names = build_features(df, amount_fields=["price"])
        assert X["price__margin"].iloc[0] == pytest.approx(0.0)
        assert X["price__within_margin"].iloc[0] == 1

    def test_10pct_margin_pipeline(self):
        df = _pair_df(100.0, 90.0)
        X, names = build_features(df, amount_fields=["price"])
        assert X["price__margin"].iloc[0] == pytest.approx(0.10, abs=1e-9)
        assert X["price__within_margin"].iloc[0] == 1

    def test_no_nans_in_output(self):
        rows = [
            {"left_price": 100, "right_price": 90},
            {"left_price": None, "right_price": 50},
            {"left_price": 0,    "right_price": 0},
        ]
        df = pd.DataFrame(rows)
        X, _ = build_features(df, amount_fields=["price"])
        assert not X.isnull().any().any()

    def test_vector_batch(self):
        """build_features handles multiple rows correctly (vectorised)."""
        prices_l = [100, 200, 300]
        prices_r = [100, 180, 400]
        df = pd.DataFrame({"left_price": prices_l, "right_price": prices_r})
        X, names = build_features(df, amount_fields=["price"])
        expected_margins = [0.0, 1 - 180/200, 1 - 300/400]
        for i, em in enumerate(expected_margins):
            assert X["price__margin"].iloc[i] == pytest.approx(em, abs=1e-6)

    def test_custom_expected_margins(self):
        # With expected_margins=[0.05], a 5% spread should be within_margin
        df = _pair_df(100.0, 95.0)
        X, _ = build_features(df, amount_fields=["price"],
                               expected_margins=[0.05], margin_tolerance=0.005)
        assert X["price__within_margin"].iloc[0] == 1
