"""Tests for src/block.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.block import _amount_bucket, _vendor_key, block_pairs


# ---------------------------------------------------------------------------
# Unit tests: key functions
# ---------------------------------------------------------------------------

class TestVendorKey:
    def test_lowercases_and_truncates(self):
        s = pd.Series(["ACME Corp", "acme ltd", "XYZ"])
        keys = _vendor_key(s)
        assert keys.iloc[0] == "acm"
        assert keys.iloc[1] == "acm"
        assert keys.iloc[2] == "xyz"

    def test_strips_punctuation(self):
        keys = _vendor_key(pd.Series(["A&B Supply"]))
        assert keys.iloc[0] == "abs"

    def test_empty_string_stays_empty(self):
        keys = _vendor_key(pd.Series([""]))
        assert keys.iloc[0] == ""

    def test_nan_treated_as_empty(self):
        keys = _vendor_key(pd.Series([None, float("nan")]))
        assert all(k == "" for k in keys)


class TestAmountBucket:
    def test_rounds_to_bucket(self):
        buckets = _amount_bucket(pd.Series(["25.00", "34.99", "10.00"]), width=10.0)
        assert buckets.iloc[0] == 20.0
        assert buckets.iloc[1] == 30.0
        assert buckets.iloc[2] == 10.0

    def test_strips_currency_symbol(self):
        buckets = _amount_bucket(pd.Series(["$99.99", "£50.00"]), width=10.0)
        assert buckets.iloc[0] == 90.0
        assert buckets.iloc[1] == 50.0

    def test_nan_for_unparseable(self):
        buckets = _amount_bucket(pd.Series(["N/A", ""]), width=10.0)
        assert buckets.isna().all()


# ---------------------------------------------------------------------------
# Integration tests: block_pairs
# ---------------------------------------------------------------------------

def _left(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{f"left_{k}": v for k, v in r.items()} for r in records])


def _right(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{f"right_{k}": v for k, v in r.items()} for r in records])


class TestBlockPairs:
    def test_vendor_match_creates_pair(self):
        left = _left([{"vendor": "ACME Corp", "total": "100.00"}])
        right = _right([{"vendor": "ACME Ltd", "total": "200.00"}])
        pairs = block_pairs(left, right)
        # Both have vendor key "acm" → should produce a pair
        assert len(pairs) == 1
        assert "left_vendor" in pairs.columns
        assert "right_vendor" in pairs.columns

    def test_amount_match_creates_pair(self):
        left = _left([{"vendor": "AAA", "total": "105.00"}])
        right = _right([{"vendor": "ZZZ", "total": "108.00"}])  # same bucket (100)
        pairs = block_pairs(left, right, amount_bucket_width=10.0)
        assert len(pairs) >= 1

    def test_no_match_on_either_key_produces_no_pair(self):
        left = _left([{"vendor": "ACME", "total": "100.00"}])
        right = _right([{"vendor": "XYZ Inc", "total": "500.00"}])
        pairs = block_pairs(left, right, amount_bucket_width=10.0)
        assert len(pairs) == 0

    def test_deduplication_of_pairs(self):
        """A pair matching on both keys should appear only once."""
        left = _left([{"vendor": "ACME Corp", "total": "105.00"}])
        right = _right([{"vendor": "ACME Ltd", "total": "108.00"}])
        pairs = block_pairs(left, right, amount_bucket_width=10.0)
        assert len(pairs) == 1

    def test_output_columns_are_left_and_right_prefixed(self):
        left = _left([{"vendor": "ACME", "total": "50.00", "description": "Widget"}])
        right = _right([{"vendor": "ACME", "total": "50.00", "description": "Widget"}])
        pairs = block_pairs(left, right)
        assert all(c.startswith("left_") or c.startswith("right_") for c in pairs.columns)

    def test_cross_join_fallback_when_no_blocking_columns(self):
        """When no vendor or amount columns exist, all pairs should be returned."""
        left = pd.DataFrame([{"left_note": "a"}, {"left_note": "b"}])
        right = pd.DataFrame([{"right_note": "x"}, {"right_note": "y"}])
        pairs = block_pairs(left, right)
        assert len(pairs) == 4  # 2 × 2 cross join

    def test_empty_vendor_not_used_as_key(self):
        """Records with empty vendor strings must not match each other via vendor key."""
        left = _left([{"vendor": "", "total": "100.00"}])
        right = _right([{"vendor": "", "total": "200.00"}])
        pairs = block_pairs(left, right, amount_bucket_width=10.0)
        # Different amount buckets, empty vendors → no pair
        assert len(pairs) == 0

    def test_adds_prefix_to_unprefixed_input(self):
        left = pd.DataFrame([{"vendor": "ACME", "total": "50.00"}])
        right = pd.DataFrame([{"vendor": "ACME", "total": "50.00"}])
        pairs = block_pairs(left, right)
        assert "left_vendor" in pairs.columns
        assert "right_vendor" in pairs.columns

    def test_multiple_left_records(self):
        left = _left([
            {"vendor": "ACME Corp", "total": "100.00"},
            {"vendor": "XYZ Ltd",   "total": "200.00"},
        ])
        right = _right([{"vendor": "ACME Ltd", "total": "100.00"}])
        pairs = block_pairs(left, right, amount_bucket_width=10.0)
        assert len(pairs) >= 1
        # Only the ACME record should match (by vendor or exact amount bucket)
        vendors_matched = pairs["left_vendor"].unique()
        assert any("ACME" in v for v in vendors_matched)
