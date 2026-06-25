"""
End-to-end tests for predict.py.

Uses synthetic, fully in-process data (no disk fixtures):
  - 60-row training CSV (30 match + 30 no-match) written to a tmp path
  - 4 invoice rows, 4 reference rows (enough to exercise best-match selection)
  - Verifies: one output row per invoice, required columns present,
    probability in [0, 1], action in known set, margin_consistent is bool.
"""
from __future__ import annotations
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from predict import run_predict


# ── Fixture builders ───────────────────────────────────────────────────────────

def _make_training_csv(path: str, n_pairs: int = 60) -> None:
    """
    Synthetic training CSV: left_price, right_price, left_title, right_title, label.
    Matches: same price ± tiny noise; no-matches: very different prices.
    """
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_pairs):
        is_match = i % 2 == 0
        base_price = rng.uniform(10, 500)
        l_price = round(float(base_price), 2)
        if is_match:
            r_price = round(float(base_price * (1 + rng.uniform(-0.02, 0.02))), 2)
            l_title = r_title = f"item_{i}"
        else:
            r_price = round(float(base_price * rng.uniform(1.5, 3.0)), 2)
            l_title = f"item_left_{i}"
            r_title = f"item_right_{i}"
        rows.append({
            "left_price": l_price,
            "right_price": r_price,
            "left_title": l_title,
            "right_title": r_title,
            "label": int(is_match),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_invoices_csv(path: str) -> None:
    rows = [
        {"title": "acme widget",    "price": 100.0},
        {"title": "beta gadget",    "price": 200.0},
        {"title": "gamma doohickey","price": 300.0},
        {"title": "delta thingamajig", "price": 50.0},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_reference_csv(path: str) -> None:
    rows = [
        {"title": "acme widget",    "price": 100.0},   # exact match for invoice 0
        {"title": "beta gadget",    "price": 180.0},   # 10 % margin for invoice 1
        {"title": "zeta unrelated", "price": 999.0},   # no obvious match
        {"title": "delta thingamajig", "price": 50.0}, # exact match for invoice 3
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestRunPredict:
    @pytest.fixture(autouse=True)
    def tmp_files(self, tmp_path):
        self.train_csv  = str(tmp_path / "train.csv")
        self.inv_csv    = str(tmp_path / "invoices.csv")
        self.ref_csv    = str(tmp_path / "reference.csv")
        self.out_csv    = str(tmp_path / "out" / "predictions.csv")
        _make_training_csv(self.train_csv)
        _make_invoices_csv(self.inv_csv)
        _make_reference_csv(self.ref_csv)

    def _run(self, **kwargs) -> pd.DataFrame:
        defaults = dict(
            invoices_path=self.inv_csv,
            reference_path=self.ref_csv,
            train_path=self.train_csv,
            out_path=self.out_csv,
            config_path="nonexistent_config.yaml",  # avoid reading project config
            amount_fields=["price"],
            expected_margins=[0.0, 0.10],
            margin_tolerance=0.005,
        )
        defaults.update(kwargs)
        return run_predict(**defaults)

    def test_one_row_per_invoice(self):
        result = self._run()
        assert len(result) == 4

    def test_required_columns_present(self):
        result = self._run()
        for col in ("match_probability", "action", "margin_consistent"):
            assert col in result.columns, f"Missing column: {col}"

    def test_probability_in_unit_interval(self):
        result = self._run()
        assert (result["match_probability"] >= 0.0).all()
        assert (result["match_probability"] <= 1.0).all()

    def test_action_values_valid(self):
        valid = {"auto-approve", "flag-review", "auto-reject"}
        result = self._run()
        unknown = set(result["action"].unique()) - valid
        assert not unknown, f"Unexpected action values: {unknown}"

    def test_margin_consistent_is_bool_like(self):
        result = self._run()
        # May be stored as bool or int 0/1
        assert result["margin_consistent"].isin([True, False, 0, 1]).all()

    def test_output_file_created(self):
        self._run()
        assert os.path.exists(self.out_csv)

    def test_output_file_matches_returned_df(self):
        result = self._run()
        saved = pd.read_csv(self.out_csv)
        assert len(saved) == len(result)
        assert list(saved.columns) == list(result.columns)

    def test_custom_thresholds_respected(self):
        # approve=1.0 means nothing auto-approved; reject=0.0 means nothing auto-rejected
        result = self._run(approve_threshold=1.0, reject_threshold=0.0)
        assert (result["action"] == "flag-review").all()

    def test_approve_threshold_respected(self):
        # Very low approve threshold: every high-confidence pair gets auto-approved
        result = self._run(approve_threshold=0.0, reject_threshold=-1.0)
        assert (result["action"] == "auto-approve").all()

    def test_invoice_columns_present_without_prefix(self):
        result = self._run()
        # "title" and "price" from invoices.csv should appear without left_ prefix
        assert "title" in result.columns
        assert "price" in result.columns

    def test_ref_columns_present_with_ref_prefix(self):
        result = self._run()
        # Reference fields should appear with ref_ prefix
        ref_cols = [c for c in result.columns if c.startswith("ref_")]
        assert len(ref_cols) > 0, "No ref_* columns in output"

    def test_no_duplicate_invoice_rows(self):
        result = self._run()
        # Invoice index should be unique (one match per invoice)
        assert result.index.is_unique

    def test_already_prefixed_invoices_handled(self):
        # If invoices already have left_ prefix, should not double-prefix
        inv = pd.read_csv(self.inv_csv)
        inv = inv.rename(columns={c: f"left_{c}" for c in inv.columns})
        prefixed_path = self.inv_csv.replace("invoices.csv", "inv_prefixed.csv")
        inv.to_csv(prefixed_path, index=False)
        result = self._run(invoices_path=prefixed_path)
        assert len(result) == 4
        # Columns should not contain double-prefix
        assert not any("left_left_" in c for c in result.columns)
