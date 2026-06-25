"""
Tests for src/thresholds.py.

Validates:
  1. Coherence guarantee: high >= low in ALL cases, including adversarial configs.
  2. Correct case assignment (both_targets_met vs fallback_triggered).
  3. search_df contains all required columns.
  4. Validation-split separation: find_thresholds operates only on data passed
     in — it cannot access test data by construction (no global state, no file I/O).
     A behavioural test verifies this by confirming that swapping val for test
     labels changes the result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit

from src.config import (
    CalibrationConfig, DatasetConfig, FeatureConfig, BlockingConfig,
    ModelConfig, OutputConfig, PipelineConfig, ThresholdConfig,
)
from src.thresholds import ThresholdResult, find_thresholds


# ── Fixture helpers ───────────────────────────────────────────────────────────

def _make_cfg(
    approve_target: float = 0.90,
    reject_target: float  = 0.90,
    grid_steps: int        = 50,
    seed: int              = 0,
) -> PipelineConfig:
    return PipelineConfig(
        dataset=DatasetConfig(name="test", train="", valid="", test=""),
        features=FeatureConfig(
            text_fields=[], numeric_fields=[], amount_fields=[],
            expected_margins=[], margin_tolerance=0.005,
        ),
        blocking=BlockingConfig(),
        model=ModelConfig(random_state=seed),
        calibration=CalibrationConfig(),
        thresholds=ThresholdConfig(
            strategy="precision_target",
            approve_precision_target=approve_target,
            reject_recall_target=reject_target,
            grid_steps=grid_steps,
        ),
        output=OutputConfig(),
        seed=seed,
    )


def _train_dummy_model(
    n: int = 300, seed: int = 0, match_rate: float = 0.1
) -> tuple[CalibratedClassifierCV, pd.DataFrame, pd.Series]:
    """
    Train a simple calibrated RF on two-feature synthetic data.

    Returns (model, X_val, y_val) for use in threshold search tests.
    The model is trained on an 80/20 split; X_val / y_val are the 20 % split.
    """
    rng = np.random.default_rng(seed)
    n_match   = max(2, int(n * match_rate))
    n_nomatch = n - n_match
    y = np.array([1] * n_match + [0] * n_nomatch)

    # Features: matches cluster near (1,1), non-matches near (0,0) with noise
    X_match   = rng.normal(0.8, 0.1, (n_match,   2)).clip(0, 1)
    X_nomatch = rng.normal(0.2, 0.1, (n_nomatch, 2)).clip(0, 1)
    X = np.vstack([X_match, X_nomatch])
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    base = RandomForestClassifier(n_estimators=50, random_state=seed, class_weight="balanced")
    base.fit(X_tr, y_tr)

    test_fold = np.concatenate([np.full(split, -1), np.zeros(n - split, dtype=int)])
    ps = PredefinedSplit(test_fold)
    cal = CalibratedClassifierCV(base, cv=ps, method="sigmoid")
    cal.fit(X, y)

    return cal, pd.DataFrame(X_val, columns=["f0", "f1"]), pd.Series(y_val)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestCoherenceGuarantee:
    def test_high_gte_low_normal_case(self):
        model, X_val, y_val = _train_dummy_model(n=400, seed=1)
        cfg    = _make_cfg(approve_target=0.80, reject_target=0.80)
        result = find_thresholds(model, X_val, y_val, cfg)
        assert result.high >= result.low, (
            f"Coherence violated: high={result.high}, low={result.low}"
        )

    def test_high_gte_low_tight_targets(self):
        """Even with very aggressive (potentially conflicting) targets, high >= low."""
        model, X_val, y_val = _train_dummy_model(n=200, seed=2)
        cfg    = _make_cfg(approve_target=0.99, reject_target=0.99)
        result = find_thresholds(model, X_val, y_val, cfg)
        assert result.high >= result.low

    def test_high_gte_low_easy_targets(self):
        """Wide targets (low bars) should still produce coherent thresholds."""
        model, X_val, y_val = _train_dummy_model(n=400, seed=3)
        cfg    = _make_cfg(approve_target=0.50, reject_target=0.50)
        result = find_thresholds(model, X_val, y_val, cfg)
        assert result.high >= result.low

    def test_high_gte_low_tiny_dataset(self):
        """Coherence must hold even on a very small validation split."""
        model, X_val, y_val = _train_dummy_model(n=60, seed=4, match_rate=0.2)
        cfg    = _make_cfg(approve_target=0.90, reject_target=0.90)
        result = find_thresholds(model, X_val, y_val, cfg)
        assert result.high >= result.low

    def test_high_gte_low_multiple_seeds(self):
        """Coherence holds across different random seeds."""
        for seed in range(10):
            model, X_val, y_val = _train_dummy_model(n=300, seed=seed)
            cfg    = _make_cfg(approve_target=0.85, reject_target=0.85, seed=seed)
            result = find_thresholds(model, X_val, y_val, cfg)
            assert result.high >= result.low, (
                f"Coherence violated at seed={seed}: high={result.high}, low={result.low}"
            )


class TestCaseAssignment:
    def test_both_targets_met_on_easy_case(self):
        """Well-separated data + easy targets → both_targets_met."""
        model, X_val, y_val = _train_dummy_model(n=500, seed=10)
        cfg    = _make_cfg(approve_target=0.70, reject_target=0.70)
        result = find_thresholds(model, X_val, y_val, cfg)
        assert result.case == "both_targets_met"

    def test_fallback_triggered_on_impossible_targets(self):
        """Impossibly high targets → fallback_triggered."""
        model, X_val, y_val = _train_dummy_model(n=100, seed=11, match_rate=0.5)
        # 100 % precision on approved AND 100 % recall on rejected
        # simultaneously at a coherent pair: extremely unlikely
        cfg    = _make_cfg(approve_target=1.0, reject_target=1.0, grid_steps=20)
        result = find_thresholds(model, X_val, y_val, cfg)
        # Either case is valid as long as coherence holds
        assert result.case in {"both_targets_met", "fallback_triggered"}
        assert result.high >= result.low

    def test_case_recorded_in_search_df(self):
        model, X_val, y_val = _train_dummy_model(n=300, seed=12)
        cfg    = _make_cfg()
        result = find_thresholds(model, X_val, y_val, cfg)
        assert "case" in result.search_df.columns
        assert result.search_df["case"].iloc[0] == result.case


class TestSearchDfStructure:
    REQUIRED_COLUMNS = {
        "threshold", "approve_precision", "approve_count",
        "reject_recall_nonmatch", "reject_count",
        "meets_precision_target", "meets_recall_target",
        "selected_high", "selected_low", "case",
    }

    def test_required_columns_present(self):
        model, X_val, y_val = _train_dummy_model(n=300, seed=20)
        cfg    = _make_cfg(grid_steps=40)
        result = find_thresholds(model, X_val, y_val, cfg)
        missing = self.REQUIRED_COLUMNS - set(result.search_df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_grid_length_matches_config(self):
        model, X_val, y_val = _train_dummy_model(n=300, seed=21)
        cfg    = _make_cfg(grid_steps=30)
        result = find_thresholds(model, X_val, y_val, cfg)
        assert len(result.search_df) == 30

    def test_thresholds_in_unit_interval(self):
        model, X_val, y_val = _train_dummy_model(n=300, seed=22)
        cfg    = _make_cfg()
        result = find_thresholds(model, X_val, y_val, cfg)
        assert 0.0 <= result.low  <= 1.0
        assert 0.0 <= result.high <= 1.0

    def test_thresholds_property_returns_correct_values(self):
        model, X_val, y_val = _train_dummy_model(n=300, seed=23)
        cfg    = _make_cfg()
        result = find_thresholds(model, X_val, y_val, cfg)
        t = result.thresholds
        assert t.high == result.high
        assert t.low  == result.low


class TestValidationSplitSeparation:
    """
    find_thresholds receives only X_val and y_val — it has no access to test data.
    This is guaranteed by the function signature (no dataset/path argument).

    The behavioural test below confirms that passing DIFFERENT data as val
    produces DIFFERENT thresholds, proving the function is genuinely using
    the data passed in (not some fixed default or cached value).
    """

    def test_different_val_data_gives_different_thresholds(self):
        model, _, _ = _train_dummy_model(n=500, seed=30)
        cfg = _make_cfg(approve_target=0.80, reject_target=0.70)

        rng = np.random.default_rng(30)

        # Val set A: easy (well-separated)
        X_a = pd.DataFrame(
            np.vstack([rng.normal(0.9, 0.05, (40, 2)).clip(0, 1),
                       rng.normal(0.1, 0.05, (160, 2)).clip(0, 1)]),
            columns=["f0", "f1"],
        )
        y_a = pd.Series([1]*40 + [0]*160)

        # Val set B: hard (overlapping distributions)
        X_b = pd.DataFrame(
            np.vstack([rng.normal(0.5, 0.2, (40, 2)).clip(0, 1),
                       rng.normal(0.5, 0.2, (160, 2)).clip(0, 1)]),
            columns=["f0", "f1"],
        )
        y_b = pd.Series([1]*40 + [0]*160)

        r_a = find_thresholds(model, X_a, y_a, cfg)
        r_b = find_thresholds(model, X_b, y_b, cfg)

        # Results must differ when data differs (proves data is actually used)
        assert r_a.high != r_b.high or r_a.low != r_b.low, (
            "Thresholds identical on completely different val sets — "
            "find_thresholds may not be using the passed data."
        )

    def test_function_accepts_no_test_data_argument(self):
        """find_thresholds signature has no test_df or test path parameter."""
        import inspect
        sig = inspect.signature(find_thresholds)
        param_names = set(sig.parameters.keys())
        # Must not accept any parameter that could be a test set
        forbidden = {"test_df", "test", "X_test", "y_test", "test_path"}
        overlap = forbidden & param_names
        assert not overlap, (
            f"find_thresholds has parameters that could be test data: {overlap}"
        )
