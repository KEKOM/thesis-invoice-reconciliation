"""
Pipeline configuration loader.

Each dataset has its own YAML file under config/ (e.g. config/walmart_amazon.yaml).
Call load_config(path) to get a typed PipelineConfig object that every pipeline
stage accepts as its single settings argument.

Design notes
------------
* Nested dataclasses for each concern (features, model, thresholds, …) keep
  the interface self-documenting and make it easy to pass only the relevant
  slice to each module.
* Required fields raise ConfigError immediately on load — fail fast rather than
  producing a cryptic error mid-pipeline.
* All numeric randomness (model, threshold grid) is seeded from the top-level
  ``seed`` field so a single number makes the full run reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


class ConfigError(ValueError):
    """Raised when a required config key is missing or has an invalid value."""


# ── Sub-configs ───────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    name:  str
    train: str
    valid: str
    test:  str


@dataclass
class FeatureConfig:
    text_fields:      list[str]
    numeric_fields:   list[str]
    amount_fields:    list[str]
    expected_margins: list[float]
    margin_tolerance: float = 0.005


@dataclass
class BlockingConfig:
    vendor_key_fields:   list[str] = field(default_factory=lambda: ["vendor", "title", "name", "supplier"])
    amount_key_fields:   list[str] = field(default_factory=lambda: ["price", "total", "amount", "unit_price"])
    year_key_fields:     list[str] = field(default_factory=list)
    amount_bucket_width: float = 10.0


@dataclass
class ModelConfig:
    type:          str   = "rf"     # "rf" | "xgboost"
    n_estimators:  int   = 300
    max_depth:     Optional[int] = None
    class_weight:  str   = "balanced"
    random_state:  int   = 42       # overridden by top-level seed at load time
    # XGBoost-specific (ignored when type == "rf")
    xgb_learning_rate: float = 0.1
    xgb_max_depth:     int   = 6
    xgb_subsample:     float = 0.8


@dataclass
class CalibrationConfig:
    val_fraction:         float = 0.20
    method:               str   = "auto"   # "auto" | "isotonic" | "sigmoid"
    min_isotonic_samples: int   = 1000


@dataclass
class ThresholdConfig:
    strategy:                 str   = "precision_target"
    approve_precision_target: float = 0.90
    reject_recall_target:     float = 0.90
    grid_steps:               int   = 100


@dataclass
class OutputConfig:
    results_dir: str = "results"


@dataclass
class RetrievalConfig:
    pool_size: int | None = None   # None = all available right-side test records


# ── Top-level config ──────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    dataset:     DatasetConfig
    features:    FeatureConfig
    blocking:    BlockingConfig
    model:       ModelConfig
    calibration: CalibrationConfig
    thresholds:  ThresholdConfig
    output:      OutputConfig
    retrieval:   RetrievalConfig = field(default_factory=RetrievalConfig)
    seed:        int = 42


# ── Loader ────────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> PipelineConfig:
    """
    Load and validate a pipeline config YAML.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file (e.g. "config/walmart_amazon.yaml").

    Returns
    -------
    PipelineConfig
        Fully typed configuration object.

    Raises
    ------
    ConfigError
        If a required section or field is missing or has an invalid value.
    FileNotFoundError
        If the YAML file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as fh:
        raw: dict = yaml.safe_load(fh) or {}

    seed: int = int(raw.get("seed", 42))

    # ── dataset ──────────────────────────────────────────────────────────────
    ds_raw = _require_section(raw, "dataset", path)
    dataset = DatasetConfig(
        name  = _require_str(ds_raw, "name",  "dataset.name",  path),
        train = _require_str(ds_raw, "train", "dataset.train", path),
        valid = _require_str(ds_raw, "valid", "dataset.valid", path),
        test  = _require_str(ds_raw, "test",  "dataset.test",  path),
    )

    # ── features ─────────────────────────────────────────────────────────────
    ft_raw = _require_section(raw, "features", path)
    features = FeatureConfig(
        text_fields      = list(ft_raw.get("text_fields",      [])),
        numeric_fields   = list(ft_raw.get("numeric_fields",   [])),
        amount_fields    = list(ft_raw.get("amount_fields",    [])),
        expected_margins = [float(m) for m in ft_raw.get("expected_margins", [])],
        margin_tolerance = float(ft_raw.get("margin_tolerance", 0.005)),
    )

    # ── blocking ─────────────────────────────────────────────────────────────
    bl_raw = raw.get("blocking", {})
    blocking = BlockingConfig(
        vendor_key_fields   = list(bl_raw.get("vendor_key_fields",   ["vendor", "title", "name", "supplier"])),
        amount_key_fields   = list(bl_raw.get("amount_key_fields",   ["price", "total", "amount", "unit_price"])),
        year_key_fields     = list(bl_raw.get("year_key_fields",     [])),
        amount_bucket_width = float(bl_raw.get("amount_bucket_width", 10.0)),
    )

    # ── model ─────────────────────────────────────────────────────────────────
    mo_raw = raw.get("model", {})
    model_type = str(mo_raw.get("type", "rf")).lower()
    if model_type not in {"rf", "xgboost"}:
        raise ConfigError(f"[{path}] model.type must be 'rf' or 'xgboost', got {model_type!r}")
    model = ModelConfig(
        type          = model_type,
        n_estimators  = int(mo_raw.get("n_estimators", 300)),
        max_depth     = None if mo_raw.get("max_depth") is None else int(mo_raw["max_depth"]),
        class_weight  = str(mo_raw.get("class_weight", "balanced")),
        random_state  = seed,    # always propagate top-level seed
        xgb_learning_rate = float(mo_raw.get("xgb_learning_rate", 0.1)),
        xgb_max_depth     = int(mo_raw.get("xgb_max_depth", 6)),
        xgb_subsample     = float(mo_raw.get("xgb_subsample", 0.8)),
    )

    # ── calibration ──────────────────────────────────────────────────────────
    ca_raw = raw.get("calibration", {})
    cal_method = str(ca_raw.get("method", "auto")).lower()
    if cal_method not in {"auto", "isotonic", "sigmoid"}:
        raise ConfigError(
            f"[{path}] calibration.method must be 'auto', 'isotonic', or 'sigmoid', got {cal_method!r}"
        )
    calibration = CalibrationConfig(
        val_fraction         = float(ca_raw.get("val_fraction",         0.20)),
        method               = cal_method,
        min_isotonic_samples = int(ca_raw.get("min_isotonic_samples",   1000)),
    )

    # ── thresholds ────────────────────────────────────────────────────────────
    th_raw = raw.get("thresholds", {})
    th_strategy = str(th_raw.get("strategy", "precision_target")).lower()
    if th_strategy not in {"precision_target"}:
        raise ConfigError(
            f"[{path}] thresholds.strategy must be 'precision_target', got {th_strategy!r}"
        )
    thresholds = ThresholdConfig(
        strategy                 = th_strategy,
        approve_precision_target = float(th_raw.get("approve_precision_target", 0.90)),
        reject_recall_target     = float(th_raw.get("reject_recall_target",     0.90)),
        grid_steps               = int(th_raw.get("grid_steps", 100)),
    )

    # ── output ────────────────────────────────────────────────────────────────
    ou_raw = raw.get("output", {})
    output = OutputConfig(
        results_dir = str(ou_raw.get("results_dir", f"results/{dataset.name}")),
    )

    # ── retrieval ─────────────────────────────────────────────────────────────
    re_raw = raw.get("retrieval", {})
    retrieval = RetrievalConfig(
        pool_size = None if re_raw.get("pool_size") is None else int(re_raw["pool_size"]),
    )

    return PipelineConfig(
        dataset=dataset,
        features=features,
        blocking=blocking,
        model=model,
        calibration=calibration,
        thresholds=thresholds,
        output=output,
        retrieval=retrieval,
        seed=seed,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _require_section(raw: dict, key: str, path: Path) -> dict:
    if key not in raw:
        raise ConfigError(f"[{path}] Required section '{key}' is missing.")
    if not isinstance(raw[key], dict):
        raise ConfigError(f"[{path}] Section '{key}' must be a mapping, got {type(raw[key]).__name__}.")
    return raw[key]


def _require_str(section: dict, key: str, full_key: str, path: Path) -> str:
    if key not in section or section[key] is None:
        raise ConfigError(f"[{path}] Required field '{full_key}' is missing.")
    return str(section[key])
