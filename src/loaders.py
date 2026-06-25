"""
Dataset loaders.

Wraps the low-level ER-Magellan / CSV parser (src.load) with dataset-aware
path resolution, missing-file hints, and a single callable entry point for
the rest of the pipeline.

Usage
-----
    from src.config import load_config
    from src.loaders import load_splits

    cfg = load_config("config/walmart_amazon.yaml")
    train_df, valid_df, test_df = load_splits(cfg)

Each returned DataFrame has left_* / right_* columns (one per shared field)
plus a ``label`` column (1 = match, 0 = non-match).  This layout is the same
regardless of which dataset is loaded.

Separation guarantee
--------------------
load_splits returns three completely independent DataFrames.  The pipeline
must NEVER use test_df for anything except the final held-out evaluation:
  - feature engineering parameters are derived from train_df only
  - thresholds are calibrated on valid_df only
  - test_df is touched exactly once, in src.evaluate.compute_metrics
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PipelineConfig
from src.load import load_csv, load_ermagellan


def load_splits(
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, valid, and test splits for the dataset described by *cfg*.

    Parameters
    ----------
    cfg : PipelineConfig
        Must have dataset.train / dataset.valid / dataset.test pointing at
        valid file paths (ER-Magellan .txt or generic left_/right_/label .csv).

    Returns
    -------
    (train_df, valid_df, test_df) : tuple of DataFrames

    Raises
    ------
    FileNotFoundError
        If a split file is missing.  A download hint is appended to the message
        (reads DOWNLOAD.md if present in the same directory).
    """
    train = _load_one(cfg.dataset.train, cfg.dataset.name)
    valid = _load_one(cfg.dataset.valid, cfg.dataset.name)
    test  = _load_one(cfg.dataset.test,  cfg.dataset.name)
    return train, valid, test


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_one(path: str, dataset_name: str = "") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n{_download_hint(dataset_name, p.parent)}"
        )
    return load_ermagellan(path) if p.suffix == ".txt" else load_csv(path)


def _download_hint(dataset_name: str, data_dir: Path) -> str:
    dl = data_dir / "DOWNLOAD.md"
    if dl.exists():
        return f"See {dl} for download instructions."
    _hints: dict[str, str] = {
        "dblp_acm":       "Download from https://github.com/anhaidgroup/deepmatcher → Datasets.md",
        "walmart_amazon": "Data should be in data/walmart_amazon/ — see README.md.",
    }
    return _hints.get(dataset_name, "Check README.md for dataset setup instructions.")
