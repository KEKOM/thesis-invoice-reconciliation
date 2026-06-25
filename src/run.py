"""
Main pipeline runner.

Usage
-----
    # Full pipeline for one dataset (primary RF model)
    python -m src.run --dataset walmart_amazon
    python -m src.run --dataset dblp_acm
    python -m src.run --dataset walmart_amazon --model xgboost
    python -m src.run --dataset walmart_amazon --seed 0

    # RF vs XGBoost benchmark (identical splits, side-by-side comparison)
    python -m src.run --dataset walmart_amazon --benchmark
    python -m src.run --dataset dblp_acm --benchmark

    # Cross-dataset comparison (loads saved metrics from results/)
    python -m src.run --compare

    # End-to-end retrieval mode (one invoice in → ranked matches out)
    python -m src.run --dataset walmart_amazon  --mode retrieve
    python -m src.run --dataset dblp_acm        --mode retrieve
    python -m src.run --dataset dblp_googlescholar    --mode retrieve

    # Unified retrieval comparison table (loads retrieval_metrics.json from each dataset)
    python -m src.run --compare --mode retrieve

Output artifacts (written to results/{dataset_name}/)
------------------------------------------------------
Single-model run (default / --model):
    results.csv              — all test pairs with probability, action, label
    review_queue.csv         — flagged pairs sorted by probability descending
    feature_importance.csv   — top-15 feature importances from the base RF
    metrics.json             — all scalar metrics + threshold info
    threshold_search.csv     — full validation-split grid search table

Benchmark run (--benchmark):
    metrics_rf.json          — scalar metrics for Random Forest
    metrics_xgboost.json     — scalar metrics for XGBoost
    metrics.json             — copy of RF metrics (keeps --compare working)
    model_comparison.csv     — side-by-side RF vs XGBoost table

Retrieval run (--mode retrieve):
    retrieval_metrics.json   — match accuracy, false approval rate, review/rejection rates,
                               MRR, top-3/top-5 accuracy, mean confidence scores

Retrieval comparison (--compare --mode retrieve):
    results/retrieval_comparison.csv  — unified table across all three datasets
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import load_config, PipelineConfig
from src.evaluate import (
    compare_datasets, compare_models, compute_metrics, load_metrics, save_metrics,
)
from src.retrieve import retrieve_all, save_retrieval_metrics
from src.features import build_features
from src.loaders import load_splits
from src.model import (
    Thresholds, feature_importances,
    roc_curve_data, pr_curve_data, threshold_sweep,
    train_calibrated_model,
)
from src.thresholds import find_thresholds


# ── Core pipeline function (importable by dashboard and tests) ─────────────────

def run_pipeline(cfg: PipelineConfig, verbose: bool = True) -> dict:
    """
    Execute the full reconciliation pipeline for one dataset.

    Parameters
    ----------
    cfg : PipelineConfig
        Loaded from e.g. config/walmart_amazon.yaml.
    verbose : bool
        Print progress and summary to stdout.

    Returns
    -------
    dict with keys:
        model, names, train_df, valid_df, test_df,
        X_train, X_valid, X_test,
        threshold_result, thresholds, metrics,
        results_df, review_queue_df, importances_df
    """
    t0 = time.perf_counter()

    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    _log(f"\n{'─'*60}")
    _log(f"  Dataset : {cfg.dataset.name}")
    _log(f"  Model   : {cfg.model.type}")
    _log(f"  Seed    : {cfg.seed}")
    _log(f"{'─'*60}")

    # ── 1. Load splits ────────────────────────────────────────────────────────
    _log("Loading data …")
    train_df, valid_df, test_df = load_splits(cfg)
    _log(f"  train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")
    _log(f"  match rate: train={train_df['label'].mean():.1%}  "
         f"test={test_df['label'].mean():.1%}")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    _log("Building features …")
    X_train, names = build_features(train_df, cfg)
    X_valid, _     = build_features(valid_df, cfg)
    X_test,  _     = build_features(test_df,  cfg)
    X_valid = X_valid.reindex(columns=names, fill_value=0.0)
    X_test  = X_test.reindex(columns=names,  fill_value=0.0)
    _log(f"  {len(names)} features")

    # ── 3. Train calibrated model ─────────────────────────────────────────────
    _log(f"Training calibrated {cfg.model.type.upper()} …")
    model, base = train_calibrated_model(X_train, train_df["label"], cfg)

    # ── 4. Threshold search on VALIDATION split only ──────────────────────────
    _log("Searching thresholds on validation split …")
    threshold_result = find_thresholds(model, X_valid, valid_df["label"], cfg)
    thresholds = threshold_result.thresholds
    _log(f"  high={thresholds.high}  low={thresholds.low}  "
         f"case={threshold_result.case}")

    # ── 5. Evaluate on TEST split (one time, held-out) ────────────────────────
    _log("Evaluating on test split …")
    metrics = compute_metrics(model, X_test, test_df["label"], thresholds)

    # ── 6. Assemble result tables ─────────────────────────────────────────────
    results_df = test_df.copy().reset_index(drop=True)
    results_df["match_probability"] = metrics["proba"].round(4)
    results_df["action"]            = metrics["actions"]

    review_queue_df = (
        results_df[results_df["action"] == "flag-review"]
        .sort_values("match_probability", ascending=False)
        .reset_index(drop=True)
    )

    importances_df = feature_importances(model, names)

    # ── 7. Persist results ────────────────────────────────────────────────────
    out = Path(cfg.output.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(out / "results.csv", index=False)
    review_queue_df.to_csv(out / "review_queue.csv", index=False)
    importances_df.to_csv(out / "feature_importance.csv", index=False)
    threshold_result.search_df.to_csv(out / "threshold_search.csv", index=False)

    # Save thresholds alongside metrics
    meta = {
        **{k: v for k, v in metrics.items() if k not in {"proba", "actions", "pred"}},
        "thresholds": {
            "high": thresholds.high,
            "low":  thresholds.low,
            "case": threshold_result.case,
        },
        "model_type": cfg.model.type,
        "seed":       cfg.seed,
        "dataset":    cfg.dataset.name,
        "n_features": len(names),
        "feature_names": names,
    }
    save_metrics(meta, out / "metrics.json")

    # ── 8. Summary ────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    _log(f"\n{'─'*60}")
    _log(f"  Precision  : {metrics['precision']:.4f}")
    _log(f"  Recall     : {metrics['recall']:.4f}")
    _log(f"  F1         : {metrics['f1']:.4f}")
    _log(f"  PR-AUC     : {metrics['pr_auc']:.4f}")
    _log(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    _log(f"  Automated  : {metrics['pct_automated']} %")
    _log(f"  Review Q   : {metrics['review_queue_count']} pairs "
         f"({metrics['review_queue_pct']} %)")
    _log(f"  Missed ⚠   : {metrics['missed_matches']} matches auto-rejected "
         f"({metrics['missed_matches_pct']} %)")
    _log(f"  Elapsed    : {elapsed:.1f} s")
    _log(f"  Results    : {out}/")
    _log(f"{'─'*60}\n")

    return {
        "cfg": cfg,
        "model": model,
        "base_estimator": base,
        "names": names,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df":  test_df,
        "X_train":  X_train,
        "X_valid":  X_valid,
        "X_test":   X_test,
        "threshold_result": threshold_result,
        "thresholds":       thresholds,
        "metrics":          metrics,
        "results_df":       results_df,
        "review_queue_df":  review_queue_df,
        "importances_df":   importances_df,
    }


# ── Retrieval mode ────────────────────────────────────────────────────────────

def run_retrieval_mode(cfg: PipelineConfig, verbose: bool = True) -> dict:
    """
    Train a calibrated model and evaluate in end-to-end retrieval mode.

    Treats each unique left-side test record as an incoming invoice and scores
    it against the full right-side test pool.  The model is trained once using
    the same procedure as run_pipeline — no separate training step, no reuse
    of a pickled model.

    Only the test split is used for retrieval evaluation; the candidate pool
    consists exclusively of unique right-side test records.

    Output
    ------
    results/{dataset}/retrieval_metrics.json
        match_accuracy     — correct approvals / total approvals with ground truth
        false_approval_rate — wrong approvals  / total approvals with ground truth
        review_rate        — proportion of invoices flagged for manual review
        rejection_rate     — proportion of invoices auto-rejected

    Parameters
    ----------
    cfg : PipelineConfig
        Dataset config.  model.type selects RF or XGBoost.
    verbose : bool
        Print progress and summary to stdout.
    """
    t0 = time.perf_counter()

    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    _log(f"\n{'─' * 60}")
    _log(f"  Retrieval mode")
    _log(f"  Dataset   : {cfg.dataset.name}")
    _log(f"  Model     : {cfg.model.type}")
    _log(f"  Seed      : {cfg.seed}")
    _log(f"{'─' * 60}")

    # ── Standard pipeline steps (same model, same splits as classify mode) ────
    _log("Loading data …")
    train_df, valid_df, test_df = load_splits(cfg)
    _log(f"  train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")

    _log("Building features …")
    X_train, names = build_features(train_df, cfg)
    X_valid, _     = build_features(valid_df, cfg)
    X_valid = X_valid.reindex(columns=names, fill_value=0.0)
    _log(f"  {len(names)} features")

    _log(f"Training calibrated {cfg.model.type.upper()} …")
    model, _ = train_calibrated_model(X_train, train_df["label"], cfg)

    _log("Threshold search on validation split …")
    threshold_result = find_thresholds(model, X_valid, valid_df["label"], cfg)
    thresholds = threshold_result.thresholds
    _log(f"  high={thresholds.high}  low={thresholds.low}  case={threshold_result.case}")

    # ── Retrieval evaluation on test split ────────────────────────────────────
    left_cols  = [c for c in test_df.columns if c.startswith("left_")]
    right_cols = [c for c in test_df.columns if c.startswith("right_")]
    n_invoices = test_df[left_cols].drop_duplicates().shape[0]
    n_pool     = test_df[right_cols].drop_duplicates().shape[0]
    pool_size  = cfg.retrieval.pool_size

    _log(f"\nRunning retrieval …")
    _log(f"  {n_invoices} unique invoices  ×  "
         f"{'all' if pool_size is None else pool_size} pool records per query "
         f"(right-side pool: {n_pool} unique)")

    results, metrics = retrieve_all(
        test_df, model, thresholds, cfg, names,
        top_n=10, verbose=verbose,
    )

    # ── Persist ───────────────────────────────────────────────────────────────
    out = Path(cfg.output.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_retrieval_metrics(metrics, out / "retrieval_metrics.json")

    def _pct(v: float) -> str:
        return f"{v:.1%}" if v == v else "n/a"  # nan check

    elapsed = time.perf_counter() - t0
    _log(f"\n{'─' * 60}")
    _log(f"  Retrieval Results — {cfg.dataset.name}")
    _log(f"{'─' * 60}")
    _log(f"  Invoices evaluated  : {metrics.n_invoices}")
    _log(f"  Match accuracy      : {_pct(metrics.match_accuracy)}  "
         f"({metrics.n_correct_approved} correct auto-approvals)")
    _log(f"  False approval rate : {_pct(metrics.false_approval_rate)}  "
         f"({metrics.n_false_approved} wrong auto-approvals)")
    _log(f"  Review rate         : {_pct(metrics.review_rate)}  "
         f"({metrics.n_review} flagged)")
    _log(f"  Rejection rate      : {_pct(metrics.rejection_rate)}  "
         f"({metrics.n_rejected} rejected)")
    _log(f"  MRR                 : {metrics.mrr:.4f}" if metrics.mrr == metrics.mrr else
         f"  MRR                 : n/a")
    _log(f"  Top-3 accuracy      : {_pct(metrics.top_3_accuracy)}")
    _log(f"  Top-5 accuracy      : {_pct(metrics.top_5_accuracy)}")
    _log(f"  Avg conf (correct)  : {metrics.mean_confidence_correct:.4f}"
         if metrics.mean_confidence_correct == metrics.mean_confidence_correct else
         f"  Avg conf (correct)  : n/a")
    _log(f"  Avg conf (false app): {metrics.mean_confidence_false:.4f}"
         if metrics.mean_confidence_false == metrics.mean_confidence_false else
         f"  Avg conf (false app): n/a")
    if metrics.no_ground_truth:
        _log(f"  Skipped (≠1 match)  : {metrics.no_ground_truth} invoices")
    _log(f"  Saved → {out}/retrieval_metrics.json")
    _log(f"  Elapsed: {elapsed:.1f} s")
    _log(f"{'─' * 60}\n")

    return {"metrics": metrics, "results": results}


# ── RF vs XGBoost benchmark ──────────────────────────────────────────────────

def run_benchmark(cfg: PipelineConfig, verbose: bool = True) -> dict:
    """
    Train and evaluate both RF and XGBoost on identical train/validation/test
    splits, then print a side-by-side comparison.

    Using the same splits is essential: any difference in metrics reflects model
    behaviour, not data luck.  The calibration val-split inside
    train_calibrated_model also uses cfg.seed, so both models see the same
    calibration subset.

    Output artifacts (written to results/{dataset_name}/)
    -------------------------------------------------------
    metrics_rf.json         — scalar metrics for Random Forest
    metrics_xgboost.json    — scalar metrics for XGBoost
    metrics.json            — copy of RF metrics (keeps --compare working)
    model_comparison.csv    — side-by-side table for thesis

    Parameters
    ----------
    cfg : PipelineConfig
        Dataset config.  model.type is overridden internally — do not set it
        before calling this function.
    verbose : bool
        Print progress and comparison table to stdout.

    Returns
    -------
    dict with keys "rf", "xgboost" (scalar metric dicts) and "comparison"
    (pd.DataFrame).
    """
    t0 = time.perf_counter()

    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    _log(f"\n{'─' * 60}")
    _log(f"  Benchmark : RF vs XGBoost")
    _log(f"  Dataset   : {cfg.dataset.name}")
    _log(f"  Seed      : {cfg.seed}")
    _log(f"{'─' * 60}")

    # ── Load splits ONCE — both models see identical data ─────────────────────
    _log("Loading data …")
    train_df, valid_df, test_df = load_splits(cfg)
    _log(f"  train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")
    _log(f"  match rate: train={train_df['label'].mean():.1%}  "
         f"test={test_df['label'].mean():.1%}")

    # ── Build features ONCE ───────────────────────────────────────────────────
    _log("Building features …")
    X_train, names = build_features(train_df, cfg)
    X_valid, _     = build_features(valid_df, cfg)
    X_test,  _     = build_features(test_df,  cfg)
    X_valid = X_valid.reindex(columns=names, fill_value=0.0)
    X_test  = X_test.reindex(columns=names,  fill_value=0.0)
    _log(f"  {len(names)} features")

    out = Path(cfg.output.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    for model_type in ("rf", "xgboost"):
        _log(f"\n  ── {model_type.upper()} {'─' * (54 - len(model_type))}")

        # Deep-copy so we can safely mutate model.type without touching the
        # caller's config (also keeps random_state identical across models).
        _cfg = copy.deepcopy(cfg)
        _cfg.model.type = model_type
        _cfg.model.random_state = cfg.seed

        _log(f"  Training calibrated {model_type.upper()} …")
        model, _base = train_calibrated_model(X_train, train_df["label"], _cfg)

        _log("  Threshold search on validation split …")
        threshold_result = find_thresholds(model, X_valid, valid_df["label"], _cfg)
        thresholds = threshold_result.thresholds
        _log(f"  high={thresholds.high}  low={thresholds.low}  "
             f"case={threshold_result.case}")

        _log("  Evaluating on test split …")
        metrics = compute_metrics(model, X_test, test_df["label"], thresholds)

        meta = {
            **{k: v for k, v in metrics.items() if k not in {"proba", "actions", "pred"}},
            "thresholds": {
                "high": thresholds.high,
                "low":  thresholds.low,
                "case": threshold_result.case,
            },
            "model_type":    model_type,
            "seed":          cfg.seed,
            "dataset":       cfg.dataset.name,
            "n_features":    len(names),
            "feature_names": names,
        }
        save_metrics(meta, out / f"metrics_{model_type}.json")
        results[model_type] = meta

        _log(f"  Precision={metrics['precision']:.4f}  "
             f"Recall={metrics['recall']:.4f}  "
             f"F1={metrics['f1']:.4f}")
        _log(f"  PR-AUC={metrics['pr_auc']:.4f}  "
             f"Review-Q={metrics['review_queue_count']} "
             f"({metrics['review_queue_pct']} %)")

    # RF is the primary model — keep metrics.json pointing at it so --compare works
    save_metrics(results["rf"], out / "metrics.json")

    # ── Side-by-side comparison table ────────────────────────────────────────
    comparison = compare_models(results["rf"], results["xgboost"], cfg.dataset.name)
    comparison.to_csv(out / "model_comparison.csv")

    elapsed = time.perf_counter() - t0
    _log(f"\n{'═' * 60}")
    _log(f"  RF vs XGBoost — {cfg.dataset.name}")
    _log(f"{'═' * 60}")
    _log(comparison.to_string())
    _log(f"{'═' * 60}")
    _log(f"\n  Saved → {out}/metrics_rf.json")
    _log(f"  Saved → {out}/metrics_xgboost.json")
    _log(f"  Saved → {out}/model_comparison.csv")
    _log(f"  Elapsed: {elapsed:.1f} s\n")

    return {
        "rf":         results["rf"],
        "xgboost":    results["xgboost"],
        "comparison": comparison,
    }


# ── Cross-dataset comparison ──────────────────────────────────────────────────

_DEFAULT_DATASETS = ["walmart_amazon", "dblp_acm", "dblp_googlescholar"]


def print_comparison(dataset_names: list[str] | None = None) -> pd.DataFrame:
    """
    Load saved classification metrics for each dataset and print a comparison table.

    Parameters
    ----------
    dataset_names : list[str], optional
        Defaults to all three benchmark datasets.
    """
    if dataset_names is None:
        dataset_names = _DEFAULT_DATASETS

    loaded: dict[str, dict] = {}
    for name in dataset_names:
        path = Path(f"results/{name}/metrics.json")
        if not path.exists():
            print(f"  ⚠  {path} not found — run pipeline first: "
                  f"python -m src.run --dataset {name}")
            continue
        loaded[name] = load_metrics(path)

    if not loaded:
        print("No results to compare.")
        return pd.DataFrame()

    df = compare_datasets(loaded)
    print("\n" + "═" * 70)
    print("  Cross-dataset comparison")
    print("═" * 70)
    print(df.to_string())
    print("═" * 70 + "\n")

    out = Path("results/comparison.csv")
    df.to_csv(out)
    print(f"Saved → {out}")
    return df


def print_retrieval_comparison(dataset_names: list[str] | None = None) -> pd.DataFrame:
    """
    Load saved retrieval_metrics.json for each dataset and print a comparison table.

    Saves the unified table to results/retrieval_comparison.csv.

    Parameters
    ----------
    dataset_names : list[str], optional
        Defaults to all three benchmark datasets.
    """
    if dataset_names is None:
        dataset_names = _DEFAULT_DATASETS

    rows = []
    for name in dataset_names:
        path = Path(f"results/{name}/retrieval_metrics.json")
        if not path.exists():
            print(f"  ⚠  {path} not found — run retrieval first: "
                  f"python -m src.run --dataset {name} --mode retrieve")
            continue
        with open(path) as fh:
            m = json.load(fh)
        rows.append({
            "dataset":                  name,
            "n_invoices":               m.get("n_invoices"),
            "match_accuracy":           m.get("match_accuracy"),
            "false_approval_rate":      m.get("false_approval_rate"),
            "review_rate":              m.get("review_rate"),
            "rejection_rate":           m.get("rejection_rate"),
            "top_3_accuracy":           m.get("top_3_accuracy"),
            "top_5_accuracy":           m.get("top_5_accuracy"),
            "mrr":                      m.get("mrr"),
            "mean_confidence_correct":  m.get("mean_confidence_correct"),
            "mean_confidence_false":    m.get("mean_confidence_false"),
        })

    if not rows:
        print("No retrieval results to compare.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("dataset")

    print("\n" + "═" * 80)
    print("  Retrieval evaluation — cross-dataset comparison")
    print("═" * 80)
    print(df.to_string())
    print("═" * 80 + "\n")

    out = Path("results/retrieval_comparison.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"Saved → {out}")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.run",
        description="Invoice reconciliation pipeline — train, evaluate, and compare.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--dataset", metavar="NAME",
        help="Dataset to run (e.g. walmart_amazon, dblp_acm). "
             "Config is loaded from config/{NAME}.yaml.",
    )
    g.add_argument(
        "--compare", action="store_true",
        help="Load saved results and print the cross-dataset comparison table.",
    )
    p.add_argument(
        "--benchmark", action="store_true",
        help="Train and evaluate BOTH RF and XGBoost on identical splits and print "
             "a side-by-side comparison.  Requires --dataset.  Saves "
             "metrics_rf.json, metrics_xgboost.json, and model_comparison.csv.",
    )
    p.add_argument(
        "--mode", choices=["classify", "retrieve"], default="classify",
        help="'classify' (default): standard classification pipeline.  "
             "'retrieve': end-to-end retrieval — score each test invoice against "
             "the full right-side pool and report match accuracy, false approval "
             "rate, review rate, and rejection rate.",
    )
    p.add_argument(
        "--model", choices=["rf", "xgboost"], default=None,
        help="Override cfg.model.type (ignored when --benchmark is set).",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Override cfg.seed (affects model init, train/val split, threshold grid).",
    )
    p.add_argument(
        "--config-dir", default="config", metavar="DIR",
        help="Directory containing dataset YAML configs (default: config/).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.compare:
        if args.mode == "retrieve":
            print_retrieval_comparison()
        else:
            print_comparison()
        return

    config_path = Path(args.config_dir) / f"{args.dataset}.yaml"
    cfg = load_config(config_path)

    # CLI overrides
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.model.random_state = args.seed

    if args.benchmark:
        run_benchmark(cfg)
        return

    if args.mode == "retrieve":
        run_retrieval_mode(cfg)
        return

    if args.model is not None:
        cfg.model.type = args.model

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
