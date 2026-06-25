"""
Invoice Reconciliation Dashboard.

Run:
    streamlit run dashboard/app.py

Tabs
----
1. Overview          — run summary, decision-split donut, probability histogram
2. Metrics           — all metrics, confusion matrix, ROC/PR curves, threshold sensitivity
3. Review Queue      — flagged pairs sorted by probability, CSV download
4. Feature Importance— RF feature importances (horizontal bar)
5. Data Upload       — upload a CSV of candidate pairs and score with the current model

Usage notes
-----------
* Select a dataset in the sidebar.  If results are already saved under
  results/{dataset}/, they load instantly.
* Drag the threshold sliders to re-route pairs live (no retraining).
* Click "Re-run pipeline" to retrain from scratch using the data files.
* Tab 5 accepts any CSV with left_* / right_* columns (or plain columns
  that will be auto-prefixed) and scores them with the cached model.
"""
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure repo root is on path when launched from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, PipelineConfig
from src.evaluate import compute_metrics, load_metrics
from src.features import build_features
from src.loaders import load_splits
from src.model import (
    Thresholds, decide, feature_importances,
    pr_curve_data, roc_curve_data, threshold_sweep,
    train_calibrated_model,
)
from src.retrieve import retrieve
from src.run import run_pipeline, print_retrieval_comparison
from src.thresholds import find_thresholds

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Invoice Reconciliation",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

_PL = dict(
    template="plotly_white",
    font=dict(family="Inter, ui-sans-serif, sans-serif", size=13, color="#0f172a"),
    margin=dict(l=16, r=16, t=36, b=16),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
C = dict(approve="#16a34a", flag="#d97706", reject="#dc2626",
         primary="#2563EB", purple="#7c3aed", muted="#94a3b8")

DATASET_LABELS = {
    "walmart_amazon": "Walmart-Amazon",
    "dblp_acm":       "DBLP-ACM",
    "dblp_googlescholar":   "DBLP-GoogleScholar",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _results_exist(cfg: PipelineConfig) -> bool:
    return (Path(cfg.output.results_dir) / "metrics.json").exists()


def _load_saved_thresholds(cfg: PipelineConfig) -> tuple[float, float]:
    path = Path(cfg.output.results_dir) / "metrics.json"
    if path.exists():
        m = load_metrics(path)
        t = m.get("thresholds", {})
        return float(t.get("high", 0.80)), float(t.get("low", 0.10))
    return 0.80, 0.10


def _metric_card(col, label: str, value: str, delta: str = "") -> None:
    col.metric(label, value, delta or None)


# ── Cached model + data loading ───────────────────────────────────────────────

@st.cache_resource(show_spinner="Training calibrated model…")
def _get_model(dataset_name: str, config_path: str):
    """Train and cache the model for a given dataset (keyed by path)."""
    cfg = load_config(config_path)
    train_df, valid_df, test_df = load_splits(cfg)
    X_train, names = build_features(train_df, cfg)
    X_valid, _     = build_features(valid_df, cfg)
    X_test,  _     = build_features(test_df,  cfg)
    X_valid = X_valid.reindex(columns=names, fill_value=0.0)
    X_test  = X_test.reindex(columns=names,  fill_value=0.0)
    model, _       = train_calibrated_model(X_train, train_df["label"], cfg)
    roc  = roc_curve_data(model, X_test, test_df["label"])
    pr   = pr_curve_data(model, X_test, test_df["label"])
    sweep = threshold_sweep(model, X_test, test_df["label"])
    return model, names, train_df, valid_df, test_df, X_valid, X_test, roc, pr, sweep, cfg


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Invoice Reconciliation")
    st.caption("Calibrated Random Forest · confidence-based exception flagging")
    st.divider()

    dataset_key = st.selectbox(
        "Dataset",
        options=list(DATASET_LABELS.keys()),
        format_func=lambda k: DATASET_LABELS[k],
    )
    config_path = f"config/{dataset_key}.yaml"

    try:
        cfg_sidebar = load_config(config_path)
    except FileNotFoundError:
        st.error(f"Config not found: {config_path}")
        st.stop()

    st.divider()
    st.markdown("**Confidence thresholds**")
    st.caption("Drag to re-route pairs live.  Model is cached — no retraining.")
    _def_high, _def_low = _load_saved_thresholds(cfg_sidebar)
    # Clamp defaults to slider bounds — saved thresholds can be as low as 0.01
    # (e.g. DBLP-ACM where the model is very confident on everything).
    _def_high = float(np.clip(_def_high, 0.01, 1.00))
    _def_low  = float(np.clip(_def_low,  0.00, 0.99))
    high = st.slider("Auto-approve ≥", 0.01, 1.00, _def_high, 0.01,
                     help="p(match) ≥ this → auto-approve")
    low  = st.slider("Auto-reject  ≤", 0.00, 0.99, _def_low,  0.01,
                     help="p(match) ≤ this → auto-reject")

    if high <= low:
        st.warning("Auto-approve threshold must be > auto-reject threshold.")

    st.divider()
    if st.button("🔄 Re-run pipeline", use_container_width=True):
        with st.spinner("Running full pipeline…"):
            try:
                _get_model.clear()
                run_pipeline(cfg_sidebar, verbose=False)
                st.success("Pipeline complete — reload the page to see updated results.")
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")

    st.caption("Change dataset or click Re-run to retrain.")


# ── Load model (training may happen here the first time) ─────────────────────
try:
    (model, names, train_df, valid_df, test_df,
     X_valid, X_test, roc, prc, sweep, cfg) = _get_model(dataset_key, config_path)
except FileNotFoundError as exc:
    st.error(f"**Data not found:** {exc}")
    st.info("See the dataset's DOWNLOAD.md for setup instructions.")
    st.stop()
except Exception as exc:
    st.error(f"**Pipeline error:** {exc}")
    st.stop()

t       = Thresholds(high=high, low=low)
metrics = compute_metrics(model, X_test, test_df["label"], t)
cm      = metrics["confusion"]
n       = metrics["n_pairs"]

results_df = test_df.copy().reset_index(drop=True)
results_df["match_probability"] = metrics["proba"].round(4)
results_df["action"]            = metrics["actions"]

review_queue = (
    results_df[results_df["action"] == "flag-review"]
    .sort_values("match_probability", ascending=False)
    .reset_index(drop=True)
)
importances = feature_importances(model, names)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Overview", "Metrics", "Review Queue", "Feature Importance",
     "Data Upload", "Retrieval", "Retrieval Evaluation"]
)

# ══ Tab 1 — Overview ══════════════════════════════════════════════════════════
with tab1:
    label_name = DATASET_LABELS[dataset_key]
    st.subheader(f"{label_name} — pipeline summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    _metric_card(c1, "F1 score",    f"{metrics['f1']:.4f}")
    _metric_card(c2, "PR-AUC",      f"{metrics['pr_auc']:.4f}")
    _metric_card(c3, "ROC-AUC",     f"{metrics['roc_auc']:.4f}")
    _metric_card(c4, "% Automated", f"{metrics['pct_automated']} %")
    _metric_card(c5, "Review Queue",
                 f"{metrics['review_queue_count']}",
                 f"{metrics['review_queue_pct']} % of test pairs")

    st.divider()

    # Missed-matches warning
    if metrics["missed_matches"] > 0:
        st.warning(
            f"⚠ **{metrics['missed_matches']} missed match(es)** "
            f"({metrics['missed_matches_pct']} % of all true matches) are in the "
            f"auto-reject zone with the current thresholds.  "
            f"Raise the auto-reject threshold to reduce this.",
            icon=None,
        )

    col_donut, col_hist = st.columns([1, 2])
    with col_donut:
        st.markdown("**Decision split**")
        fig_donut = go.Figure(go.Pie(
            labels=["Auto-approve", "Flag-review", "Auto-reject"],
            values=[metrics["auto_approve"], metrics["flag_review"], metrics["auto_reject"]],
            marker_colors=[C["approve"], C["flag"], C["reject"]],
            hole=0.5,
            textinfo="percent+label",
        ))
        fig_donut.update_layout(**_PL, showlegend=False, height=300)
        st.plotly_chart(fig_donut, width="stretch")

    with col_hist:
        st.markdown("**Match-probability distribution**")
        fig_hist = px.histogram(
            results_df, x="match_probability", color="action",
            nbins=40, barmode="stack",
            color_discrete_map={
                "auto-approve": C["approve"],
                "flag-review":  C["flag"],
                "auto-reject":  C["reject"],
            },
            labels={"match_probability": "Calibrated p(match)", "action": "Decision"},
        )
        fig_hist.add_vline(x=high, line_dash="dash", line_color=C["approve"],
                           annotation_text="approve", annotation_position="top right")
        fig_hist.add_vline(x=low,  line_dash="dash", line_color=C["reject"],
                           annotation_text="reject",  annotation_position="top left")
        fig_hist.update_layout(**_PL, height=300)
        st.plotly_chart(fig_hist, width="stretch")

    st.divider()
    c_tr, c_val, c_te = st.columns(3)
    c_tr.metric("Train pairs", f"{len(train_df):,}",
                f"{train_df['label'].mean():.1%} match rate")
    c_val.metric("Valid pairs", f"{len(valid_df):,}")
    c_te.metric("Test pairs",  f"{len(test_df):,}",
                f"{test_df['label'].mean():.1%} match rate")


# ══ Tab 2 — Metrics ═══════════════════════════════════════════════════════════
with tab2:
    st.subheader("Detailed metrics")

    c1, c2, c3 = st.columns(3)
    _metric_card(c1, "Precision", f"{metrics['precision']:.4f}")
    _metric_card(c2, "Recall",    f"{metrics['recall']:.4f}")
    _metric_card(c3, "F1",        f"{metrics['f1']:.4f}")
    c4, c5, c6 = st.columns(3)
    _metric_card(c4, "PR-AUC",   f"{metrics['pr_auc']:.4f}")
    _metric_card(c5, "ROC-AUC",  f"{metrics['roc_auc']:.4f}")
    _metric_card(c6, "Accuracy", f"{metrics['accuracy']:.4f}",
                 "(misleading on imbalanced data — see PR-AUC)")
    st.divider()

    col_cm, col_roc, col_pr = st.columns(3)

    with col_cm:
        st.markdown("**Confusion matrix**")
        cm_data = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        fig_cm = px.imshow(
            cm_data,
            text_auto=True,
            color_continuous_scale="Blues",
            x=["Pred 0", "Pred 1"],
            y=["True 0", "True 1"],
            labels=dict(x="Predicted", y="Actual"),
        )
        fig_cm.update_layout(**_PL, height=260, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, width="stretch")

    with col_roc:
        st.markdown(f"**ROC curve** (AUC = {roc['auc']:.4f})")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"], mode="lines",
            line=dict(color=C["primary"], width=2), name="ROC",
        ))
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color=C["muted"]))
        fig_roc.update_layout(**_PL, height=260,
                              xaxis_title="FPR", yaxis_title="TPR", showlegend=False)
        st.plotly_chart(fig_roc, width="stretch")

    with col_pr:
        st.markdown(f"**PR curve** (AUC = {prc['auc']:.4f})")
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=prc["recall"], y=prc["precision"], mode="lines",
            line=dict(color=C["purple"], width=2), name="PR",
        ))
        fig_pr.update_layout(**_PL, height=260,
                             xaxis_title="Recall", yaxis_title="Precision", showlegend=False)
        st.plotly_chart(fig_pr, width="stretch")

    st.divider()
    col_tsens, col_rsens = st.columns(2)

    with col_tsens:
        st.markdown("**Threshold sensitivity** (classification at p ≥ t)")
        fig_ts = go.Figure()
        for col, label, color in [
            ("precision", "Precision", C["approve"]),
            ("recall",    "Recall",    C["reject"]),
            ("f1",        "F1",        C["primary"]),
        ]:
            fig_ts.add_trace(go.Scatter(
                x=sweep["threshold"], y=sweep[col], mode="lines",
                line=dict(color=color), name=label,
            ))
        fig_ts.add_vline(x=high, line_dash="dash", line_color=C["approve"])
        fig_ts.update_layout(**_PL, height=260,
                             xaxis_title="Threshold", yaxis_title="Score")
        st.plotly_chart(fig_ts, width="stretch")

    with col_rsens:
        st.markdown("**Routing sensitivity** (pair counts vs approve threshold)")
        ths = np.linspace(0.01, 0.99, 80)
        p   = metrics["proba"]
        rows = [
            {
                "threshold": round(float(t_), 3),
                "auto_approve": int((p >= t_).sum()),
                "flag_review":  int(((p > low) & (p < t_)).sum()),
                "auto_reject":  int((p <= low).sum()),
            }
            for t_ in ths
        ]
        rsens = pd.DataFrame(rows)
        fig_rs = go.Figure()
        for col, label, color in [
            ("auto_approve", "Auto-approve", C["approve"]),
            ("flag_review",  "Flag-review",  C["flag"]),
            ("auto_reject",  "Auto-reject",  C["reject"]),
        ]:
            fig_rs.add_trace(go.Scatter(
                x=rsens["threshold"], y=rsens[col], mode="lines",
                line=dict(color=color), name=label,
            ))
        fig_rs.add_vline(x=high, line_dash="dash", line_color=C["approve"])
        fig_rs.update_layout(**_PL, height=260,
                             xaxis_title="Approve threshold", yaxis_title="Count")
        st.plotly_chart(fig_rs, width="stretch")


# ══ Tab 3 — Review Queue ══════════════════════════════════════════════════════
with tab3:
    st.subheader(f"Review queue  —  {len(review_queue)} pair(s) flagged")
    st.caption(
        "Pairs where the model is uncertain.  Sorted by confidence descending "
        "(highest-confidence uncertain first)."
    )

    if review_queue.empty:
        st.info("No pairs in the review queue at the current thresholds.")
    else:
        st.dataframe(review_queue, height=500, use_container_width=True)

        csv_bytes = review_queue.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download review queue CSV",
            data=csv_bytes,
            file_name=f"review_queue_{dataset_key}.csv",
            mime="text/csv",
        )


# ══ Tab 4 — Feature Importance ════════════════════════════════════════════════
with tab4:
    st.subheader("Feature importance (Random Forest)")
    st.caption(
        "Importances from the base Random Forest, unwrapped from the calibration "
        "wrapper.  Top-15 features shown."
    )
    fig_fi = px.bar(
        importances.sort_values("importance"),
        x="importance", y="feature", orientation="h",
        color="importance",
        color_continuous_scale=[[0, C["muted"]], [1, C["primary"]]],
        labels={"importance": "Importance", "feature": ""},
    )
    fig_fi.update_layout(**_PL, height=420, coloraxis_showscale=False)
    st.plotly_chart(fig_fi, width="stretch")

    with st.expander("Full importances table"):
        all_imp = feature_importances(model, names, top=len(names))
        st.dataframe(all_imp, use_container_width=True)


# ══ Tab 5 — Data Upload ═══════════════════════════════════════════════════════
with tab5:
    st.subheader("Score new candidate pairs")
    st.markdown(
        "Upload a CSV of candidate pairs.  Columns should be `left_*` and "
        "`right_*` (or plain names — they'll be auto-prefixed as `left_`).\n\n"
        "The current cached model for **" + DATASET_LABELS[dataset_key] + "** "
        "will score them."
    )

    uploaded = st.file_uploader(
        "Candidate pairs CSV", type="csv", key="upload_csv",
        help="Each row is one (left record, right record) pair.",
    )

    if uploaded is not None:
        try:
            pairs_df = pd.read_csv(uploaded)
            st.caption(f"Uploaded: {len(pairs_df)} rows · {pairs_df.shape[1]} columns")

            # Auto-prefix if needed
            if not any(c.startswith("left_") or c.startswith("right_") for c in pairs_df.columns):
                st.info(
                    "No `left_*` / `right_*` columns detected.  Treating ALL "
                    "columns as left-side fields (reference side will have no "
                    "matching columns — similarity scores may be 0)."
                )
                pairs_df = pairs_df.rename(columns={c: f"left_{c}" for c in pairs_df.columns})

            with st.spinner("Extracting features and scoring…"):
                X_up, _ = build_features(pairs_df, cfg)
                X_up    = X_up.reindex(columns=names, fill_value=0.0)
                proba_up = model.predict_proba(X_up)[:, 1]
                actions_up = decide(proba_up, t)

            out_df = pairs_df.copy()
            out_df["match_probability"] = proba_up.round(4)
            out_df["action"]            = actions_up

            n_app = int((actions_up == "auto-approve").sum())
            n_rev = int((actions_up == "flag-review").sum())
            n_rej = int((actions_up == "auto-reject").sum())

            c1u, c2u, c3u = st.columns(3)
            _metric_card(c1u, "Auto-approved", str(n_app))
            _metric_card(c2u, "Flag-review",   str(n_rev))
            _metric_card(c3u, "Auto-rejected", str(n_rej))

            st.dataframe(out_df.sort_values("match_probability", ascending=False),
                         height=400, use_container_width=True)

            st.download_button(
                "⬇ Download scored CSV",
                data=out_df.to_csv(index=False).encode(),
                file_name="scored_pairs.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"Scoring failed: {exc}")
    else:
        st.markdown(
            "**Expected column format (example)**\n\n"
            "| left_title | left_price | right_title | right_price |\n"
            "|------------|------------|-------------|-------------|\n"
            "| Acme Widget | 99.99 | Acme Widget v2 | 95.00 |\n"
        )


# ══ Tab 6 — Retrieval ═════════════════════════════════════════════════════════
with tab6:
    st.subheader("Retrieval — live record search")
    st.caption(
        "Select an invoice from the test set (or paste values below). "
        "The model scores it against every record in the right-side pool "
        "and applies the current threshold sliders to decide."
    )

    _left_cols  = [c for c in test_df.columns if c.startswith("left_")]
    _right_cols = [c for c in test_df.columns if c.startswith("right_")]

    _invoices = test_df[_left_cols].drop_duplicates().reset_index(drop=True)
    _pool     = test_df[_right_cols].drop_duplicates().reset_index(drop=True)

    # Pick a display field for the dropdown label
    _label_col = next(
        (f for f in ["left_title", "left_name", "left_description"] if f in _invoices.columns),
        _left_cols[0] if _left_cols else None,
    )

    def _invoice_label(i: int) -> str:
        if _label_col:
            return f"#{i}  {str(_invoices.iloc[i][_label_col])[:70]}"
        return f"#{i}"

    col_sel, col_info = st.columns([3, 1])
    with col_sel:
        _sel = st.selectbox(
            "Invoice (left-side test record)",
            options=range(len(_invoices)),
            format_func=_invoice_label,
        )
    col_info.metric("Pool size", f"{len(_pool)} records")

    _invoice = _invoices.iloc[_sel]

    with st.expander("Invoice fields", expanded=False):
        st.dataframe(_invoice.to_frame().T.reset_index(drop=True), use_container_width=True)

    with st.spinner("Scoring candidates …"):
        _result = retrieve(_invoice, _pool, model, t, cfg, names, top_n=10)

    # ── Decision banner ───────────────────────────────────────────────────────
    st.divider()
    _dc, _sc, _nc = st.columns(3)
    _dc.metric("Decision", _result.decision)
    _sc.metric("Top score", f"{_result.top_score:.4f}")
    _nc.metric("Candidates scored", _result.n_candidates)

    if _result.decision == "auto-approve":
        st.success(
            f"Automatically approved — top candidate cleared the approve "
            f"threshold ({t.high}).",
        )
    elif _result.decision == "flag-review":
        st.warning(
            "Flagged for manual review — top score sits in the uncertain zone "
            f"({t.low} < p < {t.high}).  Inspect the ranked list below.",
        )
    else:
        st.error(
            f"Automatically rejected — no candidate exceeded the reject "
            f"threshold ({t.low}).",
        )

    # ── Ranked candidate list ─────────────────────────────────────────────────
    st.divider()
    if _result.candidates:
        st.markdown(f"**Ranked candidates** (top {len(_result.candidates)})")
        _cand_df = pd.DataFrame(_result.candidates)
        # Put score first, strip right_ prefix from display labels
        _score_col = _cand_df.pop("score")
        _cand_df.insert(0, "score", _score_col)
        _cand_df.columns = [
            c.replace("right_", "") if c != "score" else c
            for c in _cand_df.columns
        ]
        st.dataframe(
            _cand_df.style.background_gradient(subset=["score"], cmap="RdYlGn"),
            use_container_width=True,
            height=min(40 + 35 * len(_cand_df), 420),
        )

        # Download
        st.download_button(
            "⬇ Download candidates CSV",
            data=_cand_df.to_csv(index=False).encode(),
            file_name=f"candidates_{dataset_key}_{_sel}.csv",
            mime="text/csv",
        )
    else:
        st.info("No candidates were returned.")


# ══ Tab 7 — Retrieval Evaluation ══════════════════════════════════════════════
with tab7:
    st.subheader("Retrieval evaluation — cross-dataset comparison")
    st.caption(
        "Aggregate metrics from end-to-end retrieval runs on all three datasets.  "
        "Each query has exactly one correct match in its pool; ranking metrics "
        "(MRR, top-k) reflect how well the model ranks the true match.  "
        "Generate or refresh with:  "
        "`python -m src.run --dataset <name> --mode retrieve`  then  "
        "`python -m src.run --compare --mode retrieve`"
    )

    _ret_cmp_path = Path("results/retrieval_comparison.csv")

    if not _ret_cmp_path.exists():
        st.info(
            "No retrieval comparison found yet.  Run retrieval for each dataset, then:\n\n"
            "```\n"
            "python -m src.run --dataset walmart_amazon --mode retrieve\n"
            "python -m src.run --dataset dblp_acm       --mode retrieve\n"
            "python -m src.run --dataset dblp_googlescholar   --mode retrieve\n"
            "python -m src.run --compare --mode retrieve\n"
            "```"
        )
    else:
        _ret_df = pd.read_csv(_ret_cmp_path, index_col="dataset")

        # ── Summary metric cards ──────────────────────────────────────────────
        _datasets_present = list(_ret_df.index)
        for ds_name in _datasets_present:
            row = _ret_df.loc[ds_name]
            ds_label = DATASET_LABELS.get(ds_name, ds_name)
            st.markdown(f"**{ds_label}**")
            c1, c2, c3, c4, c5 = st.columns(5)
            def _fmt(v, pct=False):
                if v is None or (isinstance(v, float) and v != v):
                    return "n/a"
                return f"{v:.1%}" if pct else f"{v:.4f}"
            c1.metric("Match accuracy",      _fmt(row.get("match_accuracy"),      pct=True))
            c2.metric("False approval rate", _fmt(row.get("false_approval_rate"), pct=True))
            c3.metric("MRR",                 _fmt(row.get("mrr")))
            c4.metric("Top-3 accuracy",      _fmt(row.get("top_3_accuracy"),      pct=True))
            c5.metric("Top-5 accuracy",      _fmt(row.get("top_5_accuracy"),      pct=True))

        st.divider()

        # ── Full comparison table ─────────────────────────────────────────────
        st.markdown("**Full metrics table**")
        _display_cols = [
            "n_invoices", "match_accuracy", "false_approval_rate",
            "review_rate", "rejection_rate",
            "top_3_accuracy", "top_5_accuracy", "mrr",
            "mean_confidence_correct", "mean_confidence_false",
        ]
        _display_cols = [c for c in _display_cols if c in _ret_df.columns]
        st.dataframe(_ret_df[_display_cols], use_container_width=True)

        st.download_button(
            "⬇ Download retrieval comparison CSV",
            data=_ret_df.to_csv().encode(),
            file_name="retrieval_comparison.csv",
            mime="text/csv",
        )

        st.divider()

        # ── MRR bar chart ─────────────────────────────────────────────────────
        if "mrr" in _ret_df.columns and _ret_df["mrr"].notna().any():
            st.markdown("**Mean Reciprocal Rank by dataset**")
            _mrr_df = _ret_df[["mrr"]].dropna().reset_index()
            _mrr_df["dataset_label"] = _mrr_df["dataset"].map(
                lambda k: DATASET_LABELS.get(k, k)
            )
            fig_mrr = px.bar(
                _mrr_df, x="dataset_label", y="mrr",
                color="mrr",
                color_continuous_scale=[[0, C["reject"]], [0.5, C["flag"]], [1, C["approve"]]],
                labels={"dataset_label": "Dataset", "mrr": "MRR"},
                text_auto=".3f",
            )
            fig_mrr.update_layout(**_PL, height=280, coloraxis_showscale=False,
                                  yaxis_range=[0, 1])
            st.plotly_chart(fig_mrr, width="stretch")

        # ── Top-k accuracy grouped bar chart ──────────────────────────────────
        _topk_cols = [c for c in ["top_3_accuracy", "top_5_accuracy"] if c in _ret_df.columns]
        if _topk_cols:
            st.markdown("**Top-k accuracy by dataset**")
            _topk_df = _ret_df[_topk_cols].reset_index().melt(
                id_vars="dataset", var_name="k", value_name="accuracy"
            )
            _topk_df["dataset_label"] = _topk_df["dataset"].map(
                lambda k: DATASET_LABELS.get(k, k)
            )
            _topk_df["k"] = _topk_df["k"].map(
                {"top_3_accuracy": "Top-3", "top_5_accuracy": "Top-5"}
            )
            fig_topk = px.bar(
                _topk_df.dropna(), x="dataset_label", y="accuracy", color="k",
                barmode="group",
                color_discrete_map={"Top-3": C["primary"], "Top-5": C["purple"]},
                labels={"dataset_label": "Dataset", "accuracy": "Accuracy", "k": ""},
                text_auto=".1%",
            )
            fig_topk.update_layout(**_PL, height=280, yaxis_range=[0, 1])
            st.plotly_chart(fig_topk, width="stretch")
