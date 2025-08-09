#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dashboard.py — quick eval viewer
Run:  streamlit run dashboard.py
"""

import json
from pathlib import Path

import streamlit as st
import pandas as pd

OUT_DIR = Path("out")
BY_RUN_DIR = OUT_DIR / "by_run"
EVAL_CSV = OUT_DIR / "gold_eval.csv"
OMIT_CSV = OUT_DIR / "omissions.csv"
NOTE_EVAL_CSV = OUT_DIR / "note_eval.csv"
PRIOR_EVAL_CSV = OUT_DIR / "prioritized_eval.csv"

st.set_page_config(page_title="Omission Detector – Eval Dashboard", layout="wide")
st.title("📊 Omission Detector – Evaluation Dashboard")

# --------- Data loading ----------
@st.cache_data(show_spinner=False)
def load_eval_csv():
    if EVAL_CSV.exists():
        df = pd.read_csv(EVAL_CSV)
        # de-duplicate on latest ts per run_id in case of multiple runs
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df.sort_values(["run_id","ts"], ascending=[True, False], inplace=True)
        df = df.drop_duplicates(subset=["run_id"], keep="first")
        return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_note_eval_csv():
    if NOTE_EVAL_CSV.exists():
        df = pd.read_csv(NOTE_EVAL_CSV)
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df.sort_values(["run_id","ts"], ascending=[True, False], inplace=True)
        df = df.drop_duplicates(subset=["run_id"], keep="first")
        return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_prioritized_eval_csv():
    if PRIOR_EVAL_CSV.exists():
        df = pd.read_csv(PRIOR_EVAL_CSV)
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df.sort_values(["run_id","ts"], ascending=[True, False], inplace=True)
        df = df.drop_duplicates(subset=["run_id"], keep="first")
        return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_omissions_csv():
    if OMIT_CSV.exists():
        df = pd.read_csv(OMIT_CSV)
        return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def list_by_run_files():
    return sorted([p for p in BY_RUN_DIR.glob("*.json")]) if BY_RUN_DIR.exists() else []

eval_df = load_eval_csv()
note_df = load_note_eval_csv()
prior_df = load_prioritized_eval_csv()
omit_df = load_omissions_csv()
by_run_files = list_by_run_files()

if eval_df.empty and note_df.empty and prior_df.empty and not by_run_files:
    st.info("No outputs found yet. Run the pipeline, then refresh.")
    st.stop()

# --------- Global summary ----------
st.subheader("Overall")
tab1, tab2, tab3 = st.tabs([
    "Transcript facts eval (gold_eval)",
    "HPI note eval (note_eval)",
    "Prioritized omissions eval",
])

with tab1:
    if not eval_df.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        macro_p = eval_df["precision"].mean() if "precision" in eval_df else 0.0
        macro_r = eval_df["recall"].mean() if "recall" in eval_df else 0.0
        macro_f = eval_df["f1"].mean() if "f1" in eval_df else 0.0
        avg_sim = eval_df["avg_similarity"].mean() if "avg_similarity" in eval_df else 0.0
        n_runs = len(eval_df)

        col1.metric("Runs", n_runs)
        col2.metric("Precision (strict)", f"{macro_p:.3f}")
        col3.metric("Recall (strict)", f"{macro_r:.3f}")
        col4.metric("F1 (strict)", f"{macro_f:.3f}")
        col5.metric("Avg Evidence Similarity", f"{avg_sim:.3f}")

        st.write("Per-run (strict) metrics")
        st.dataframe(
            eval_df[["run_id","n_pred","n_gold","precision","recall","f1","avg_similarity","avg_rougeL_f","thr"]],
            use_container_width=True
        )

        st.write("F1 distribution")
        st.bar_chart(eval_df.set_index("run_id")["f1"])
    else:
        st.info("No gold_eval.csv yet.")

with tab2:
    if not note_df.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        macro_p = note_df["precision"].mean() if "precision" in note_df else 0.0
        macro_r = note_df["recall"].mean() if "recall" in note_df else 0.0
        macro_f = note_df["f1"].mean() if "f1" in note_df else 0.0
        avg_sim = note_df["avg_similarity"].mean() if "avg_similarity" in note_df else 0.0
        n_runs = len(note_df)

        col1.metric("Runs", n_runs)
        col2.metric("Precision (strict)", f"{macro_p:.3f}")
        col3.metric("Recall (strict)", f"{macro_r:.3f}")
        col4.metric("F1 (strict)", f"{macro_f:.3f}")
        col5.metric("Avg Evidence Similarity", f"{avg_sim:.3f}")

        st.write("Per-run (strict) metrics")
        st.dataframe(
            note_df[["run_id","n_pred","n_gold","precision","recall","f1","avg_similarity","avg_rougeL_f","thr"]],
            use_container_width=True
        )

        st.write("F1 distribution")
        st.bar_chart(note_df.set_index("run_id")["f1"])
    else:
        st.info("No note_eval.csv yet.")

with tab3:
    if not prior_df.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        macro_p = prior_df["precision"].mean() if "precision" in prior_df else 0.0
        macro_r = prior_df["recall"].mean() if "recall" in prior_df else 0.0
        macro_f = prior_df["f1"].mean() if "f1" in prior_df else 0.0
        avg_sim = prior_df["avg_similarity"].mean() if "avg_similarity" in prior_df else 0.0
        n_runs = len(prior_df)

        col1.metric("Runs", n_runs)
        col2.metric("Precision (strict)", f"{macro_p:.3f}")
        col3.metric("Recall (strict)", f"{macro_r:.3f}")
        col4.metric("F1 (strict)", f"{macro_f:.3f}")
        col5.metric("Avg Evidence Similarity", f"{avg_sim:.3f}")

        st.write("Per-run (strict) metrics")
        st.dataframe(
            prior_df[["run_id","n_pred","n_gold","precision","recall","f1","avg_similarity","avg_rougeL_f","thr"]],
            use_container_width=True
        )

        st.write("F1 distribution")
        st.bar_chart(prior_df.set_index("run_id")["f1"])
    else:
        st.info("No prioritized_eval.csv yet.")

st.divider()

# --------- Run drilldown ----------
st.subheader("Run drilldown")
choices = [p.stem for p in by_run_files]
if choices:
    selected = st.selectbox("Select a run_id", options=choices)
    if selected:
        fpath = BY_RUN_DIR / f"{selected}.json"
        if fpath.exists():
            with fpath.open("r", encoding="utf-8") as f:
                blob = json.load(f)
            colA, colB = st.columns([2,1])
            with colA:
                st.write("**By-run summary**")
                st.json(blob.get("counts", {}))
                if blob.get("gold_eval"):
                    st.write("**Gold overlap (strict)**")
                    st.json({
                        "n_pred": blob["gold_eval"].get("n_pred"),
                        "n_gold": blob["gold_eval"].get("n_gold"),
                        **(blob["gold_eval"].get("strict", {}))
                    })
                if blob.get("note_eval"):
                    st.write("**HPI note overlap (strict)**")
                    st.json({
                        "n_pred": blob["note_eval"].get("n_pred"),
                        "n_gold": blob["note_eval"].get("n_gold"),
                        **(blob["note_eval"].get("strict", {}))
                    })
                if blob.get("prioritized_eval"):
                    st.write("**Prioritized omissions overlap (strict)**")
                    st.json({
                        "n_pred": blob["prioritized_eval"].get("n_pred"),
                        "n_gold": blob["prioritized_eval"].get("n_gold"),
                        **(blob["prioritized_eval"].get("strict", {}))
                    })
            with colB:
                st.write("**Problems**")
                st.json(blob.get("problems", []))
            st.write("**Prioritized omissions**")
            st.json(blob.get("prioritized", []))
else:
    st.info("No by-run JSON files found.")

st.divider()

# --------- Raw omissions table ----------
st.subheader("Raw prioritized omissions (CSV)")
if not omit_df.empty:
    st.dataframe(omit_df, use_container_width=True, height=300)
else:
    st.info("No omissions.csv yet.")

st.caption("Tip: re-run the pipeline to refresh metrics; this dashboard auto-caches files and reloads when they change.")
