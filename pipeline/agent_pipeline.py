# ----------------------------------------------------------------------
# agent_pipeline.py
# LangGraph pipeline: ingest → note gen (baseline + any variants) →
# fact extraction → enqueue metrics payload → return baseline note
# ----------------------------------------------------------------------
from collections.abc import Iterable
import uuid

from pipeline.state import PipelineState
from langgraph.graph import StateGraph, START, END
from pipeline.agents import (gold_fact_extract, problem_labeler, omissions_detector)

# --- Mock agent functions (no-ops) -----------------------------------
def omission_scorer(state: PipelineState) -> PipelineState:
    return state

def metric_omitter(state: PipelineState) -> PipelineState:
    return state

# ---------------- Build Graph -----------------------------------------
def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("GoldFacts", gold_fact_extract)
    g.add_edge(START, "GoldFacts")
    
    # ─── Independent branch C: gold facts → buckets
    g.add_node("ProblemLabeler",  problem_labeler)
    g.add_edge("GoldFacts",       "ProblemLabeler")

    # ─── Omission detection ─────────────────────────────────
    g.add_node("OmissionsDetector", omissions_detector)
    g.add_edge(["GoldFacts","ProblemLabeler"] , "OmissionsDetector")

    # ─── Scoring & emit ─────────────────────────────────────
    g.add_node("Scorer", omission_scorer)       # uses state.gold_facts + state.facts[…]
    g.add_edge(["OmissionsDetector", "ProblemLabeler"], "Scorer")
    g.add_edge("Scorer", "MetricOmitter")

    g.add_node("MetricOmitter", metric_omitter)
    g.add_edge("MetricOmitter", END)

    compiled = g.compile()

    # Optional: write a pipeline diagram (PNG) to the repo root
    try:
        png_bytes = compiled.get_graph().draw_mermaid_png()
        with open("pipeline_graph.png", "wb") as f:
            f.write(png_bytes)
    except Exception as e:
        # Diagram is optional; keep running even if rendering isn't available
        print(f"ℹ️  Skipping pipeline_graph.png render: {e}")

    return compiled

# ---------------- Runner ----------------------------------------------
def run_pipeline(state):
    """
    EXACTLY the dict produced by run_all.build_state_from_row(state).
    No kwargs. No alternative signatures.
    """
    graph = build_graph()

    init_state = PipelineState(
        run_id        = state.get("run_id") or str(uuid.uuid4()),
        transcript    = state.get("transcript") or "",
        source        = state.get("source"),
        pre_chart     = state.get("pre_chart"),
        interpreter   = state.get("interpreter"),
        hpi_input     = state.get("hpi_input"),
    )

    graph.invoke(init_state)



if __name__ == "__main__":
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else "raw_data/Z792.txt"
    version = sys.argv[2] if len(sys.argv) > 2 else "default"
    note = run_pipeline(version=version, file_path=fp)   # returns baseline note (or None)
    print(f"Baseline note: {note!r}")
