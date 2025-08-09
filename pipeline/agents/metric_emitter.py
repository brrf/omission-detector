# pipeline/agents/metric_emitter.py
# Persist prioritized omissions + per-run blobs + gold/note overlap metrics
import os, json, csv, pathlib
from datetime import datetime
from pipeline.state import PipelineState

_OUT_DIR = pathlib.Path("out")
_OUT_DIR.mkdir(exist_ok=True, parents=True)
(_OUT_DIR / "by_run").mkdir(exist_ok=True, parents=True)

def _append_csv_row(path, header, row_dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row_dict)

def _flatten_gold_eval(run_id: str, ts: str, ge: dict) -> dict:
    """
    Flatten state.metrics['gold_eval'] (or note_eval) to a single CSV row for dashboards.
    The schema is shared by both so we can reuse it.
    """
    base = {
        "ts": ts,
        "run_id": run_id,
        "found": bool(ge.get("found", False)),
        "annotation_path": ge.get("annotation_path"),
        "n_pred": ge.get("n_pred", 0),
        "n_gold": ge.get("n_gold", 0),
    }
    strict = ge.get("strict", {}) or {}
    code_only = ge.get("code_only", {}) or {}

    row = {
        **base,
        "thr": strict.get("threshold"),
        "tp": strict.get("tp", 0),
        "fp": strict.get("fp", 0),
        "fn": strict.get("fn", 0),
        "precision": strict.get("precision", 0.0),
        "recall": strict.get("recall", 0.0),
        "f1": strict.get("f1", 0.0),
        "avg_similarity": strict.get("avg_similarity", 0.0),
        "avg_rougeL_f": strict.get("avg_rougeL_f"),
        "tp_code": code_only.get("tp", 0),
        "fp_code": code_only.get("fp", 0),
        "fn_code": code_only.get("fn", 0),
        "precision_code": code_only.get("precision", 0.0),
        "recall_code": code_only.get("recall", 0.0),
        "f1_code": code_only.get("f1", 0.0),
    }
    return row

def metric_emitter(state: PipelineState) -> PipelineState:
    """
    Persist prioritized omissions + a compact per-run blob for humans & evals.
    Also persists gold-overlap metrics and HPI note-overlap metrics if present.
    """
    ts = datetime.utcnow().isoformat()
    run_id = state.run_id or "unknown"

    # A) Omissions JSONL (one line per prioritized item)
    jsonl_path = _OUT_DIR / "omissions.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for cls in (state.prioritized or []):
            tf = next((x for x in (state.transcript_facts or []) if x.id == cls.fact_id), None)
            rec = {
                "ts": ts,
                "run_id": run_id,
                "status": cls.status,
                "priority": cls.priority_label,
                "materiality": cls.materiality,
                "code": getattr(tf, "code", None),
                "value": getattr(tf, "value", None),
                "polarity": getattr(tf, "polarity", None),
                "problem_id": getattr(tf, "problem_id", None),
                "time_scope": getattr(tf, "time_scope", None),
                "evidence_text": getattr(getattr(tf, "evidence_span", None), "text", None),
                "recommended_inclusion": cls.recommended_inclusion,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # B) CSV mirror for omissions
    csv_path = _OUT_DIR / "omissions.csv"
    header = ["ts","run_id","status","priority","materiality","code","value","polarity",
              "problem_id","time_scope","evidence_text","recommended_inclusion"]
    for cls in (state.prioritized or []):
        tf = next((x for x in (state.transcript_facts or []) if x.id == cls.fact_id), None)
        row = {
            "ts": ts,
            "run_id": run_id,
            "status": cls.status,
            "priority": cls.priority_label,
            "materiality": cls.materiality,
            "code": getattr(tf, "code", None),
            "value": getattr(tf, "value", None),
            "polarity": getattr(tf, "polarity", None),
            "problem_id": getattr(tf, "problem_id", None),
            "time_scope": getattr(tf, "time_scope", None),
            "evidence_text": getattr(getattr(tf, "evidence_span", None), "text", None),
            "recommended_inclusion": cls.recommended_inclusion,
        }
        _append_csv_row(csv_path, header, row)

    # C) Per-run compact JSON (for quick drilldown in the dashboard)
    gold_eval = (state.metrics or {}).get("gold_eval", None)
    note_eval = (state.metrics or {}).get("note_eval", None)
    by_run = {
        "run_id": run_id,
        "problems": [{"id": p.id, "name": p.name, "active_today": bool(p.active_today)} for p in (state.problems or [])],
        "counts": {
            "total_transcript_facts": len(state.transcript_facts or []),
            "total_note_facts": len(state.note_facts or []),
            "omitted_or_conflict": len([c for c in (state.classifications or []) if c.status in {"omitted","conflict"}]),
            "prioritized": len(state.prioritized or []),
            "high": len([c for c in (state.prioritized or []) if c.priority_label=="high"]),
            "medium": len([c for c in (state.prioritized or []) if c.priority_label=="medium"]),
            "low": len([c for c in (state.prioritized or []) if c.priority_label=="low"]),
        },
        "prioritized": [
            {
              "fact_id": c.fact_id,
              "status": c.status,
              "priority": c.priority_label,
              "materiality": c.materiality
            } for c in (state.prioritized or [])
        ],
        "gold_eval": gold_eval if gold_eval else None,
        "note_eval": note_eval if note_eval else None,
    }
    with open(_OUT_DIR / "by_run" / f"{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(by_run, f, ensure_ascii=False, indent=2)

    # D) Persist transcript facts overlap (strict) if available
    if gold_eval and gold_eval.get("found"):
        # 1) Full blob per run (JSONL)
        ev_jsonl = _OUT_DIR / "gold_eval.jsonl"
        with open(ev_jsonl, "a", encoding="utf-8") as f:
            rec = {"ts": ts, "run_id": run_id, **gold_eval}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 2) Compact CSV row (one per run, latest wins in dashboard)
        ev_csv = _OUT_DIR / "gold_eval.csv"
        row = _flatten_gold_eval(run_id, ts, gold_eval)
        header = ["ts","run_id","found","annotation_path","n_pred","n_gold",
                  "thr","tp","fp","fn","precision","recall","f1","avg_similarity","avg_rougeL_f",
                  "tp_code","fp_code","fn_code","precision_code","recall_code","f1_code"]
        _append_csv_row(ev_csv, header, row)

    # E) Persist HPI note overlap (strict) if available
    if note_eval and note_eval.get("found"):
        # 1) Full blob per run (JSONL)
        ne_jsonl = _OUT_DIR / "note_eval.jsonl"
        with open(ne_jsonl, "a", encoding="utf-8") as f:
            rec = {"ts": ts, "run_id": run_id, **note_eval}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 2) Compact CSV row (one per run, latest wins in dashboard)
        ne_csv = _OUT_DIR / "note_eval.csv"
        row = _flatten_gold_eval(run_id, ts, note_eval)  # identical schema to gold_eval
        header = ["ts","run_id","found","annotation_path","n_pred","n_gold",
                  "thr","tp","fp","fn","precision","recall","f1","avg_similarity","avg_rougeL_f",
                  "tp_code","fp_code","fn_code","precision_code","recall_code","f1_code"]
        _append_csv_row(ne_csv, header, row)

    return state
