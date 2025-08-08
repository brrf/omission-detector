# --- Mock agent functions (no-ops) -----------------------------------
# REPLACE the no-op with a writer
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

def metric_omitter(state: PipelineState) -> PipelineState:
    """
    Persist prioritized omissions + a compact per-run blob for humans & evals.
    """
    ts = datetime.utcnow().isoformat()
    run_id = state.run_id or "unknown"

    # 1) JSONL: one line per prioritized omission
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

    # 2) CSV mirror (for PMs/Slides)
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

    # 3) Per-run compact JSON (humans can open one file and see the visit)
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
    }
    with open(_OUT_DIR / "by_run" / f"{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(by_run, f, ensure_ascii=False, indent=2)

    return state
