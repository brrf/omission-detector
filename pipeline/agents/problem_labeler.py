# pipeline/agents/problem_labeler.py
from __future__ import annotations

import json
import hashlib
import uuid
from typing import Any, Dict, List, Optional

from pipeline.state import PipelineState, HPIFact, Problem
from llm_utils import call_llm

_MODEL = "gpt-4o-mini"

_PROMPT = """
You are an expert clinical scribe. You will receive:
1) The diarized clinical transcript (Clinician/Patient turns).
2) A JSON array of HPI facts extracted from the transcript (id, code, polarity, value, evidence_span.text).

Task:
- Define a concise list of PROBLEMS (complaints/conditions) active in today's visit.
- Then assign every fact to exactly one of those problems.

Guidance:
- Problem names should be short clinical phrases (e.g., "Left knee pain", "Migraine flare", "Cough/URI", "LE edema").
- Merge OLDCARTS qualifiers (onset, severity, location, trajectory, aggravators, etc.) under the same problem.
- Keep the problem set small and clinically meaningful.
- Use active_today=true for problems discussed in today's assessment/plan.

Return ONLY this JSON object shape (no prose):
{{
  "problems": [{{"name": "<short phrase>", "active_today": true|false}}, ...],
  "assignments": [{{"fact_id": "<id from input>", "problem": "<exact problem name from problems>"}} , ...]
}}

Constraints:
- Every input fact id must appear exactly once in assignments.
- Each "problem" string in assignments MUST exactly match one of problems[].name.
- If a fact is purely contextual but clearly belongs to a main complaint, still assign it to that complaint.

=== TRANSCRIPT ===
{transcript}

=== HPI FACTS (JSON array) ===
{facts_json}
""".strip()


def _llm_json_obj(model: str, prompt: str) -> Dict[str, Any]:
    raw = call_llm(model, prompt)
    try:
        data = json.loads(raw)
    except Exception:
        # minimal recovery: slice between first '{' and last '}'
        l = raw.find("{")
        r = raw.rfind("}")
        if l != -1 and r != -1 and r > l:
            data = json.loads(raw[l : r + 1])
        else:
            raise
    if not isinstance(data, dict):
        raise RuntimeError("ProblemLabeler: LLM did not return a JSON object.")
    return data


def _stable_problem_id(run_id: Optional[str], name: str) -> str:
    """
    Stable UUIDv5 per (run_id, normalized name). Ensures id is reproducible within a run.
    """
    base = f"{(run_id or '').strip()}||{name.strip().lower()}"
    sig = hashlib.md5(base.encode("utf-8")).hexdigest()
    return uuid.uuid5(uuid.NAMESPACE_URL, sig).hex


def problem_labeler(state: PipelineState) -> PipelineState:
    """
    1) Ask LLM to list problems and map each fact -> one problem.
    2) Create Problem objects with stable per-run IDs.
    3) Assign problem_id on every HPIFact.
    4) Append problems to state.problems.
    """
    if not state.transcript_facts:
        # Nothing to label; keep state unchanged.
        return state

    # Build a compact facts payload for the LLM
    facts_payload = []
    for f in state.transcript_facts:
        facts_payload.append({
            "id": f.id,
            "code": f.code,
            "polarity": f.polarity,
            "value": f.value,
            "evidence_span": {"text": getattr(f.evidence_span, "text", "") if f.evidence_span else ""},
        })
    prompt = _PROMPT.format(
        transcript=state.transcript or "",
        facts_json=json.dumps(facts_payload, ensure_ascii=False),
    )
    data = _llm_json_obj(_MODEL, prompt)
    problems_in = data.get("problems") or []
    assignments_in = data.get("assignments") or []

    # 1) Build Problem objects, de-dup by name
    name_to_problem: Dict[str, Problem] = {}
    for p in problems_in:
        name = (p.get("name") or "").strip()
        if not name:
            continue
        active_today = bool(p.get("active_today", True))
        pid = _stable_problem_id(state.run_id, name)
        if name not in name_to_problem:
            name_to_problem[name] = Problem(id=pid, name=name, active_today=active_today)

    # If the model returned no problems, create a single fallback bucket
    if not name_to_problem:
        fallback_name = "General HPI problem"
        name_to_problem[fallback_name] = Problem(
            id=_stable_problem_id(state.run_id, fallback_name),
            name=fallback_name,
            active_today=True,
        )

    # 2) Build fact_id -> problem_name map from assignments
    fact_to_pname: Dict[str, str] = {}
    for a in assignments_in:
        fid = a.get("fact_id")
        pname = (a.get("problem") or "").strip()
        if fid and pname:
            fact_to_pname[fid] = pname

    # 3) Ensure every fact is assigned; default to the first problem if missing
    default_problem_name = next(iter(name_to_problem.keys()))
    for f in state.transcript_facts:
        pname = fact_to_pname.get(f.id, default_problem_name)
        if pname not in name_to_problem:
            # If LLM invented a new name in assignments, create it on the fly
            name_to_problem[pname] = Problem(
                id=_stable_problem_id(state.run_id, pname),
                name=pname,
                active_today=True,
            )
        f.problem_id = name_to_problem[pname].id

    # 4) Merge Problems into state (avoid duplicates by id)
    existing_ids = {p.id for p in state.problems}
    new_problems = [p for p in name_to_problem.values() if p.id not in existing_ids]
    print(new_problems)
    state.problems.extend(new_problems)

    return state
