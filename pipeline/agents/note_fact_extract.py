# pipeline/agents/omissions_detector.py
from __future__ import annotations

import json
import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.state import PipelineState, HPIFact, EvidenceSpan, Problem
from llm_utils import call_llm

_MODEL = "gpt-4o-mini"

_PROMPT = """
You are an expert clinician. You will receive:
1) A classification taxonomy/weights in YAML (omission_framework.yaml).
2) A PROBLEM LIST (id + name).
3) An HPI NOTE (provider-written HPI text, not a chat).

Your tasks:
A) Read the HPI NOTE and extract **atomic HPI facts** (one clinical concept per item).
B) For each fact, assign **one best taxonomy code** from the YAML (e.g., "D2.ON-01").
C) Set **polarity** to exactly one of: "present", "denied", "uncertain".
D) Produce a concise, **normalized value** string (e.g., "onset: 3 days ago", "location: RUQ", "severity: 8/10").
E) Include a short **evidence_span** with the exact quote from the HPI NOTE that supports the fact.
F) For each fact, choose a **problem_id** from the PROBLEM LIST (use the `id` field).
G) If applicable, add **time_scope** as one of: "acute", "chronic", "baseline", "changed" (else omit or null).

Return format:
- Output **only** a JSON array (no prose) of objects with **exactly** these keys:
  - id: string (any unique string; will be normalized downstream)
  - code: string taxonomy code (e.g., "D2.ON-01" or "UNMAPPED" if truly unmappable)
  - polarity: "present" | "denied" | "uncertain"
  - value: short normalized string
  - problem_id: string | null   // must be one of the PROBLEM LIST `id` values, or null
  - time_scope: "acute" | "chronic" | "baseline" | "changed" | null (optional)
  - evidence_span: {{ "text": "<verbatim quote from HPI NOTE>" }}

Constraints:
- Keep each fact single-concept; split multi-concept statements.
- Prefer clinically useful normalizations (units, sides, timing windows).
- If a statement is a denial (e.g., "No fever"), use polarity="denied" and value like "fever: denied".
- Output JSON **array only**, with no commentary.
- Choose `problem_id` from the PROBLEM LIST by id (not name). Use null if genuinely none apply.

=== TAXONOMY YAML (omission_framework.yaml) ===
{framework_yaml}

=== PROBLEM LIST (JSON) ===
{problems_json}

=== HPI NOTE ===
{hpi_note}
""".strip()


def _read_framework_yaml() -> str:
    """
    Locate and read framework/omission_framework.yaml by walking up from this file.
    """
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        candidate = base / "framework" / "omission_framework.yaml"
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    # Also try CWD (useful when running from repo root)
    candidate = Path("framework/omission_framework.yaml")
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError("framework/omission_framework.yaml not found")


def _llm_json_only(model: str, prompt: str) -> List[Dict[str, Any]]:
    raw = call_llm(model, prompt)
    try:
        return json.loads(raw)
    except Exception:
        # minimal recovery: slice between first '[' and last ']'
        l = raw.find("[")
        r = raw.rfind("]")
        if l != -1 and r != -1 and r > l:
            return json.loads(raw[l : r + 1])
        raise


def _deterministic_id(code: str, polarity: str, value: str) -> str:
    """
    Stable ID from primary fields so repeated runs dedupe cleanly.
    We intentionally do NOT include problem_id so that transcript vs note
    facts with the same core content share the same identity.
    """
    base = f"{(code or '').strip().upper()}||{(polarity or '').strip().lower()}||{(value or '').strip().lower()}"
    sig = hashlib.md5(base.encode("utf-8")).hexdigest()
    return uuid.uuid5(uuid.NAMESPACE_URL, sig).hex


def _norm_polarity(p: Optional[str]) -> str:
    p = (p or "").strip().lower()
    return p if p in {"present", "denied", "uncertain"} else "present"


def _norm_time_scope(t: Optional[str]) -> Optional[str]:
    t = (t or "").strip().lower()
    return t if t in {"acute", "chronic", "baseline", "changed"} else None


def _problems_to_payload(problems: List[Problem]) -> str:
    """
    Render problems as a compact JSON list for the prompt.
    Example: [{"id": "p1", "name": "LE edema/pain"}, ...]
    """
    data = [{"id": p.id, "name": p.name} for p in (problems or [])]
    return json.dumps(data, ensure_ascii=False)


def _map_problem_id(
    raw_pid: Optional[str],
    problems: List[Problem],
) -> Optional[str]:
    """
    Accepts an LLM-provided problem identifier or name and maps it
    to a known problem id if possible; otherwise returns None.
    """
    if not problems:
        return None
    ids = {p.id for p in problems}
    by_name = {p.name.strip().lower(): p.id for p in problems if p.name}

    token = (raw_pid or "").strip()
    if not token:
        return None
    # Exact id match
    if token in ids:
        return token
    # Name match (case-insensitive)
    lower = token.lower()
    if lower in by_name:
        return by_name[lower]
    return None


def _to_hpifact(item: Dict[str, Any], problems: List[Problem]) -> HPIFact:
    code = (item.get("code") or "UNMAPPED").strip()
    polarity = _norm_polarity(item.get("polarity"))
    value = (item.get("value") or "").strip()
    problem_id = _map_problem_id(item.get("problem_id") or item.get("problem"), problems)
    time_scope = _norm_time_scope(item.get("time_scope"))

    # Recompute a stable id regardless of what the LLM emitted
    fid = _deterministic_id(code, polarity, value)

    ev = None
    ev_obj = item.get("evidence_span")
    if isinstance(ev_obj, dict):
        txt = (ev_obj.get("text") or "").strip()
        if txt:
            ev = EvidenceSpan(text=txt)

    return HPIFact(
        id=fid,
        code=code,
        polarity=polarity,
        value=value,
        problem_id=problem_id,
        time_scope=time_scope,   # may be None
        evidence_span=ev,
    )


def note_fact_extract(state: PipelineState) -> PipelineState:
    """
    1) Read omission_framework.yaml and pass it to the LLM with:
         - the current problem list (state.problems) with ids+names
    2) Expect a JSON **list** of HPIFact-like dicts (see prompt).
    3) Convert each to the runtime HPIFact dataclass.
    4) Append to state.note_facts and return state.
    """
    hpi_note = (state.hpi_input).strip()
    if not hpi_note:
        # Nothing to do
        return state

    framework_yaml = _read_framework_yaml()
    problems_payload = _problems_to_payload(state.problems or [])

    prompt = _PROMPT.format(
        framework_yaml=framework_yaml,
        problems_json=problems_payload,
        hpi_note=hpi_note,
    )
    items = _llm_json_only(_MODEL, prompt)
    if not isinstance(items, list):
        raise RuntimeError("LLM did not return a JSON list for note HPIFacts")

    hpifacts: List[HPIFact] = []
    for it in items:
        try:
            hpifacts.append(_to_hpifact(it if isinstance(it, dict) else {}, state.problems or []))
        except Exception:
            # Skip malformed entries rather than failing the whole run
            continue

    # Append to state.note_facts
    state.note_facts.extend(hpifacts)
    return state
