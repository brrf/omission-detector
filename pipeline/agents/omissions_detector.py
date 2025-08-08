from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pipeline.state import PipelineState, HPIFact, FactClassification
from llm_utils import call_llm

_MODEL = "gpt-4o-mini"
_MAX_CANDIDATES = 10  # keep prompts cheap

# ---------------- helpers ----------------

def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s.strip())
    return s

def _same_problem(tf: HPIFact, nf: HPIFact, *, total_problems: int = 0) -> bool:
    """
    Strict: if both sides have problem_id, require equality.
    Soft: if the visit has exactly ONE problem, allow match even if an id is missing.
    """
    tpid = getattr(tf, "problem_id", None)
    npid = getattr(nf, "problem_id", None)
    if tpid and npid:
        return tpid == npid
    return total_problems == 1

def _compact_fact_view(f: HPIFact) -> Dict[str, Any]:
    return {
        "id": getattr(f, "id", None),
        "code": _norm(getattr(f, "code", None)),
        "polarity": _norm(getattr(f, "polarity", None)),
        "time_scope": _norm(getattr(f, "time_scope", None)) or None,
        "value": _norm(getattr(f, "value", None)) or None,
        "evidence_text": _norm(getattr(getattr(f, "evidence_span", None), "text", None)) or None,
    }

def _build_prompt(transcript_fact: HPIFact, candidates: List[HPIFact]) -> str:
    """
    Lenient, semantics-first instructions. Return JSON only.
    """
    tf = _compact_fact_view(transcript_fact)
    cands = [_compact_fact_view(nf) for nf in candidates[:_MAX_CANDIDATES]]

    guidance = (
        "You are matching a single TRANSCRIPT fact to any of several NOTE facts. "
        "All items already share the same HPI code and problem context. "
        "Decide if the transcript fact is PRESENT in the note, OMITTED from the note, or in direct CONFLICT.\n\n"
        "Lenient matching rules:\n"
        " • Consider semantics over surface form: synonyms, paraphrases, and small wording differences count as the same.\n"
        " • Minor time phrasing differences (e.g., 'last couple of weeks' vs 'several weeks ago') count as the same onset.\n"
        " • Time scope labels (e.g., None vs 'changed') do not by themselves cause conflict.\n"
        " • Use CONFLICT only for clear contradictions (e.g., polarity disagreement: 'present' vs 'absent'/'denies'; "
        "   or mutually exclusive values).\n"
        " • Default to PRESENT if at least one note candidate describes the same real-world fact.\n\n"
        "Return strict JSON with: "
        "{status: 'present'|'omitted'|'conflict', best_note_fact_id: string|null, match_confidence: 0.0-1.0, reason: string}."
    )

    prompt = (
        f"{guidance}\n\n"
        f"TRANSCRIPT_FACT:\n{json.dumps(tf, ensure_ascii=False)}\n\n"
        f"NOTE_FACT_CANDIDATES ({len(cands)}):\n{json.dumps(cands, ensure_ascii=False)}\n"
        "Respond with ONLY the JSON object."
    )
    return prompt

def _llm_json_only(prompt: str) -> Dict[str, Any]:
    """
    Calls the LLM and returns a Python dict.
    Minimal parsing: JSON loads; if that fails, slice first {...}.
    """
    raw = call_llm(_MODEL, prompt)
    try:
        return json.loads(raw)
    except Exception:
        l = raw.find("{")
        r = raw.rfind("}")
        if l != -1 and r != -1 and r > l:
            return json.loads(raw[l : r + 1])
        raise RuntimeError(f"LLM did not return JSON: {raw[:200]}...")

# ---------------- core ----------------

def omissions_detector(state: PipelineState) -> PipelineState:
    """
    LLM-driven matching:
      - For each transcript fact, filter note facts by (code AND problem_id).
      - If none -> 'omitted'.
      - Else ask the LLM to decide present/omitted/conflict.
    """
    tfacts = state.transcript_facts or []
    nfacts = state.note_facts or []
    total_problems = len(state.problems or [])

    out: List[FactClassification] = []

    for tf in tfacts:
        code = (getattr(tf, "code", None) or "").strip()
        candidates = [
            nf for nf in nfacts
            if (getattr(nf, "code", None) or "").strip() == code
            and _same_problem(tf, nf, total_problems=total_problems)
        ]

        if not candidates:
            status = "omitted"
        else:
            prompt = _build_prompt(tf, candidates)
            resp = _llm_json_only(prompt)
            status = str(resp.get("status", "omitted")).lower()
            if status not in {"present", "omitted", "conflict"}:
                status = "omitted"

        out.append(FactClassification(
            fact_id=tf.id,
            status=status,
            recommended_inclusion=None,
            materiality=0.0,
            priority_label=None,
        ))

    state.classifications = out
    return state
