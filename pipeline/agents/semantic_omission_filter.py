from __future__ import annotations

import json
from typing import Any, Dict, List

from pipeline.state import PipelineState, HPIFact, FactClassification
from llm_utils import call_llm

_MODEL = "gpt-4o"

_PROMPT = """
You are an expert clinical reviewer. You will receive:
1) ORIGINAL_HPI — provider-written HPI text (string).
2) OMISSIONS — JSON array of HPIFact-like items from the TRANSCRIPT that were flagged as omitted/conflict.
   Each item has:
     - fact_id: string
     - value: string | null
     - polarity: "present" | "denied" | "uncertain"
     - time_scope: "acute" | "chronic" | "baseline" | "changed" | null
     - problem: short problem name (string) or null
     - evidence: short verbatim quote from transcript (string) or null
     - is_prechart: boolean  (true if fact came from pre-chart text)

Task:
- Decide which items are TRUE omissions worth keeping.

Keep an item only if:
- Its concept is **not already expressed** in ORIGINAL_HPI by meaning
  (treat synonyms/paraphrases, numeric/date format equivalents, and negations as the same), AND
- It is **HPI-relevant** to today's visit, e.g.:
  • chief concern; symptoms/trajectory;
  • severity or frequency;
  • triggers/modifiers;
  • medications/adherence/tolerability;
  • key measurements tied to the visit (e.g., BP/HR);
  • recent ED/hospital events;
  • pertinent psychosocial/trauma.

Drop if the concept is semantically present in ORIGINAL_HPI or not HPI-relevant.

Return ONLY this JSON object (no prose):
{{"kept_omissions":["<fact_id>", "..."]}}

Constraints:
- Do not rewrite the HPI.
- Do not invent content.
- If none qualify, return {{"kept_omissions":[]}}.
- Return the JSON object only (no additional text).

=== ORIGINAL_HPI ===
{original_hpi}

=== OMISSIONS (JSON array) ===
{omissions_json}
""".strip()


def _llm_json_obj(prompt: str) -> Dict[str, Any]:
    raw = call_llm(_MODEL, prompt)
    try:
        data = json.loads(raw)
    except Exception:
        l = raw.find("{"); r = raw.rfind("}")
        if l != -1 and r != -1 and r > l:
            data = json.loads(raw[l : r + 1])
        else:
            raise RuntimeError(f"SemanticOmissionFilter: LLM did not return JSON: {raw[:200]}...")
    if not isinstance(data, dict):
        raise RuntimeError("SemanticOmissionFilter: LLM did not return a JSON object.")
    return data


def semantic_omission_filter(state: PipelineState) -> PipelineState:
    """
    Inputs:
      - state.hpi_input (string HPI text to compare)
      - state.classifications (FactClassification list from OmissionsDetector)

    Behavior:
      - Build OMISSIONS as enriched HPIFact-like dicts (no taxonomy code; include problem name + evidence).
      - Ask LLM for {{"kept_omissions":["<fact_id>", ...]}}.
      - Remove from state.classifications any omitted/conflict items NOT in kept_omissions.
      - Record stats in state.metrics["semantic_filter"].
    """
    if not state.classifications:
        return state

    original_hpi = (state.hpi_input or "").strip()

    tf_by_id: Dict[str, HPIFact] = {f.id: f for f in (state.transcript_facts or [])}
    # Map problem_id -> problem name for readability
    pid2name = {
        getattr(p, "id", None): getattr(p, "name", None)
        for p in (state.problems or [])
        if getattr(p, "id", None) and getattr(p, "name", None)
    }

    # Collect alleged omissions from classifications and enrich with HPIFact fields
    omissions: List[Dict[str, Any]] = []
    target_cls_ids: List[str] = []
    for cls in state.classifications:
        if cls.status not in {"omitted", "conflict"}:
            continue
        tf = tf_by_id.get(cls.fact_id)
        if not tf:
            continue
        omissions.append({
            "fact_id": getattr(tf, "id", None),
            "value": (tf.value or "").strip() or None,
            "polarity": (tf.polarity or "").strip() or None,
            "time_scope": getattr(tf, "time_scope", None),
            "problem": pid2name.get(getattr(tf, "problem_id", None)),
            "evidence": getattr(getattr(tf, "evidence_span", None), "text", None),
            "is_prechart": bool(getattr(tf, "isPrechartFact", False)),
        })
        target_cls_ids.append(cls.fact_id)

    if not omissions:
        return state

    prompt = _PROMPT.format(
        original_hpi=original_hpi,
        omissions_json=json.dumps(omissions, ensure_ascii=False),
    )

    try:
        resp = _llm_json_obj(prompt)
        kept_ids = resp.get("kept_omissions")
        if not isinstance(kept_ids, list):
            kept_ids = []
        kept_ids = {str(x).strip() for x in kept_ids if x is not None}
    except Exception as e:
        # Fail-open: keep everything if model response is malformed
        kept_ids = set(target_cls_ids)
        sf = state.metrics.setdefault("semantic_filter", {})
        sf["fallback"] = "keep_all"
        sf["error"] = str(e)

    new_classifications: List[FactClassification] = []
    dropped_items: List[Dict[str, Any]] = []

    for cls in state.classifications:
        if cls.status in {"omitted", "conflict"}:
            if cls.fact_id in kept_ids:
                new_classifications.append(cls)
            else:
                tf = tf_by_id.get(cls.fact_id)
                dropped_items.append({
                    "fact_id": cls.fact_id,
                    "value": (tf.value if tf else None),
                    "polarity": (tf.polarity if tf else None),
                    "time_scope": (tf.time_scope if tf else None),
                    "problem": pid2name.get(getattr(tf, "problem_id", None)) if tf else None,
                    "status": cls.status,
                })
        else:
            # Non-omission items pass through untouched
            new_classifications.append(cls)

    state.classifications = new_classifications

    # Metrics
    sf = state.metrics.setdefault("semantic_filter", {})
    sf["input_omissions"] = omissions
    sf["kept_ids"] = sorted(list(kept_ids))
    sf["input_count"] = len(omissions)
    sf["kept_count"] = len(kept_ids)
    sf["dropped_count"] = len(omissions) - len(kept_ids)
    sf["dropped_items"] = dropped_items

    return state
