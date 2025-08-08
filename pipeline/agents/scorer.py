# pipeline/agents/scorer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.state import PipelineState, HPIFact, FactClassification, Problem
from llm_utils import call_llm

# We parse the framework YAML to get code -> weight.
# PyYAML is small and reliable; added to requirements.txt below.
import yaml

_MODEL = "gpt-4o-mini"


# ---------------- framework helpers ----------------

def _read_framework_yaml_text() -> str:
    """
    Locate and read framework/omission_framework.yaml by walking up from this file.
    """
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        candidate = base / "framework" / "omission_framework.yaml"
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    candidate = Path("framework/omission_framework.yaml")
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError("framework/omission_framework.yaml not found")


def _load_code_weight_map() -> Dict[str, int]:
    """
    Parse the YAML and build {code -> weight} (int). Unknown codes default to 0.
    """
    txt = _read_framework_yaml_text()
    data = yaml.safe_load(txt) or {}
    domains = ((data.get("taxonomy") or {}).get("domains") or [])
    code2w: Dict[str, int] = {}
    for d in domains:
        for it in (d.get("items") or []):
            code = (it.get("code") or "").strip()
            w = it.get("weight", 0)
            try:
                code2w[code] = int(w)
            except Exception:
                code2w[code] = 0
    return code2w


# ---------------- LLM helpers ----------------

_PROMPT = """
You are an expert clinical reviewer scoring a SINGLE transcript fact that was
**omitted or in conflict** with the HPI NOTE.

Decide four booleans about THIS transcript fact by looking at ALL note facts:

• redundant: true if this fact is already documented elsewhere in the NOTE
  (same real‑world information, even if phrased differently).
• recent: true if the fact’s onset/change is within the last ~14 days.
  (Use the fact’s value/time_scope text; treat phrases like “last couple of weeks”,
   “this week”, “since yesterday”, “3–10 days” as recent.)
• acute: true if the fact clearly describes a new/acute episode or a marked change/worsening.
• highly_relevant: true if including this fact would plausibly change TODAY’s assessment/plan
  (e.g., red flags, significant trajectory change, important modifier that alters risk or workup).

Consider SEMANTICS over exact wording. Small phrasing/time differences should still count as the same.

Return ONLY strict JSON:
{{
  "redundant": true|false,
  "recent": true|false,
  "acute": true|false,
  "highly_relevant": true|false,
  "reason": "<<= 200 chars, one-line justification>"
}}

=== TRANSCRIPT_FACT ===
{tf_json}

=== NOTE_FACTS (all) ===
{note_facts_json}

=== PROBLEMS (id, name, active_today) ===
{problems_json}
""".strip()


def _llm_json_obj(prompt: str) -> Dict[str, Any]:
    raw = call_llm(_MODEL, prompt)
    try:
        data = json.loads(raw)
    except Exception:
        # minimal recovery: slice between first '{' and last '}'
        l = raw.find("{")
        r = raw.rfind("}")
        if l != -1 and r != -1 and r > l:
            data = json.loads(raw[l:r+1])
        else:
            raise RuntimeError(f"Scorer LLM did not return JSON: {raw[:200]}...")
    if not isinstance(data, dict):
        raise RuntimeError("Scorer: LLM did not return a JSON object.")
    return data


def _compact_fact_for_llm(f: HPIFact) -> Dict[str, Any]:
    return {
        "id": getattr(f, "id", None),
        "code": (f.code or "").strip(),
        "polarity": (f.polarity or "").strip(),
        "value": (f.value or "").strip() or None,
        "problem_id": getattr(f, "problem_id", None),
        "time_scope": getattr(f, "time_scope", None),
        "evidence": getattr(getattr(f, "evidence_span", None), "text", None),
    }


def _problems_payload(problems: List[Problem]) -> List[Dict[str, Any]]:
    return [{"id": p.id, "name": p.name, "active_today": bool(p.active_today)} for p in (problems or [])]


# ---------------- core agent ----------------

def scorer(state: PipelineState) -> PipelineState:
    """
    Input: state.classifications, state.transcript_facts, state.note_facts, state.problems
    Output:
      - updates each FactClassification.materiality (int)
      - sets priority_label ("high"/"medium"/"low"/None)
      - populates state.prioritized with items having materiality >= 2 (omitted/conflict)
    """
    if not state.classifications:
        return state

    code2w = _load_code_weight_map()
    tf_by_id: Dict[str, HPIFact] = {f.id: f for f in (state.transcript_facts or [])}
    problems_by_id: Dict[str, Problem] = {p.id: p for p in (state.problems or [])}

    note_payload = [_compact_fact_for_llm(nf) for nf in (state.note_facts or [])]
    problems_payload = _problems_payload(state.problems or [])

    prioritized: List[FactClassification] = []
    updated: List[FactClassification] = []

    for cls in state.classifications:
        tf = tf_by_id.get(cls.fact_id)
        if not tf:
            # No matching transcript fact → leave as-is
            updated.append(cls)
            continue

        # 1) taxonomy weight
        taxonomy_weight = int(code2w.get((tf.code or "").strip(), 0))

        # 2) active_today
        active_today = True
        if getattr(tf, "problem_id", None):
            prob = problems_by_id.get(tf.problem_id)
            if prob is not None:
                active_today = bool(prob.active_today)

        recency_bonus = 0
        linkage_bonus = 0
        redundancy_penalty = 0
        active_today_bonus = 0

        # 3) LLM signals only for omitted/conflict
        if cls.status in {"omitted", "conflict"}:
            tf_payload = _compact_fact_for_llm(tf)
            prompt = _PROMPT.format(
                tf_json=json.dumps(tf_payload, ensure_ascii=False),
                note_facts_json=json.dumps(note_payload, ensure_ascii=False),
                problems_json=json.dumps(problems_payload, ensure_ascii=False),
            )
            resp = _llm_json_obj(prompt)

            redundant = bool(resp.get("redundant", False))
            recent = bool(resp.get("recent", False))
            acute = bool(resp.get("acute", False))
            highly_relevant = bool(resp.get("highly_relevant", False))

            if recent or acute:
                recency_bonus = 1
            if highly_relevant:
                linkage_bonus = 1
            if redundant:
                redundancy_penalty = 1
            if active_today:
                active_today_bonus = 1

        # 4) Materiality score
        materiality = taxonomy_weight + recency_bonus + linkage_bonus - redundancy_penalty + active_today_bonus

        # 5) Update classification in-place
        cls.materiality = float(materiality)
        if materiality >= 4:
            cls.priority_label = "high"
        elif materiality >= 3:
            cls.priority_label = "medium"
        elif materiality >= 2:
            cls.priority_label = "low"
        else:
            cls.priority_label = None

        updated.append(cls)

        # 6) Prioritize if threshold met and status merits inclusion
        if cls.status in {"omitted", "conflict"} and materiality >= 2:
            prioritized.append(cls)

    state.classifications = updated
    state.prioritized = prioritized


    return state
