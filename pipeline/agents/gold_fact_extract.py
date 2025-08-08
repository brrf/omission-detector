# pipeline/agents/gold_fact_extract.py
from __future__ import annotations

import json
import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.state import PipelineState, HPIFact, EvidenceSpan
from llm_utils import call_llm

_MODEL = "gpt-4o-mini"

# Single prompt used for BOTH transcript and pre-chart text
_PROMPT = """
You are an expert clinician. You will receive:
1) A classification taxonomy/weights in YAML (omission_framework.yaml).
2) A block of clinical text that is either:
   • A diarized clinical CHAT transcript (Clinician/Patient turns), OR
   • PRE-CHART TEXT (provider-written notes captured before the visit).

Your task:
A) Read the text and extract **atomic HPI facts** (each at most one clinical concept).
B) For each fact, assign a **single best taxonomy code** from the YAML (e.g., "D2.ON-01").
C) Set **polarity** to exactly one of: "present", "denied", "uncertain".
D) Produce a concise, **normalized value** string (e.g., "onset: 3 days ago", "location: RUQ", "severity: 8/10").
E) Include a short **evidence_span** with the exact quote from the input text that supports the fact.

Return format:
- Output **only** a JSON array (no prose) of objects with **exactly** these keys:
  - id: string (unique; leave as any string — it will be normalized downstream)
  - code: string taxonomy code (e.g., "D2.ON-01" or "UNMAPPED" if you genuinely cannot map)
  - polarity: "present" | "denied" | "uncertain"
  - value: short normalized string
  - evidence_span: {{"text": "<verbatim quote>"}}

Constraints:
- Do not include types, labels, or weights in the output—use only the keys above.
- Keep each fact as a single concept. Split multi-concept statements into separate items.
- Prefer clinically useful normalizations (units, sides, timing windows).
- If a statement is a clear denial (e.g., "Denies fever"), set polarity="denied" and value like "fever: denied".
- Output JSON **array only**, with no commentary.

=== TAXONOMY YAML (omission_framework.yaml) ===
{framework_yaml}

=== INPUT TEXT ===
{text_block}
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
    Stable ID from primary fields so repeated runs dedupe cleanly on meaning.
    (We still do NOT perform cross-input deduplication here.)
    """
    base = f"{(code or '').strip().upper()}||{(polarity or '').strip().lower()}||{(value or '').strip().lower()}"
    sig = hashlib.md5(base.encode("utf-8")).hexdigest()
    return uuid.uuid5(uuid.NAMESPACE_URL, sig).hex


def _norm_polarity(p: Optional[str]) -> str:
    p = (p or "").strip().lower()
    return p if p in {"present", "denied", "uncertain"} else "present"


def _to_hpifact(item: Dict[str, Any], *, is_prechart: bool) -> HPIFact:
    code = (item.get("code") or "UNMAPPED").strip()
    polarity = _norm_polarity(item.get("polarity"))
    value = (item.get("value") or "").strip()
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
        evidence_span=ev,
        isPrechartFact=bool(is_prechart),
    )


def _extract_facts(framework_yaml: str, text_block: str, *, is_prechart: bool) -> List[HPIFact]:
    """
    Run the single prompt against any text block (transcript OR pre-chart),
    then map the results into HPIFact with the appropriate isPrechartFact flag.
    """
    prompt = _PROMPT.format(framework_yaml=framework_yaml, text_block=text_block or "")
    items = _llm_json_only(_MODEL, prompt)
    if not isinstance(items, list):
        raise RuntimeError("LLM did not return a JSON list for HPIFacts")
    facts: List[HPIFact] = []
    for it in items:
        try:
            facts.append(_to_hpifact(it if isinstance(it, dict) else {}, is_prechart=is_prechart))
        except Exception as e:
            print(f"Error processing item {it!r}: {e}")
            continue
    return facts


def gold_fact_extract(state: PipelineState) -> PipelineState:
    """
    1) Read omission_framework.yaml.
    2) Extract facts from:
         a) state.transcript  -> isPrechartFact = False
         b) state.pre_chart   -> isPrechartFact = True
       using the SAME prompt.
    3) Concatenate results (NO de-duplication).
    4) Set state.transcript_facts and return state.
    """
    framework_yaml = _read_framework_yaml()

    out: List[HPIFact] = []

    # (a) Transcript facts (if present)
    transcript_text = (state.transcript or "").strip()
    if transcript_text:
        out.extend(_extract_facts(framework_yaml, transcript_text, is_prechart=False))

    # (b) Pre-chart facts (if present) — separate input, no dedupe, just append
    prechart_text = (state.pre_chart or "").strip() if isinstance(state.pre_chart, str) else ""
    if prechart_text:
        out.extend(_extract_facts(framework_yaml, prechart_text, is_prechart=True))

    state.transcript_facts = out
    return state
