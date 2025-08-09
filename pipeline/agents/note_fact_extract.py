# pipeline/agents/note_fact_extract.py
from __future__ import annotations

import json
import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.state import PipelineState, HPIFact, EvidenceSpan, Problem
from llm_utils import call_llm

# NEW: imports for eval against annotations
import re
import yaml
from rapidfuzz import fuzz

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


def _to_hpifact(item: Dict[str, Any], problems: List[Problem], *, is_prechart: bool = False) -> HPIFact:
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
        isPrechartFact=bool(is_prechart),  # note facts are not pre-chart by default
    )


# ---------- EVAL against annotations/hpi_facts/{run_id}.yaml ----------

_SIM_THRESHOLD = 0.80  # match on evidence text within code bucket

def _repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        if (base / "annotations").exists() and (base / "framework").exists():
            return base
    return Path.cwd()

def _find_hpi_annotation_file(run_id: Optional[str]) -> Optional[Path]:
    if not run_id:
        return None
    root = _repo_root_from_here()
    fp = root / "annotations" / "hpi_facts" / f"{run_id}.yaml"
    return fp if fp.exists() else None

def _parse_annotations_yaml(path: Path) -> List[Dict[str, str]]:
    """
    Expect a YAML list of:
      - code: "D2.ON-01"
        evidence_span: "verbatim quote from HPI"
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a YAML list of items.")
    out: List[Dict[str, str]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path} item #{idx+1} must be a mapping.")
        code = (item.get("code") or "").strip()
        ev = item.get("evidence_span")
        if not code:
            raise ValueError(f"{path} item #{idx+1} missing 'code'.")
        if not isinstance(ev, str):
            raise ValueError(f"{path} item #{idx+1} 'evidence_span' must be a string.")
        out.append({"code": code, "evidence": ev.strip()})
    return out

def _normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _pairwise_score(a: str, b: str) -> float:
    return fuzz.ratio(_normalize_text(a), _normalize_text(b)) / 100.0

def _greedy_match_by_code(preds, golds, threshold: float):
    """
    preds: [(pred_idx, pred_ev)], golds: [(gold_idx, gold_ev)]
    return [(pred_idx, gold_idx, sim), ...] greedily by highest similarity
    """
    scored = []
    for i, pev in preds:
        for j, gev in golds:
            sim = _pairwise_score(pev, gev)
            if sim >= threshold:
                scored.append((sim, i, j))
    scored.sort(reverse=True)
    used_p, used_g = set(), set()
    pairs = []
    for sim, i, j in scored:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        pairs.append((i, j, sim))
    return pairs

def _compute_code_only_counts(pred_codes: List[str], gold_codes: List[str]):
    from collections import Counter
    pc = Counter(pred_codes); gc = Counter(gold_codes)
    tp = sum(min(pc[c], gc[c]) for c in set(pc) | set(gc))
    fp = len(pred_codes) - tp
    fn = len(gold_codes) - tp
    return tp, fp, fn

def _safe_prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f

def _evaluate_hpi_against_annotations(run_id: Optional[str], hpifacts: List[HPIFact]) -> Dict[str, Any]:
    """
    Compare NOTE-extracted HPIFacts to annotations/hpi_facts/{run_id}.yaml.
    Structure mirrors gold_fact_extract()'s eval so dashboard code can reuse it.
    """
    ann_path = _find_hpi_annotation_file(run_id)
    if not ann_path:
        return {"found": False, "reason": "annotation_file_not_found", "run_id": run_id}

    gold_items = _parse_annotations_yaml(ann_path)

    preds = []
    for f in hpifacts or []:
        code = (getattr(f, "code", None) or "").strip()
        ev = getattr(getattr(f, "evidence_span", None), "text", None)
        ev = (ev or "").strip()
        if code:
            preds.append({"id": f.id, "code": code, "evidence": ev})

    golds = [{"code": g["code"].strip(), "evidence": (g["evidence"] or "").strip()} for g in gold_items]

    from collections import defaultdict
    by_code_pred, by_code_gold = defaultdict(list), defaultdict(list)
    for i, pr in enumerate(preds):
        by_code_pred[pr["code"]].append((i, pr["evidence"]))
    for j, gd in enumerate(golds):
        by_code_gold[gd["code"]].append((j, gd["evidence"]))

    matches = []
    for code in set(by_code_pred) | set(by_code_gold):
        matches.extend(_greedy_match_by_code(by_code_pred.get(code, []), by_code_gold.get(code, []), _SIM_THRESHOLD))

    tp = len(matches)
    fp = len(preds) - tp
    fn = len(golds) - tp
    precision, recall, f1 = _safe_prf(tp, fp, fn)

    avg_sim = (sum(sim for _, _, sim in matches) / tp) if tp else 0.0

    tp_c, fp_c, fn_c = _compute_code_only_counts([p["code"] for p in preds], [g["code"] for g in golds])
    p_c, r_c, f_c = _safe_prf(tp_c, fp_c, fn_c)

    sample_matches = []
    for (pi, gj, sim) in sorted(matches, key=lambda x: x[2], reverse=True)[:50]:
        sample_matches.append({
            "pred_id": preds[pi]["id"],
            "code": preds[pi]["code"],
            "pred_evidence": preds[pi]["evidence"],
            "gold_evidence": golds[gj]["evidence"],
            "similarity": round(sim, 4),
        })

    return {
        "found": True,
        "run_id": run_id,
        "annotation_path": str(ann_path),
        "n_pred": len(preds),
        "n_gold": len(golds),
        "strict": {
            "threshold": _SIM_THRESHOLD,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "avg_similarity": avg_sim,
            "matches_preview": sample_matches,
        },
        "code_only": {
            "tp": tp_c, "fp": fp_c, "fn": fn_c,
            "precision": p_c, "recall": r_c, "f1": f_c,
        },
    }


def note_fact_extract(state: PipelineState) -> PipelineState:
    """
    1) Read omission_framework.yaml and pass it to the LLM with:
         - the current problem list (state.problems) with ids+names
    2) Expect a JSON **list** of HPIFact-like dicts (see prompt).
    3) Convert each to the runtime HPIFact dataclass.
    4) Append to state.note_facts and return state.
    5) NEW: Evaluate against annotations/hpi_facts/<run_id>.yaml and stash in state.metrics['note_eval'].
    """
    hpi_note = (state.hpi_input or "").strip()
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

    # NEW: metrics for note_fact_extract vs annotations/hpi_facts
    try:
        eval_blob = _evaluate_hpi_against_annotations(state.run_id, hpifacts)
    except Exception as e:
        eval_blob = {"found": False, "error": str(e), "run_id": state.run_id}

    state.metrics = state.metrics or {}
    state.metrics["note_eval"] = eval_blob

    return state
