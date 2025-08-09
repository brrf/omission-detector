# pipeline/agents/gold_fact_extract.py
from __future__ import annotations

import json
import hashlib
import uuid
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from rapidfuzz import fuzz

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

# ===================== framework helpers =====================

def _read_framework_yaml() -> str:
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

# ===================== GOLD EVAL (clean + strict) =====================

_SIM_THRESHOLD = 0.80  # RapidFuzz ratio (0..1) cutoff for evidence span match

def _repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        if (base / "annotations" / "gold_facts").exists() and (base / "framework").exists():
            return base
    return Path.cwd()

def _find_annotation_file(run_id: Optional[str]) -> Optional[Path]:
    if not run_id:
        return None
    root = _repo_root_from_here()
    fp = root / "annotations" / "gold_facts" / f"{run_id}.yaml"
    return fp if fp.exists() else None

def _parse_annotations_yaml(path: Path) -> List[Dict[str, str]]:
    """
    Strict parser. Expect a YAML list of items:
      - code: "D2.ON-01"
        evidence_span: "verbatim quote"
    Returns [{"code": str, "evidence": str}, ...].
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
            raise ValueError(
                f"{path} item #{idx+1} 'evidence_span' must be a string. "
                "Please store the exact quote as a plain string."
            )
        out.append({"code": code, "evidence": ev.strip()})
    return out

def _normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _pairwise_score(a: str, b: str) -> float:
    return fuzz.ratio(_normalize_text(a), _normalize_text(b)) / 100.0  # [0,1]

def _greedy_match_by_code(
    preds: List[Tuple[int, str]], golds: List[Tuple[int, str]], threshold: float
) -> List[Tuple[int, int, float]]:
    """
    Greedy matching within a single code bucket.
    preds: [(pred_idx, pred_ev)], golds: [(gold_idx, gold_ev)]
    Returns list of (pred_idx, gold_idx, sim).
    """
    scored: List[Tuple[float, int, int]] = []
    for i, pev in preds:
        for j, gev in golds:
            sim = _pairwise_score(pev, gev)
            if sim >= threshold:
                scored.append((sim, i, j))
    scored.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    pairs: List[Tuple[int, int, float]] = []
    for sim, i, j in scored:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        pairs.append((i, j, sim))
    return pairs

def _compute_code_only_counts(pred_codes: List[str], gold_codes: List[str]) -> Tuple[int, int, int]:
    from collections import Counter
    pc = Counter(pred_codes)
    gc = Counter(gold_codes)
    tp = sum(min(pc[c], gc[c]) for c in set(pc) | set(gc))
    fp = len(pred_codes) - tp
    fn = len(gold_codes) - tp
    return tp, fp, fn

def _safe_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f

def _evaluate_against_annotations(run_id: Optional[str], hpifacts: List[HPIFact]) -> Dict[str, Any]:
    """
    Compare extracted HPIFacts to 'annotations/gold_facts/{run_id}.yaml'.
    Returns a JSON-serializable dict with counts + metrics + sample matches.
    """
    ann_path = _find_annotation_file(run_id)
    if not ann_path:
        return {"found": False, "reason": "annotation_file_not_found", "run_id": run_id}

    gold_items = _parse_annotations_yaml(ann_path)

    # predictions (code + evidence)
    preds: List[Dict[str, Any]] = []
    for f in hpifacts or []:
        code = (getattr(f, "code", None) or "").strip()
        ev = getattr(getattr(f, "evidence_span", None), "text", None)
        ev = (ev or "").strip()
        if code:
            preds.append({"id": f.id, "code": code, "evidence": ev})

    golds: List[Dict[str, str]] = [{"code": g["code"].strip(), "evidence": (g["evidence"] or "").strip()} for g in gold_items]

    # group by code and match greedily by similarity
    from collections import defaultdict
    by_code_pred: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    by_code_gold: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    for i, pr in enumerate(preds):
        by_code_pred[pr["code"]].append((i, pr["evidence"]))
    for j, gd in enumerate(golds):
        by_code_gold[gd["code"]].append((j, gd["evidence"]))

    matches: List[Tuple[int, int, float]] = []
    for code in set(by_code_pred) | set(by_code_gold):
        matches.extend(_greedy_match_by_code(by_code_pred.get(code, []), by_code_gold.get(code, []), _SIM_THRESHOLD))

    tp = len(matches)
    fp = len(preds) - tp
    fn = len(golds) - tp
    precision, recall, f1 = _safe_prf(tp, fp, fn)

    avg_sim = (sum(sim for _, _, sim in matches) / tp) if tp else 0.0

    tp_c, fp_c, fn_c = _compute_code_only_counts([p["code"] for p in preds], [g["code"] for g in golds])
    p_c, r_c, f_c = _safe_prf(tp_c, fp_c, fn_c)

    sample_matches: List[Dict[str, Any]] = []
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

# ===================== public agent =====================

def gold_fact_extract(state: PipelineState) -> PipelineState:
    """
    1) Read omission_framework.yaml.
    2) Extract facts from:
         a) state.transcript  -> isPrechartFact = False
         b) state.pre_chart   -> isPrechartFact = True
       using the SAME prompt.
    3) Concatenate results (NO de-duplication) into state.transcript_facts.
    4) Evaluate against annotations/gold_facts/<run_id>.yaml (if present)
       and store metrics in state.metrics['gold_eval'].
    """
    framework_yaml = _read_framework_yaml()

    out: List[HPIFact] = []

    # (a) Transcript facts (if present)
    transcript_text = (state.transcript or "").strip()
    if transcript_text:
        out.extend(_extract_facts(framework_yaml, transcript_text, is_prechart=False))

    # (b) Pre-chart facts (if present)
    prechart_text = (state.pre_chart or "").strip() if isinstance(state.pre_chart, str) else ""
    if prechart_text:
        out.extend(_extract_facts(framework_yaml, prechart_text, is_prechart=True))

    state.transcript_facts = out

    # (c) Optional evaluation vs annotations
    try:
        eval_blob = _evaluate_against_annotations(state.run_id, out)
    except Exception as e:
        eval_blob = {"found": False, "error": str(e), "run_id": state.run_id}

    state.metrics = state.metrics or {}
    state.metrics["gold_eval"] = eval_blob

    return state
