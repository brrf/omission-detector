# pipeline/agents/scorer.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.state import PipelineState, HPIFact, FactClassification, Problem
from llm_utils import call_llm

# We parse the framework YAML to get code -> weight.
# PyYAML is small and reliable; added to requirements.txt below.
import yaml
from rapidfuzz import fuzz

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


# ---------------- prioritized omissions gold eval helpers ----------------

_PRIOR_SIM_THRESHOLD = 0.80  # match on evidence text within code bucket

def _repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        if (base / "annotations").exists() and (base / "framework").exists():
            return base
    return Path.cwd()

def _find_prioritized_annotation_file(run_id: Optional[str]) -> Optional[Path]:
    if not run_id:
        return None
    root = _repo_root_from_here()
    fp = root / "annotations" / "prioritized_omissions" / f"{run_id}.yaml"
    return fp if fp.exists() else None

def _parse_prioritized_yaml(path: Path) -> List[Dict[str, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a YAML list.")
    out: List[Dict[str, str]] = []
    for i, it in enumerate(data):
        if not isinstance(it, dict):
            raise ValueError(f"{path} item #{i+1} must be a mapping.")
        code = (it.get("code") or "").strip()
        ev = it.get("evidence_span")
        if not code:
            raise ValueError(f"{path} item #{i+1} missing 'code'.")
        if not isinstance(ev, str):
            raise ValueError(f"{path} item #{i+1} 'evidence_span' must be a string.")
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
    -> [(pred_idx, gold_idx, sim)]
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

def _evaluate_prioritized_omissions_against_annotations(
    run_id: Optional[str],
    classifications: List[FactClassification],
    transcript_facts: List[HPIFact],
) -> Dict[str, Any]:
    """
    Compare FINAL kept omissions (post semantic filter & scoring),
    restricted to items with status == 'omitted', against
    annotations/prioritized_omissions/<run_id>.yaml.
    """
    ann_path = _find_prioritized_annotation_file(run_id)
    if not ann_path:
        return {"found": False, "reason": "annotation_file_not_found", "run_id": run_id}

    gold_items = _parse_prioritized_yaml(ann_path)

    # Build predictions from classifications (omitted only) -> map to transcript fact
    by_id = {f.id: f for f in (transcript_facts or [])}
    preds: List[Dict[str, str]] = []
    for cls in (classifications or []):
        if getattr(cls, "status", None) != "omitted":
            continue
        tf = by_id.get(cls.fact_id)
        if not tf:
            continue
        code = (tf.code or "").strip()
        ev = (getattr(getattr(tf, "evidence_span", None), "text", "") or "").strip()
        if code:
            preds.append({"id": tf.id, "code": code, "evidence": ev})

    golds = [{"code": (g["code"] or "").strip(), "evidence": (g["evidence"] or "").strip()} for g in gold_items]

    # Group by code
    from collections import defaultdict
    by_code_pred, by_code_gold = defaultdict(list), defaultdict(list)
    for i, pr in enumerate(preds):
        by_code_pred[pr["code"]].append((i, pr["evidence"]))
    for j, gd in enumerate(golds):
        by_code_gold[gd["code"]].append((j, gd["evidence"]))

    matches = []
    for code in set(by_code_pred) | set(by_code_gold):
        matches.extend(_greedy_match_by_code(by_code_pred.get(code, []), by_code_gold.get(code, []), _PRIOR_SIM_THRESHOLD))

    tp = len(matches)
    fp = len(preds) - tp
    fn = len(golds) - tp
    precision, recall, f1 = _safe_prf(tp, fp, fn)
    avg_sim = (sum(sim for _, _, sim in matches) / tp) if tp else 0.0

    # Code-only
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
            "threshold": _PRIOR_SIM_THRESHOLD,
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


# ---------------- core agent ----------------

def scorer(state: PipelineState) -> PipelineState:
    """
    Input: state.classifications, state.transcript_facts, state.note_facts, state.problems
    Output:
      - updates each FactClassification.materiality (int)
      - sets priority_label ("high"/"medium"/"low"/None)
      - populates state.prioritized with items having materiality >= 2 (omitted/conflict)
      - NEW: adds state.metrics['prioritized_eval'] comparing FINAL omissions vs gold
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
        prechart_bonus = 0  # NEW: extra weight for omitted pre-chart facts

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

            # NEW: pre-chart omitted facts get +3 materiality
            if cls.status == "omitted" and bool(getattr(tf, "isPrechartFact", False)):
                prechart_bonus = 3

        # 4) Materiality score
        materiality = taxonomy_weight + recency_bonus + linkage_bonus - redundancy_penalty + active_today_bonus + prechart_bonus

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

    # 7) NEW: Evaluate FINAL omissions (status == 'omitted') vs gold prioritized_omissions
    try:
        prior_eval = _evaluate_prioritized_omissions_against_annotations(
            state.run_id, state.classifications, state.transcript_facts
        )
    except Exception as e:
        prior_eval = {"found": False, "error": str(e), "run_id": state.run_id}

    state.metrics = state.metrics or {}
    state.metrics["prioritized_eval"] = prior_eval

    return state
