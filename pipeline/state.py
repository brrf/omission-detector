# pipeline/state.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Annotated, TypeVar, Literal
from datetime import datetime

from schema import Conversation

# ---------- Merge helpers ----------
def merge_fact_maps(old: Dict[str, str], new: Dict[str, str]) -> Dict[str, str]:
    merged = {**old}
    merged.update(new)
    return merged

T = TypeVar('T')

def merge_union_seq(list1: List[T], list2: List[T]) -> List[T]:
    """
    Union for lists that may contain unhashable items (e.g., dataclasses).
    Preserves order of first appearance.
    """
    result: List[T] = list(list1)
    for item in list2:
        if item not in result:
            result.append(item)
    return result

# ---------- Domain for extraction/classification ----------
@dataclass
class EvidenceSpan:
    text: str

@dataclass
class HPIFact:
    id: str                         # unique per fact (string UUID or deterministic hash)
    code: str                       # e.g., "D2.ON-01"
    polarity: Literal["present", "denied", "uncertain"]
    value: str                      # normalized content (e.g., "onset: 7-10 days ago")
    problem_id: Optional[str] = None  # assigned by ProblemLabeler
    time_scope: Optional[Literal["acute", "chronic", "baseline", "changed"]] = None
    evidence_span: Optional[EvidenceSpan] = None


@dataclass
class Problem:
    id: str              # stable per run
    name: str            # e.g., "LE edema/pain"
    active_today: bool = True

@dataclass
class FactClassification:
    fact_id: str
    status: Literal["present", "omitted", "conflict", "out_of_scope"]
    recommended_inclusion: Literal["HPI", "elsewhere", "not_relevant"]
    materiality: float = 0.0
    priority_label: Optional[Literal["high", "medium", "low"]] = None

# ---------- Pipeline state ----------
@dataclass
class PipelineState:
    # Core identifiers / inputs
    run_id:        Annotated[str,           lambda x, y: y]
    transcript:    Annotated[str,           lambda x, y: y]
    note_baseline: Annotated[Optional[str], lambda x, y: y] = None

    # Exact fields provided by run_all.py
    source:      Annotated[Optional[str],            lambda x, y: y] = None
    pre_chart:   Annotated[Optional[str],            lambda x, y: y] = None
    interpreter: Annotated[Optional[Dict[str, Any]], lambda x, y: y] = None
    hpi_input:   Annotated[Optional[str],            lambda x, y: y] = None

    # Structured results
    transcript_facts: Annotated[List[HPIFact],       merge_union_seq] = field(default_factory=list)
    note_facts:       Annotated[List[HPIFact],       merge_union_seq] = field(default_factory=list)
    problems:         Annotated[List[Problem],       merge_union_seq] = field(default_factory=list)

    classifications:  Annotated[List[FactClassification], merge_union_seq] = field(default_factory=list)
    prioritized:      Annotated[List[FactClassification], merge_union_seq] = field(default_factory=list)

    metrics:          Annotated[Dict[str, Any],      lambda x, y: y]  = field(default_factory=dict)
