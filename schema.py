# schema.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Domain entities
# ---------------------------------------------------------------------------

@dataclass
class Speaker:
    """
    A participant in (or transient contributor to) the encounter.

    `kind`
        • "participant" **part of the core encounter (patient, clinician)**
        • "external"    **briefly enters the room / phone / page etc.**
        • "system"      **automated message, ASR artefact, etc.**
    """
    id: str                          # “P001”, “D002”, “N1” …
    role: str                        # “patient”, “clinician”, “nurse”, “pager”
    kind: str = "participant"        # see docstring
    meta: Dict[str, str] = field(default_factory=dict)   # extensible


@dataclass
class Utterance:
    """
    A single contiguous block spoken by one speaker *without interruption*.

    We keep the segmentation exactly as produced by the transcript source.
    If token cost later requires splitting very long turns, we can insert
    synthetic Utterance IDs that all map back to the same `speaker_id`.
    """
    id: str
    speaker_id: str                  # link to Speaker.id
    text: str
    timestamp: Optional[datetime] = None
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class Conversation:
    id: str
    speakers: Dict[str, Speaker]         # keyed by Speaker.id
    utterances: List[Utterance]          # chronological order
    source: str                          # “MedDG”, “MediDialog‑En” …
    meta: Dict[str, str] = field(default_factory=dict)  # license, split, etc.
    pre_visit: str | None = None
    # ───────────────────────── Helpers ─────────────────────────
    def get_transcript(
        self,
        sep: str = "\n",
        include_kinds: set[str] | None = None,
    ) -> str:
        """
        Return a human‑readable transcript.

        `include_kinds`
            • None  – include *all* speakers (legacy behaviour)
            • {"participant"} – core medicolegal record (typical default)
            • {"participant", "external"} – include hallway interruptions, etc.
        """
        if include_kinds is None:
            include_kinds = {s.kind for s in self.speakers.values()}

        label = {s.id: (s.role or s.id) for s in self.speakers.values()
                 if s.kind in include_kinds}

        return sep.join(
            f"{label[u.speaker_id]}: {u.text}"
            for u in self.utterances
            if self.speakers[u.speaker_id].kind in include_kinds
        )

