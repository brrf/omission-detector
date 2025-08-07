"""
loaders/parse_diarized_txt.py  – LLM‑only version
-------------------------------------------------
• Splits the transcript into sentences with spaCy
• Feeds the LLM 15 sentences at a time
• Expects a pure JSON list of labels in the same order
• Merges consecutive sentences from the same speaker into a turn

Labels supported  (case‑sensitive):
    CLINICIAN, PATIENT, FAMILY, INTERPRETER, OTHER
"""
from __future__ import annotations
import pathlib, uuid, re, json, ast, threading
from typing import Dict, List, Tuple, Generator

import spacy
from llm_utils import call_llm
from schema import Conversation, Speaker, Utterance

# ───────────────────────────── config ────────────────────────────────
_NLP          = spacy.load("en_core_web_sm")       # sentence splitter
_BATCH_SIZE   = 10
_LLM_MODEL    = "gpt-4o-mini"
_SEM          = threading.Semaphore(3)             # throttle concurrent calls

_ROLE_ID = {"CLINICIAN": "D", "PATIENT": "P",
            "FAMILY": "F", "INTERPRETER": "I", "OTHER": "X"}
_KINDS   = {"CLINICIAN": "participant", "PATIENT": "participant",
            "FAMILY": "external", "INTERPRETER": "external", "OTHER": "external"}

_GREETING = re.compile(r"\b(hello|hi|hey|good\s(morning|afternoon|evening))\b", re.I)
_INVITE   = re.compile(r"\b(come\s+on\s+in|have\s+a\s+seat|sit\s+anywhere)\b", re.I)
_NOTE_HDR = re.compile(
    r"^(chief complaint|cc|hpi|ros|assessment|plan|meds?|allergies)\b[:\-–]?",
    re.I,
)
_BULLET   = re.compile(r"^\s*[-*•]\s")
# short/interactive => likely conversation
_CONVO_CLUE = re.compile(
    r"\?|(\b(?:you|your|yeah|uh‑huh|okay|hi|hello|hey)\b)",
    re.I,
)
_INTERP_RE = re.compile(
    r"\binterpreter\b\s*(?:name\s*)?(?:number\s*)?"
    r"(?P<num>\d{3,6})"                       # 3–6 digit ID
    r"(?:[^A-Za-z0-9]+(?P<name>[A-Za-zÁÉÍÓÚÑü\'\- ]+))?",
    re.I,
)
MAX_SCAN = 15      # only the first N sentences can be pre-charting

_PROMPT = """
You are a medical documentation assistant.
For each sentence below return **one** label, chosen from:
CLINICIAN, PATIENT

Return *only* a valid JSON list, e.g.:
["CLINICIAN", "PATIENT", "CLINICIAN"]

The sentences are numbered. Your response should have the same length as the number of sentences in the prompt.
If the list has a different length I will ask you to try again. 

Sentences:
{block}
""".strip()

# ───────────────────────── helpers ───────────────────────────────────
def _split_sentences(text: str) -> List[str]:
    """Using spaCy for robust sentence detection."""
    return [s.text.strip() for s in _NLP(text).sents if s.text.strip()]

def _is_note_like(sent: str) -> bool:
    """Heuristics for EMR-style notes."""
    longish = len(sent) > 15
    return _NOTE_HDR.match(sent) or _BULLET.match(sent) or longish and not _CONVO_CLUE.search(sent)

def split_pre_charting(text: str) -> tuple[str, str]:
    """
    Return `(dialogue, pre_visit)`.

    Break as soon as we hit **any** of:
        • greeting (“Hi …”, “Good morning …”)
        • invitation (“Come on in …”, “Have a seat …”)
        • generic conversational clues (question marks, fillers, 2-person pronouns)

    Up to `MAX_SCAN` early sentences are examined for potential note-like text.
    """
    sents = _split_sentences(text)
    pre   = []

    for idx, s in enumerate(sents[:MAX_SCAN]):
        if _GREETING.search(s) or _INVITE.search(s) or _CONVO_CLUE.search(s):
            # first conversational sentence → stop accumulating
            return "\n".join(sents[idx:]), "\n".join(pre)

        if _is_note_like(s):
            pre.append(s)
        else:
            # unlikely note but also no greeting/clue → treat as dialogue
            return "\n".join(sents[idx:]), "\n".join(pre)

    # nothing looked like a note ⇒ keep whole text
    return text, ""

def _extract_interpreter(
    sents: list[str], *,
    max_scan: int = 6,
) -> tuple[list[str], Optional[Speaker], list[Utterance]]:
    """
    Scan the first `max_scan` sentences for an interpreter intro.
    Returns a tuple of:
        • the sentence list with the interpreter line(s) removed
        • a Speaker object (or None if not found)
        • a list with one Utterance for that intro (or empty)

    We remove the intro sentence so it never reaches the LLM labeler.
    """
    for idx, s in enumerate(sents[:max_scan]):
        m = _INTERP_RE.search(s)
        if m:
            num  = m.group("num")
            name = (m.group("name") or "").strip() or f"Interpreter {num}"
            spk  = Speaker(
                id=f"I{num}",          # unique but still “I…”
                role="interpreter",
                kind="external",
                meta={"interpreter_number": num},
            )
            utt  = Utterance(
                id=f"I{num}_0000",
                speaker_id=spk.id,
                text=s.strip(),
            )
            # strip the intro sentence
            clean_sents = sents[:idx] + sents[idx+1:]
            return clean_sents, spk, [utt]
    return sents, None, []


def _llm_label(batch: List[str]) -> List[str]:
    # Replace hard line-breaks inside each sentence with a space
    clean = [s.replace("\n", " ").strip() for s in batch]
    block  = "\n".join(f"{i+1}. {s}" for i, s in enumerate(clean))
    prompt = _PROMPT.format(block=block)
    with _SEM:
        raw = call_llm(_LLM_MODEL, prompt)
    try:
        labels = json.loads(raw)
    except json.JSONDecodeError:
        labels = ast.literal_eval(raw)  # fallback

    if not isinstance(labels, list) or len(labels) != len(batch):
        print("this does happen")
        print(batch, labels, prompt)
        raise RuntimeError(f"Unexpected LLM response:\n{raw}")

    return [lbl.upper() for lbl in labels]

def _tag_sentences(sents: List[str]) -> List[str]:
    """Batch sentences through the LLM."""
    tags: List[str] = []
    for i in range(0, len(sents), _BATCH_SIZE):
        batch = sents[i:i + _BATCH_SIZE]
        for _ in range(2):                 # 1 retry max
            labels = _llm_label(batch)
            if len(labels) == len(batch):
                tags.extend(labels)
                break
            print("this never happens")
    return tags

def _merge_turns(sents: List[str], tags: List[str]) -> List[Tuple[str, str]]:
    turns, buf, cur = [], [], tags[0]
    for sent, lbl in zip(sents, tags):
        if lbl != cur:
            turns.append((cur, " ".join(buf)))
            buf, cur = [sent], lbl
        else:
            buf.append(sent)
    if buf:
        turns.append((cur, " ".join(buf)))
    return turns

def _build_conversation(turns: List[Tuple[str, str]], cid: str, pre_visit: str = "") -> Conversation:
    speakers: Dict[str, Speaker] = {}
    utts: List[Utterance] = []

    for idx, (lbl, txt) in enumerate(turns):
        sid = _ROLE_ID.get(lbl, "X")
        speakers.setdefault(sid, Speaker(id=sid, role=lbl.lower(), kind=_KINDS[lbl]))
        utts.append(Utterance(id=f"{cid}_{idx:04d}", speaker_id=sid, text=txt))

    return Conversation(id=cid, speakers=speakers, utterances=utts,
                        source="FREEFORM_TXT_DIARIZED", meta={}, pre_visit=pre_visit or None)

# ───────────────────────── public API ────────────────────────────────
def diarize_text(text: str, conv_id: str | None = None) -> Conversation:
    cid   = conv_id or str(uuid.uuid4())
    main_txt, pre_visit = split_pre_charting(text)
    sents0 = _split_sentences(main_txt)
    sents, interp_spk, interp_utts = _extract_interpreter(sents0)
    tags  = _tag_sentences(sents)
    turns = _merge_turns(sents, tags)
    conv = _build_conversation(turns, cid, pre_visit=pre_visit)

    # splice in the interpreter (at the top) if present
    if interp_spk:
        conv.speakers[interp_spk.id] = interp_spk
        conv.utterances = interp_utts + conv.utterances

    return conv


def parse_diarized_txt(path: pathlib.Path) -> Generator[Conversation, None, None]:
    """File‑based entry for dataset builder, etc."""
    files = [path] if path.is_file() else list(path.glob("*.txt"))
    for fp in files:
        txt = fp.read_text(encoding="utf-8", errors="ignore")
        yield diarize_text(txt, conv_id=(fp.stem or str(uuid.uuid4())))
