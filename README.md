# Omission Detector
## Smart diarizer + HPI omission detection and scoring (LangGraph)

### Quick Start
Prereqs: Python 3.10–3.12, OPENAI_API_KEY, access to gpt-4o-mini (optional gpt-4o).

#### Setup:
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pyyaml
python -m spacy download en_core_web_sm
cp .env.example .env  # then edit .env and set OPENAI_API_KEY=sk-...
```

#### Diarize CSV:
```
python scripts/preprocess_csv.py \
  --input  raw_data/omissions_sample_encounters.csv \
  --output raw_data/omissions_sample_encounters_diarized.csv
```

#### Run Pipeline:
```
python run_all.py \
  --input  raw_data/omissions_sample_encounters_diarized.csv \
  --output out/pipeline_results.jsonl
```

## How It Works (end‑to‑end)
1) Diarization (scripts/preprocess_csv.py → loaders/parse_diarized_text.py)
Splits the raw transcript into sentences (spaCy en_core_web_sm).

- Pre‑chart detection: scans the first 15 sentences; if a sentence looks like an EMR note (headers, bullets, long declarative text without dialog clues) before the first conversational cue (greeting, invitation, question marks, 2nd‑person pronouns), it’s lifted into pre_visit.
- Interpreter detection (first 6 sentences): looks for interpreter + 3–6 digit number and optional name.
- LLM labeling (gpt‑4o‑mini) in small batches of 10 sentences, returning CLINICIAN or PATIENT per sentence.
- Consecutive sentences with the same label are merged into turns.
- Output is a Conversation dataclass, rendered back into diarized_transcript like:

2) LangGraph pipeline (pipeline/agent_pipeline.py)
Graph (simplified):

```
START
  ↓
GoldFacts (transcript + pre_chart → HPIFacts)
  ↓
ProblemLabeler (cluster facts into named problems, assign problem_id)
  ↘
    NoteFactExtract (HPI note → HPIFacts, aligned to problem_id when possible)
      ↓
    OmissionsDetector (match transcript facts to note facts by code + problem)
      ↓
    OmissionFilter (semantic filter vs original HPI; keep only true omissions)
      ↓
    Scorer (weights + bonuses → materiality & priority)
      ↓
    MetricOmitter (persist JSONL/CSV & by_run blobs)
  ↓
END
```
### Agents in one line each

- GoldFacts — single prompt extracts atomic HPI facts from both transcript and pre‑chart, tagging pre‑chart facts with isPrechartFact=true.

- ProblemLabeler — builds a small, meaningful problem list (“Left knee pain”, “Migraine flare”) and assigns each fact to exactly one problem (stable per‑run IDs).

- NoteFactExtract — parses the HPI note into facts with the same taxonomy (optionally adds time_scope).

- OmissionsDetector — for each transcript fact, looks for note facts with the same taxonomy code and (strictly) the same problem_id (or allows a soft match if there’s only one problem); asks an LLM to decide present / omitted / conflict using semantic leniency.

- SemanticOmissionFilter — compares alleged omissions to the original HPI text; drops items that are semantically present or not HPI‑relevant.

- Scorer — computes materiality from:
taxonomy weight (from framework/omission_framework.yaml)
bonuses: recent/acute, highly_relevant, active_today
pre‑chart omission bonus: +3 when a pre‑chart fact was omitted
penalties: redundancy

Priority bands: high ≥ 4, medium ≥ 3, low ≥ 2.

- MetricOmitter — appends rows to out/omissions.jsonl + out/omissions.csv and writes a compact per‑run JSON.

## Taxonomy & Configuration
- framework/omission_framework.yaml drives the taxonomy (D1–D9) and weights used in scoring.
- framework/adult_hpi_scope.yaml and framework/peds_hpi_scope.yaml sketch inclusion scopes (e.g., “always”, “conditional”). They’re not wired into scoring yet but show the intended direction (adult vs peds profiles).
- You can evolve the taxonomy or weights without touching code; scorer.py reads weights directly from omission_framework.yaml.

## Design Choices
- LLM‑first, rules‑assisted: simple heuristics handle pre‑chart/interpretation extraction; everything else is small, single‑responsibility prompts.
- Deterministic IDs: facts get stable UUIDs from (code, polarity, value) so duplicates dedupe cleanly across steps.
- Problem‑aware matching: comparing facts within the same problem avoids cross‑talk between unrelated complaints.
- Two passes. First to maximize omission detection, second to filter based on what clinicians care about
