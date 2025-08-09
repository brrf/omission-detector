# HPI Labeling & Omission Scoring Guide (for MD Reviewers)

Purpose: Create reliable, clinician-useful labels from a transcript, label the HPI note, then score any omissions/conflicts so they reflect what clinicians truly care about.

## Deliverables

Produce four files in this repo (YAML):

1. transcript_facts.yaml — atomic facts from the transcript

2. hpi_facts.yaml — atomic facts from the HPI note

3. hpi_omissions.yaml — clinically meaningful transcript facts missing from the HPI (or contradicted)

4. omission_scoring.yaml — scoring for each omitted/conflict fact (priority + materiality)

Facts format constraint: Every fact item must have exactly these keys:
code, evidence_span, rationale
(No weights or extra keys in these three files.)

## Quick Start

Read the transcript end-to-end.

Extract atomic facts (one concept each) and assign one best taxonomy code.

Label the HPI the same way.

Compare transcript vs HPI → collect omissions/conflicts that matter today.

Score each omitted/conflict fact → compute materiality and priority.

## Atomic Facts (what to extract)

One clinical concept per item. Split multi-concept sentences.

✔ “Progesterone 200 mg nightly.”

✔ “Estradiol patch 0.0375 mg twice weekly.”

✔ “Bleeding persists after dose change.” → actually two facts:

“Bleeding persists” (trajectory)

“Estradiol dose decreased from 0.05 to 0.0375” (med change)

Normalize when stated: dose/route/frequency, sidedness, sizes, and dates/durations.
Evidence spans are verbatim quotes (minimal but unambiguous).

Weights come from the framework YAML. Typical scale: 3 = high, 1 = contextual, 0 = low. Unknown codes default to 0.

## Labeling the Transcript and HPI

Build transcript_facts.yaml by walking the transcript top-to-bottom.

Build hpi_facts.yaml from the HPI note with the same rules.

Each list item: code, evidence_span, rationale. (See templates below.)

### Finding Omissions & Conflicts

Omission: transcript fact absent from the HPI (semantic coverage counts; if HPI has the drug but misses the dose, the dose is an omission).

Conflict: HPI contradicts the transcript (e.g., different dose).

Only include omissions that clinically matter today for the active problem.

Partial matches: Treat each component (drug name, dose, route, frequency, duration) as separable atomic facts.

### Scoring Omissions/Conflicts

For every item in hpi_omissions.yaml, add a corresponding entry in omission_scoring.yaml.

#### Materiality formula

**materiality =
  taxonomy_weight + recency_bonus + linkage_bonus - redundancy_penalty + active_today_bonus + prechart_bonus**

- taxonomy_weight = weight for the fact’s code from the framework (int; usually 3/1/0).
- recency_bonus = +1 if recent (≤14 days) or acute (clearly new episode / marked worsening).
- linkage_bonus = +1 if highly_relevant (including this would plausibly change today’s assessment/workup/plan).
- redundancy_penalty = −1 if redundant (already documented in HPI in substance, even if phrased differently).
- active_today_bonus = +1 if linked to a problem active today (chief concern or a problem marked active).
- prechart_bonus = +3 only if the omitted fact came from pre‑chart (entered before the visit) and is omitted in the note.

#### Priority label (from materiality)

- high ≥ 4
- medium ≥ 3
- low ≥ 2
- < 2 → no priority (None)

#### Transcript/HPI Facts (same structure)

```yaml
code: "<taxonomy code or UNMAPPED>"
  evidence_span: {{"<verbatim quote from source>"}}
  rationale: {{"<why this code fits / why clinically relevant>"}}
```

#### HPI Omissions (facts from transcript missing in HPI)

```yaml
- code: "<taxonomy code>"
  evidence_span: {{"<verbatim quote from transcript>"}}
  rationale: {{"<why omission matters clinically for today's HPI>"}}
```

#### Omission Scoring (one entry per omitted/conflict fact)

```yaml
- status: "omitted"  # or "conflict"
  code: "<taxonomy code>"
  taxonomy_weight: 3
  redundant: false
  recent: true
  acute: false
  highly_relevant: true
  active_today: true
  prechart: false
  materiality: 6
  priority: "high"
  reason: "Recent change that alters workup/plan; not captured in HPI."
```

#### Mini Worked Example

Transcript fact (omitted in HPI):

“I just had such a horrific experience with the last one.”

Labeled fact: code: D2.CHAR-04 (symptom character)

If “last one” refers to an event within 14 days → recent: true

Impacts today’s plan (e.g., choice of agent/approach) → highly_relevant: true

Not documented in HPI → redundant: false

Active problem → active_today: true

Scoring math: 3 (taxonomy_weight) +1 (recent/acute) +1 (highly_relevant) +1 (active_today) = 6 → priority: high