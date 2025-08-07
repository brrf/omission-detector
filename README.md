# Smart Diarizer & CSV Pre-Processor

This project provides a “smart” speaker-diarization loader for free-form medical transcripts, backed by fast regex/span-based rules and an LLM fallback for edge cases. It also includes a standalone CSV pre-processor to add a `diarized_transcript` column to your data.

## Quickstart

### 1. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate 

### 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

### 3. Configure your OpenAI API key
Create a file named .env in the project root with:
OPENAI_API_KEY=sk-...

### 4. Run the CSV pre-processor
Reads your source CSV (with Transcript, HPI, Clinical feedback) and writes a new file with an extra diarized_transcript column:

python scripts/preprocess_csv.py \
  --in  raw_data/omissions_sample_encounters.csv \
  --out raw_data/omissions_sample_encounters_diarized.csv

## Pipeline workflow (LangGraph)

**Graph:** START → GoldFacts → BucketLabeler → Scorer → Prioritizer → MetricOmitter → END  
with a parallel **NoteFacts** branch that also feeds **BucketLabeler**.

- **GoldFacts** (1.2): Transcript → HPI facts (structured)
- **NoteFacts** (1.3): Note HPI → HPI facts (structured)
- **BucketLabeler**: Enumerate problems and assign `problem_id` to each fact
- **Scorer** (1.4): Problem-aware matching & omission classification per transcript fact
- **Prioritizer** (1.5): Compute Materiality and flag actionables
- **MetricOmitter**: Final side-effects (emit/queue metrics)

### Fact schema (per HPI fact)

- `code`, `label`
- `polarity`: present | denied | uncertain
- `value`: normalized content
- `salience_weight`: 3/1/0 (from taxonomy)
- `problem_id`: assigned post clustering (e.g., “LE edema/pain”)
- `time_scope`: acute | chronic | baseline | changed
- `evidence_span`: short quote + character offsets
- `hpi_in_scope`: boolean (map-driven)

### Classifier output (per transcript fact)

- `status`: present | omitted | conflict | out_of_scope
- `recommended_inclusion`: HPI | elsewhere | not_relevant
- Prioritization adds `materiality` and `priority_label` (high/medium/low)