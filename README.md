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