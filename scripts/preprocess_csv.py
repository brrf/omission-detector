#!/usr/bin/env python
"""
scripts/preprocess_csv.py
-------------------------
Reads a CSV that has at least the columns
    Encounter ID | Transcript | HPI | Clinical feedback
and writes a new file with an additional
    diarized_transcript
column containing the role-labelled transcript.

Usage
-----
$ python scripts/preprocess_csv.py \
      --input  raw_data/omissions_sample_encounters.csv \
      --output raw_data/omissions_sample_encounters_diarized.csv
"""
import argparse
import csv
import pathlib
from tqdm import tqdm

from loaders.parse_diarized_text import diarize_text

def main(input_csv: pathlib.Path, output_csv: pathlib.Path) -> None:
    rows = list(csv.DictReader(input_csv.open()))
    if not rows:
        raise SystemExit(f"No rows found in {input_csv}")

    out_f = output_csv.open("w", newline="", encoding="utf-8")
    writer = None

    for row in tqdm(rows, desc="Diarizing", unit="encounter"):
        conv = diarize_text(row["Transcript"], conv_id=row["Encounter ID"])

        # ── existing diarized transcript ────────────────────────────────────────
        role_map = {s.id: s.role.capitalize() for s in conv.speakers.values()}
        diarized = "\n".join(f"{role_map[u.speaker_id]}: {u.text}"
                            for u in conv.utterances)
        row["diarized_transcript"] = diarized
        row["pre_visit"] = (conv.pre_visit or "").strip()

        interp_spks = [s for s in conv.speakers.values() if s.role == "interpreter"]
        print(interp_spks)
        # 1.  All interpreter numbers, semicolon‑separated
        row["interpreter_numbers"] = "; ".join(
            (spk.meta.get("interpreter_number") or spk.id.lstrip("I"))
            for spk in interp_spks
        )

        # 2.  The interpreter‑intro sentence(s) (joined if > 1)
        if interp_spks:
            interp_ids = {spk.id for spk in interp_spks}
            row["interpreter_intro"] = " | ".join(
                u.text for u in conv.utterances if u.speaker_id in interp_ids
            )
        else:
            row["interpreter_intro"] = ""

        # ── write (header is created on first row) ─────────────────────────────
        if writer is None:
            writer = csv.DictWriter(out_f, fieldnames=row.keys())
            writer.writeheader()
        writer.writerow(row)

        print(f"✅  Wrote {len(rows)} rows → {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add a diarized_transcript column to a raw CSV"
    )
    parser.add_argument(
        "-i", "--input",
        dest="input",
        required=True,
        type=pathlib.Path,
        help="Path to source CSV with raw transcripts"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output",
        required=True,
        type=pathlib.Path,
        help="Path to write CSV with diarized_transcript column"
    )
    args = parser.parse_args()
    main(args.input, args.output)
