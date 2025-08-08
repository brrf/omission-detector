#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_all.py
----------
Reads diarized.csv and invokes your LangGraph pipeline per encounter.

Per row, we pass:
- run_id: from "Encounter ID"
- transcript: from "diarized_transcript" (falls back to "Transcript" if needed)
- pre_chart: from "pre_visit"
- interpreter: {numbers: [...], intro: "..."} from "interpreter_numbers"/"interpreter_intro"
- hpi_input: from "HPI" (the generated HPI to evaluate/compare)

Outputs a JSONL with one line per run: {run_id, status, output|error}.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from pipeline.agent_pipeline import run_pipeline


# ----- Config for CSV headers (exact names from your spec) -----
COL_ENCOUNTER_ID = "Encounter ID"
COL_TRANSCRIPT = "Transcript"
COL_HPI = "HPI"
COL_FEEDBACK = "Clinical feedback"
COL_DIARIZED = "diarized_transcript"
COL_PREVISIT = "pre_visit"
COL_INTERP_NUM = "interpreter_numbers"
COL_INTERP_INTRO = "interpreter_intro"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pipeline across diarized.csv")
    parser.add_argument(
        "--input",
        "-i",
        default="raw_data/omissions_sample_encounters_diarized.csv",
        help="Path to input CSV (default: diarized.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="pipeline_results.jsonl",
        help="Path to JSONL results file (default: pipeline_results.jsonl)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (0-based) for batch processing (default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of rows to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and build state, but do not call run_pipeline.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_csv_rows(path: str) -> Iterable[Dict[str, str]]:
    """
    Yields each row as a dict keyed by the CSV headers.
    Uses utf-8-sig to tolerate BOM in exported files.
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            yield row


def _none_if_blank(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    return s if s else None


def _parse_numbers_field(s: Optional[str]) -> Optional[List[str]]:
    """
    interpreter_numbers may be comma- or semicolon-separated.
    Returns a cleaned list or None.
    """
    s = _none_if_blank(s)
    if not s:
        return None
    # Split on common delimiters and strip each token.
    parts: List[str] = []
    for chunk in s.replace(";", ",").split(","):
        token = chunk.strip()
        if token:
            parts.append(token)
    return parts or None


def build_state_from_row(row: Dict[str, str]) -> Dict[str, Any]:
    """
    Build the LangGraph state dict expected by your pipeline.
    Includes new keys: 'pre_chart' and 'interpreter'.
    """
    run_id = _none_if_blank(row.get(COL_ENCOUNTER_ID))
    diarized = _none_if_blank(row.get(COL_DIARIZED))
    fallback_transcript = _none_if_blank(row.get(COL_TRANSCRIPT))
    transcript_text = diarized or fallback_transcript or ""

    pre_chart = _none_if_blank(row.get(COL_PREVISIT))
    hpi_input = _none_if_blank(row.get(COL_HPI))

    interpreter_numbers = _parse_numbers_field(row.get(COL_INTERP_NUM))
    interpreter_intro = _none_if_blank(row.get(COL_INTERP_INTRO))
    interpreter_obj: Optional[Dict[str, Any]] = None
    if interpreter_numbers or interpreter_intro:
        interpreter_obj = {
            "numbers": interpreter_numbers or [],
            "intro": interpreter_intro or "",
        }

    # --- State shape ---
    # Keep keys minimal and descriptive. If your graph uses a typed state, align names here.
    state: Dict[str, Any] = {
        "run_id": run_id,
        "source": "csv",
        "transcript": transcript_text,   # <— diarized transcript (preferred)
        "pre_chart": pre_chart,          # <— NEW: pre-charting text (string or None)
        "interpreter": interpreter_obj,  # <— NEW: dict or None
        "hpi_input": hpi_input,          # <— generated HPI you want to evaluate/compare

    }

    # Optional: guard against completely empty transcripts.
    if not state["transcript"]:
        logging.warning("Encounter %s has an empty transcript.", run_id)

    return state


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    setup_logging()

    logging.info("Input:  %s", args.input)
    logging.info("Output: %s", args.output)

    rows = list(read_csv_rows(args.input))
    total = len(rows)

    start = max(0, args.start)
    end = total if args.limit is None else min(total, start + args.limit)

    logging.info("Loaded %d rows. Processing [%d, %d).", total, start, end)

    out_buffer: List[Dict[str, Any]] = []
    processed = 0
    errors = 0

    for idx in range(start, end):
        row = rows[idx]
        run_id = _none_if_blank(row.get(COL_ENCOUNTER_ID)) or f"row_{idx}"
        try:
            state = build_state_from_row(row)

            if args.dry_run:
                logging.info("DRY RUN | %s | built state keys: %s", run_id, list(state.keys()))
                record = {
                    "run_id": run_id,
                    "status": "dry_run",
                    "state_keys": list(state.keys()),
                }
            else:
                output = run_pipeline(state)
                record = {
                    "run_id": run_id,
                    "status": "ok",
                    "output": output,
                }

            out_buffer.append(record)
            processed += 1

            # Flush periodically
            if len(out_buffer) >= 50:
                write_jsonl(args.output, out_buffer)
                out_buffer.clear()

            if processed % 25 == 0:
                logging.info("Processed %d/%d...", processed, end - start)

        except Exception as e:
            logging.exception("Error processing run_id=%s: %s", run_id, e)
            errors += 1
            out_buffer.append(
                {
                    "run_id": run_id,
                    "status": "error",
                    "error": str(e),
                }
            )

    # Final flush
    if out_buffer:
        write_jsonl(args.output, out_buffer)

    logging.info("Done. Processed=%d, Errors=%d. Results -> %s", processed, errors, args.output)


if __name__ == "__main__":
    main()
