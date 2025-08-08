#!/usr/bin/env python3
"""
Run a single row through the LangGraph pipeline and pause for a debugger attach.

Default behavior:
- Reads the first row (index 0) from the given CSV (default: diarized.csv)
- Builds the pipeline state using run_all.build_state_from_row
- (Optionally) opens a debug adapter server via `debugpy` and waits for a client
- Invokes `run_pipeline(state)`

Usage examples
--------------
# Just run first row
python scripts/debug_one.py

# Run row 5 from a custom CSV
python scripts/debug_one.py -i my_diarized.csv -n 5

# Attach a debugger (VS Code / PyCharm) on port 5678 and wait
python scripts/debug_one.py --debug --port 5678 --wait

In VS Code, use the "Python: Attach using Process Id" or a launch config like:
{
  "name": "Attach to debug_one.py",
  "type": "python",
  "request": "attach",
  "connect": {"host": "localhost", "port": 5678},
}
"""
from __future__ import annotations

import argparse
import sys

try:
    import debugpy  # type: ignore
except Exception:  # pragma: no cover
    debugpy = None  # Lazy optional

from pipeline.agent_pipeline import run_pipeline
from run_all import read_csv_rows, build_state_from_row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single CSV row through the pipeline for debugging")
    p.add_argument("-i", "--input", default="raw_data/omissions_sample_encounters_diarized.csv", help="Path to input CSV")
    p.add_argument("-n", "--index", type=int, default=0, help="0-based row index to run (default: 0)")
    p.add_argument("--debug", action="store_true", help="Start debug adapter (debugpy) on given port")
    p.add_argument("--port", type=int, default=5678, help="Debug adapter port (default: 5678)")
    p.add_argument("--wait", action="store_true", help="If --debug, wait for debugger to attach before running")
    return p.parse_args()


def maybe_start_debugger(enable: bool, port: int, wait: bool) -> None:
    if not enable:
        return
    if debugpy is None:
        print("⚠️  debugpy not installed. Install with `pip install debugpy` to use --debug. Continuing without attach…")
        return
    try:
        debugpy.listen(("0.0.0.0", port))
        print(f"🐛 debugpy listening on 0.0.0.0:{port}")
        if wait:
            print("⏳ Waiting for debugger to attach…")
            debugpy.wait_for_client()
            print("✅ Debugger attached. Proceeding…")
    except Exception as e:
        print(f"⚠️  Could not start debug server on port {port}: {e}")


def main() -> int:
    args = parse_args()

    maybe_start_debugger(args.debug, args.port, args.wait)

    # Load desired row
    rows = list(read_csv_rows(args.input))
    if not rows:
        print(f"No rows found in {args.input}")
        return 1
    if args.index < 0 or args.index >= len(rows):
        print(f"Index {args.index} out of range (0..{len(rows)-1}) for {args.input}")
        return 1

    row = rows[args.index]

    # Build state exactly as run_all expects
    state = build_state_from_row(row)

    # Handy breakpoint location if not using --debug
    # Uncomment to drop into the built-in debugger before running the graph:
    # import pdb; pdb.set_trace()

    print(f"▶️  Running pipeline for run_id={state.get('run_id')!r} (row {args.index})")
    try:
        result = run_pipeline(state)
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return 2

    print("✅ Done. Result (may be None if your runner returns in-graph):")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
