#!/usr/bin/env python3
"""
demo_app/app.py
---------------
One-command local demo server (no npm). Serves the frontend and exposes two APIs:

GET  /api/data
  -> { runs: [ { run_id, hpi, omissions:[...] }, ... ],
       meta: { omissions_jsonl, hpi_csv } }

POST /api/generate
  body: { run_id, original_hpi, selected_omissions:[...] }
  -> { new_hpi }

Reads omissions from out/omissions.jsonl (one JSON object per line),
and HPIs from raw_data/omissions_sample_encounters_diarized.csv.

Uses OPENAI_API_KEY from .env (repo root).
"""
import os, json, csv, hashlib
from pathlib import Path
from urllib.parse import urlparse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

from dotenv import load_dotenv
from openai import OpenAI

# ---------- Paths ----------
DEMO_DIR   = Path(__file__).resolve().parent
REPO_ROOT  = DEMO_DIR.parent
OUT_DIR    = REPO_ROOT / "out"
RAW_DIR    = REPO_ROOT / "raw_data"
OM_JSONL   = OUT_DIR / "omissions.jsonl"
HPI_CSV    = RAW_DIR / "omissions_sample_encounters_diarized.csv"

# ---------- OpenAI ----------
load_dotenv(REPO_ROOT / ".env")
_openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30,
    max_retries=2,
)

# ---------- Helpers ----------
def _sha12(*parts: str) -> str:
    base = "||".join((p or "").strip() for p in parts)
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:12]

def _load_omissions_jsonl() -> dict:
    """
    Returns {"runs":[{"run_id":..., "omissions":[...]}]}
    Parses out/omissions.jsonl where each line is a JSON object like:
      {"ts": "...", "run_id": "...", "status":"omitted|conflict", "priority":"high|...","materiality":4.0,
       "code":"...", "value":"...", "polarity":"present|denied|uncertain",
       "problem_id":"...", "time_scope":"...", "evidence_text":"...", ...}
    """
    if not OM_JSONL.exists():
        raise FileNotFoundError(f"Missing {OM_JSONL}. Run your pipeline first to produce omissions.")

    by_run = {}
    with OM_JSONL.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"omissions.jsonl line {ln} is not valid JSON") from e

            rid = (row.get("run_id") or "").strip()
            if not rid:
                # skip lines without run_id
                continue

            code        = (row.get("code") or "").strip() or None
            value       = (row.get("value") or "").strip() or None
            polarity    = (row.get("polarity") or "").strip() or None
            problem_id  = (row.get("problem_id") or "").strip() or None
            time_scope  = (row.get("time_scope") or "").strip() or None
            evidence    = (row.get("evidence_text") or "").strip() or None
            priority    = (row.get("priority") or "").strip() or None
            materiality = row.get("materiality")

            try:
                materiality = float(materiality) if materiality is not None else None
            except Exception:
                materiality = None

            oid = _sha12(rid, code or "", value or "", polarity or "", problem_id or "", time_scope or "", evidence or "")
            item = {
                "id": oid,
                "code": code,
                "value": value,
                "polarity": polarity,
                "problem_id": problem_id,
                "time_scope": time_scope,
                "evidence_text": evidence,
                "priority": priority,
                "materiality": materiality,
            }
            by_run.setdefault(rid, []).append(item)

    return {"runs": [{"run_id": rid, "omissions": items} for rid, items in by_run.items()]}

def _load_hpi_map(csv_path: Path) -> dict:
    """
    Returns {run_id -> HPI string} from the provided CSV.
    Expects columns: 'Encounter ID', 'HPI'
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"HPI CSV not found: {csv_path}")
    mp = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("Encounter ID") or "").strip()
            hpi = (row.get("HPI") or "").strip()
            if rid:
                mp[rid] = hpi
    return mp

def _generate_new_hpi(original_hpi: str, selected_omissions: list) -> str:
    """
    Call gpt-4o-mini to produce a revised HPI that incorporates the selected omissions.
    New/edited sentences must be wrapped in **bold** (markdown).
    """
    system = (
        "You are an expert clinical scribe. Revise the HPI by INSERTING the omitted facts "
        "without deleting existing content. Use concise clinical style. "
        "Wrap every NEW or SUBSTANTIVELY EDITED sentence that incorporates an omission in **double asterisks**. "
        "Output the HPI text only (no headings, no commentary)."
    )
    user = {
        "original_hpi": original_hpi or "",
        "omissions": selected_omissions or [],
        "instructions": [
            "Integrate omissions logically (group by problem/OLDCARTS).",
            "Do not contradict the original; clarify if needed.",
            "Keep length reasonable; do not rewrite unrelated sections.",
            "Bold only the NEW/EDITED sentences that add the missing info."
        ],
    }
    resp = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()

# ---------- HTTP Handler ----------
class AppHandler(SimpleHTTPRequestHandler):
    # Serve from the repo root so /demo_app, /out, /raw_data are all visible.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def _send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/data":
            try:
                om = _load_omissions_jsonl()   # {"runs":[...]}
                hpi_map  = _load_hpi_map(HPI_CSV)

                runs = []
                for item in om.get("runs", []):
                    rid = (item.get("run_id") or "").strip()
                    if not rid:
                        continue
                    runs.append({
                        "run_id": rid,
                        "hpi": hpi_map.get(rid, ""),
                        "omissions": item.get("omissions", []),
                    })

                self._send_json({
                    "runs": runs,
                    "meta": {
                        "omissions_jsonl": str(OM_JSONL.relative_to(REPO_ROOT)),
                        "hpi_csv": str(HPI_CSV.relative_to(REPO_ROOT)),
                    }
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/generate":
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(length)
                data = json.loads(body or b"{}")
                original_hpi = data.get("original_hpi") or ""
                selected = data.get("selected_omissions") or []
                new_hpi = _generate_new_hpi(original_hpi, selected)
                self._send_json({"new_hpi": new_hpi})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        return super().do_POST()

def main():
    port = int(os.getenv("PORT") or "8000")
    server = ThreadingHTTPServer(("0.0.0.0", port), AppHandler)
    url = f"http://localhost:{port}/demo_app/"
    print(f"🚀 Demo app running at {url}")
    print("↪  Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
