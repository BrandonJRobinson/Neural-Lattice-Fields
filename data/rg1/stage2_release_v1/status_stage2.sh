#!/usr/bin/env bash
# Read-only Stage 2 progress: queue state, captured/verified chains, QC verdicts, failures, core-hours used.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
REL="$(cd "$(dirname "$0")" && pwd)"
RG1_DATA="${RG1_DATA:-$(dirname "$REL")/stage2_data}"
PY="$(cat "$REL/python_bin.txt")"
squeue -u "$USER" -h -o '%i %j %T %M %l %R' </dev/null | grep rg1-s2 || echo "(no RG-1 jobs queued)"
"$PY" - "$REL" "$RG1_DATA" <<'EOF'
import json, sys, subprocess
from pathlib import Path
rel, data = Path(sys.argv[1]), Path(sys.argv[2]); sys.path.insert(0, str(rel))
from rg1.run_chain import verify_capture
plan = json.loads((rel / "PLAN.json").read_text())
by = {}
for t in plan["tasks"]:
    d = data / t["ensemble"] / f"chain_{t['chain']:02d}"
    q = d.parent / f"{d.name}.qc.json"
    s = "pending" if not d.is_dir() else ("CORRUPT" if verify_capture(d) else json.loads(q.read_text())["status"] if q.is_file() else "QC_MISSING")
    by.setdefault(t["ensemble"], []).append(s)
for k, v in by.items():
    print(f"{k:12s} " + " ".join(f"{s}:{v.count(s)}" for s in sorted(set(v))))
fails = sorted((data / "failures").glob("*.json")) if (data / "failures").is_dir() else []
for f in fails:
    r = json.loads(f.read_text()); print("FAILURE", r["task"], r["ensemble"], r["chain"], r["reason"][:120])
led = data / "SUBMISSION_LEDGER.json"
if led.is_file():
    L = json.loads(led.read_text()); ids = ",".join(i for s in L["submissions"] for i in s["array_job_ids"] + ([s["analysis_job_id"]] if s["analysis_job_id"] != "none" else []))
    out = subprocess.run(["sacct", "-j", ids, "-X", "-P", "-n", "--format=AllocCPUS,ElapsedRaw,State"], capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout
    used = sum(int(a) * int(e) / 3600 for a, e, _ in (ln.split("|") for ln in out.splitlines() if ln))
    print(f"used so far: {used:.1f} Slurm CPU-hours (AllocCPUS x elapsed; 2 SMT threads per physical core) = {used/2:.1f} physical core-hours; ceiling 1400 physical core-hours")
EOF
