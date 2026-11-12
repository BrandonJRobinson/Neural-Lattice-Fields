#!/usr/bin/env bash
# Run the predeclared Stage 2 adjudication once every Stage 2 chain has ended (login node or any host with the release).
#   bash analyze_stage2.sh      -> stage2_data/adjudication_<utc>/STAGE2_RESULT.json (+ sacct export, CPU time, SHA256SUMS)
# Refuses while Stage 2 jobs are still queued/running, or if the release or pinned inputs changed.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
REL="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$REL")"
RG1_DATA="${RG1_DATA:-$ROOT/stage2_data}"
PY="$(cat "$REL/python_bin.txt")"
( cd "$REL" && sha256sum --quiet -c SHA256SUMS.txt ) || { echo "REFUSED: release changed" >&2; exit 3; }
LED="$RG1_DATA/SUBMISSION_LEDGER.json"
IDS=$("$PY" -c "import json; L=json.load(open('$LED')); print(','.join(i for s in L['submissions'] for i in s['array_job_ids']))")
if [[ -n "$(squeue -h -r -j "$IDS" -o %i </dev/null 2>/dev/null)" ]]; then echo "REFUSED: Stage 2 jobs still queued/running" >&2; exit 3; fi
OUT="$RG1_DATA/adjudication_$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$OUT"
sacct -j "$IDS" -P --format=JobID,JobIDRaw,JobName%30,State,ExitCode,TotalCPU,AllocCPUS,ElapsedRaw,NodeList </dev/null > "$OUT/sacct.txt"
cd "$REL"
t0=$(date +%s)
set +e
nice -n 10 "$PY" -m rg1.analyze_stage2 --plan "$REL/PLAN.json" --data "$RG1_DATA" \
  --stage1-plan "$ROOT/stage1_release_v1/PLAN.json" --stage1-data "$ROOT/stage1_data" \
  --stage1-adj "$ROOT/stage1_data/adjudication_local_20261029T100334Z/STAGE1_ADJUDICATION.json" \
  --sacct "$OUT/sacct.txt" --out "$OUT" --workers "${RG1_WORKERS:-8}" > "$OUT/stdout.log" 2> "$OUT/stderr.log"
rc=$?
set -e
{ echo "rc=$rc host=$(hostname) wall_s=$(( $(date +%s) - t0 ))"; echo "bash times (shell, then children user/sys):"; times; } > "$OUT/time.txt"
( cd "$OUT" && sha256sum $(ls | grep -v "^SHA256SUMS.txt$") > SHA256SUMS.txt )
cat "$OUT/time.txt"; tail -3 "$OUT/stdout.log"; tail -5 "$OUT/stderr.log"; echo "$OUT"
exit $rc
