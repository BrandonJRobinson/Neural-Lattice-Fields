#!/usr/bin/env bash
# One-command integrated RG-1 Stage 1 preflight (local; submits nothing). Writes PREFLIGHT_<utc>/.
#   bash preflight.sh
# 1 release checksums  2 pinned sampler/analysis (package SHA256SUMS + binary)  3 plan contract
# 4 failure-injection + end-to-end tests (runner, capture-before-QC, restart, QC, adjudication, fail-closed)
# 5 authorization status (reported; submission itself enforces it)
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
REL="$(cd "$(dirname "$0")" && pwd)"; cd "$REL"
PY="$(cat python_bin.txt)"
OUT="$REL/PREFLIGHT_$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$OUT"
echo "== 1 release checksums"; sha256sum --quiet -c SHA256SUMS.txt && echo OK
echo "== 2 pinned inputs"; "$PY" -c 'import json; from rg1.frozen import verify; v = verify(); print(json.dumps({k: x["ok"] for k, x in v.items()})); assert all(x["ok"] for x in v.values())'
echo "== 3 plan contract"; "$PY" - <<'EOF'
import json
p = json.load(open("PLAN.json"))
print(f"{len(p['tasks'])} tasks; allocated {p['allocated_core_hours']:.1f} <= ceiling {p['resource_ceiling_core_hours']} core-h")
for e in p["ensembles"]:
    print(f"  {e['ensemble']:11s} L={e['L']:2d} m0^2={e['m0sq']:+.4f} lam={e['lambda']} h={e['h']:.3e} chains={e['chains']:2d} n_meas={e['n_meas']} therm={e['n_therm']} limit={e['time_limit_hours']}h")
EOF
echo "== 4 tests"; "$PY" -B -m pytest -o addopts="" -p no:cacheprovider -q --junitxml="$OUT/junit.xml" tests | tee "$OUT/pytest.log"
echo "== 5 authorization"
if [[ -f AUTHORIZATION_RG1_STAGE1.json ]]; then echo "present (validated by submit_stage1.sh)"; else echo "ABSENT: submission will refuse"; fi
echo "PREFLIGHT PASS (software). Outputs: $OUT"
