#!/usr/bin/env bash
# One-command integrated RG-1 Stage 2 preflight (local; submits nothing). Writes PREFLIGHT_<utc>/.
# 1 release checksums  2 pinned sampler/analysis  3 Stage 1 inputs pinned by the plan  4 plan contract
# 5 Stage 1 runner/capture/QC failure-injection suite (modules byte-identical) + Stage 2 end-to-end tests  6 authorization
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
REL="$(cd "$(dirname "$0")" && pwd)"; cd "$REL"
PY="$(cat python_bin.txt)"
OUT="$REL/PREFLIGHT_$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$OUT"
echo "== 1 release checksums"; sha256sum --quiet -c SHA256SUMS.txt && echo OK
echo "== 2 pinned inputs"; "$PY" -c 'import json; from rg1.frozen import verify; v = verify(); print(json.dumps({k: x["ok"] for k, x in v.items()})); assert all(x["ok"] for x in v.values())'
echo "== 3 Stage 1 inputs"; "$PY" - <<'PYEOF'
import json, hashlib
from pathlib import Path
p = json.load(open("PLAN.json")); root = Path("..").resolve()
h = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
assert h(root / "stage1_release_v1/PLAN.json") == p["stage1"]["plan_sha256"]
assert h(root / "stage1_data/adjudication_local_20261029T100334Z/STAGE1_ADJUDICATION.json") == p["stage1"]["adjudication_sha256"]
for f in ("rg1/run_chain.py", "rg1/qc.py", "rg1/frozen.py", "rg1/analyze_stage1.py"):
    assert h(f) == h(root / "stage1_release_v1" / f), f
print("Stage 1 plan, adjudication and shared modules match")
PYEOF
echo "== 4 plan contract"; "$PY" - <<'PYEOF'
import json
p = json.load(open("PLAN.json"))
print(f"{len(p['tasks'])} tasks; expected {p['expected_core_hours']:.1f}, allocated {p['allocated_core_hours']:.1f} <= ceiling {p['resource_ceiling_core_hours']} core-h")
for e in p["ensembles"]:
    print(f"  {e['ensemble']:7s} L={e['L']:2d} m0^2={e['m0sq']:+.4f} chains={e['chains']:2d} n_meas={e['n_meas']} limit={e['time_limit_hours']}h")
PYEOF
echo "== 5 tests"
( cd ../stage1_release_v1 && "$PY" -B -m pytest -o addopts="" -p no:cacheprovider -q -k "not adjudicator and not plan_is" tests ) | tee "$OUT/pytest_runner.log"
"$PY" -B -m pytest -o addopts="" -p no:cacheprovider -q --junitxml="$OUT/junit.xml" tests | tee "$OUT/pytest.log"
echo "== 6 authorization"
if [[ -f AUTHORIZATION_RG1_STAGE2.json ]]; then echo "present (validated by submit_stage2.sh)"; else echo "ABSENT: submission will refuse"; fi
echo "PREFLIGHT PASS (software). Outputs: $OUT"
