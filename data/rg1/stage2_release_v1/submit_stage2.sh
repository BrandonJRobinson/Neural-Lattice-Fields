#!/usr/bin/env bash
# The ONLY RG-1 Stage 2 submission entry point.
#   bash submit_stage2.sh [--dry-run]          first submission (all Stage 2 tasks; adjudication runs via analyze_stage2.sh)
#   bash submit_stage2.sh --retry [--dry-run]  resubmit only tasks without a verified capture (same seeds)
# Refuses (exit 3) unless:
#   1. AUTHORIZATION_RG1_STAGE2.json names the authority over the Mission R PRODUCTION_LOCKED, states that
#      RG-1 Stage 2 HIPSTER array production is authorized, names this PLAN.json sha256, and a ceiling
#      >= every core-hour this script will have allocated (ledger + this submission);
#   2. the release is unchanged (SHA256SUMS.txt) and the pinned sampler/analysis verify;
#   3. no first submission exists yet (first mode) / no Stage 2 job is still queued (retry mode).
# The PRODUCTION_LOCKED files are read (hashed into the ledger) and never modified.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
REL="$(cd "$(dirname "$0")" && pwd)"
RG1_DATA="${RG1_DATA:-$(dirname "$REL")/stage2_data}"
PY="$(cat "$REL/python_bin.txt")"
SBATCH="${SBATCH:-sbatch}"
MODE=first; DRY=0; EXCL=(); ONLY=""; NOANA=1
for a in "$@"; do case "$a" in --retry) MODE=retry ;; --dry-run) DRY=1 ;; --exclude=*) EXCL=("$a") ;; --only=*) ONLY="${a#--only=}" ;; --no-analysis) NOANA=1 ;; *) echo "unknown $a" >&2; exit 2 ;; esac; done
refuse() { echo "REFUSED: $*" >&2; exit 3; }

( cd "$REL" && sha256sum --quiet -c SHA256SUMS.txt ) || refuse "release changed since certification"
( cd "$REL" && "$PY" -c 'from rg1.frozen import require; require()' ) || refuse "pinned inputs differ"
LEDGER="$RG1_DATA/SUBMISSION_LEDGER.json"
mkdir -p "$RG1_DATA" "/scratch/$USER/rg1_logs"

PLANS=$("$PY" - "$REL" "$RG1_DATA" "$MODE" "$LEDGER" "$ONLY" "$NOANA" <<'EOF'
import json, sys, hashlib, glob
from pathlib import Path
rel, data, mode, ledger = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
only = set()
for part in filter(None, sys.argv[5].split(",")):
    a_, _, b_ = part.partition("-"); only |= set(range(int(a_), int(b_ or a_) + 1))
noana = sys.argv[6] == "1"
sys.path.insert(0, str(rel))
from rg1.run_chain import verify_capture
plan = json.loads((rel / "PLAN.json").read_text())
plan_sha = hashlib.sha256((rel / "PLAN.json").read_bytes()).hexdigest()
auth_p = rel / "AUTHORIZATION_RG1_STAGE2.json"
if not auth_p.is_file():
    sys.exit("AUTHORIZATION_RG1_STAGE2.json missing (see AUTHORIZATION_TEMPLATE.json)")
a = json.loads(auth_p.read_text())
need = ("authority", "authority_role", "date", "lock_resolution", "scope", "approved_core_hours", "approved_plan_sha256", "statement")
miss = [k for k in need if not a.get(k)]
if miss: sys.exit(f"authorization incomplete: {miss}")
if a["approved_plan_sha256"] != plan_sha: sys.exit("authorization names a different PLAN.json")
led = json.loads(ledger.read_text()) if ledger.is_file() else {"submissions": []}
if mode == "first" and led["submissions"]: sys.exit("Stage 2 already submitted; use --retry")
if mode == "retry" and not led["submissions"]: sys.exit("nothing submitted yet")
todo = [t for t in plan["tasks"]
        if not (data / t["ensemble"] / f"chain_{t['chain']:02d}").is_dir()
        or verify_capture(data / t["ensemble"] / f"chain_{t['chain']:02d}")]
if mode == "retry":
    bad = [t["task"] for t in todo if (data / t["ensemble"] / f"chain_{t['chain']:02d}").is_dir()]
    if bad: sys.exit(f"captures failing verification (never overwritten): {bad}")
groups = {}
for t in todo:
    groups.setdefault(t["time_limit_hours"], []).append(t["task"])
if only:
    todo = [t for t in todo if t["task"] in only]
    groups = {}
    for t in todo:
        groups.setdefault(t["time_limit_hours"], []).append(t["task"])
analysis = 0.0 if noana else plan["analysis_job"]["cpus"] * plan["analysis_job"]["time_limit_hours"]
new = sum(t["time_limit_hours"] for t in todo) + analysis
spent = sum(s["allocated_core_hours"] for s in led["submissions"])
if mode == "retry":
    # Retry while other Stage 2 jobs may still run: count consumed core-h of ended jobs (sacct) plus the full
    # time-limit reservation of every job still queued/running; never resubmit a task that is queued/running.
    import subprocess, re
    # Ceiling is in PHYSICAL core-hours: Slurm counts both SMT threads of a core as CPUs (AllocCPUS=2 for one core).
    tpc_out = subprocess.run(["scontrol", "show", "node", "hipster-cn006"], capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout
    TPC = int(re.search(r"ThreadsPerCore=(\d+)", tpc_out).group(1)) if "ThreadsPerCore" in tpc_out else 1
    ids = [i for s in led["submissions"] for i in s["array_job_ids"]] + [s["analysis_job_id"] for s in led["submissions"] if s["analysis_job_id"] != "none"]
    q = subprocess.run(["squeue", "-h", "-r", "-j", ",".join(ids), "-o", "%i|%l|%C"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if q.returncode: sys.exit("squeue unavailable")
    def secs(t):
        d, _, hms = t.rpartition("-"); parts = [int(x) for x in hms.split(":")]
        while len(parts) < 3: parts.insert(0, 0)
        return (int(d) if d else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    live, reserved = set(), 0.0
    for ln in q.stdout.split():
        jid, lim, cpus = ln.split("|"); reserved += int(cpus) / TPC * secs(lim) / 3600
        if "_" in jid: live.add(int(jid.split("_")[1]))
    todo = [t for t in todo if t["task"] not in live and (not only or t["task"] in only)]
    acct = subprocess.run(["sacct", "-j", ",".join(ids), "-X", "-P", "-n", "--format=JobID,State,AllocCPUS,ElapsedRaw"],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if acct.returncode: sys.exit("sacct unavailable; cannot establish consumed core-hours")
    consumed = sum(int(c) / TPC * int(e) / 3600 for j, st, c, e in (ln.split("|") for ln in acct.stdout.splitlines() if ln.strip())
                   if st.split()[0] not in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING"))
    spent = consumed + reserved
    groups = {}
    for t in todo:
        groups.setdefault(t["time_limit_hours"], []).append(t["task"])
    new = sum(t["time_limit_hours"] for t in todo) + analysis
    if not todo: sys.exit("nothing to retry")
ceiling = min(float(a["approved_core_hours"]), plan["resource_ceiling_core_hours"])
if spent + new > ceiling + 1e-9:
    sys.exit(f"ceiling: already allocated {spent:.2f} + this {new:.2f} > authorized {ceiling:.2f} core-h")
LOCKS = ["/home/brobins/lattice_nnft_estimator_v2/PRODUCTION_LOCKED",
         "/home/brobins/lattice_nnft_upload/lattice_nnft_hipster_triviality_20260730/lattice_nnft/PRODUCTION_LOCKED",
         "/home/brobins/lattice_nnft_upload/lattice_nnft_nnft4pt_20260811/lattice_nnft/PRODUCTION_LOCKED",
         "/home/brobins/lattice_nnft_upload/lattice_nnft_nnft_source_20260814/lattice_nnft/PRODUCTION_LOCKED"]
locks = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() if Path(p).is_file() else "ABSENT" for p in LOCKS}
prior = [i for s_ in led["submissions"] for i in s_["array_job_ids"]]
print(json.dumps({"prior_arrays": prior, "noana": noana, "groups": {f"{h:g}": ids for h, ids in sorted(groups.items(), reverse=True)}, "new": new, "spent": spent,
                  "ceiling": ceiling, "plan_sha": plan_sha, "locks": locks, "n": len(todo)}))
EOF
) || refuse "authorization/ledger check failed (see message above)"

"$PY" -c "import json,sys; d=json.loads(sys.argv[1]); print(f\"{d['n']} tasks, this submission {d['new']:.2f} core-h, previously {d['spent']:.2f}, ceiling {d['ceiling']:.2f}\")" "$PLANS"
EXPORT="ALL,RG1_RELEASE=$REL,RG1_DATA=$RG1_DATA"
ARRAYS=()
while IFS=$'\t' read -r hours ids; do
  secs=$("$PY" -c "print(int(round(float('$hours')*3600)))")
  tl=$(printf '%02d:%02d:%02d' $((secs/3600)) $((secs%3600/60)) $((secs%60)))
  cmd=("$SBATCH" --parsable --array="$ids" --time="$tl" "${EXCL[@]}" --export="$EXPORT" "$REL/slurm/stage2_chain.sbatch")
  if [[ $DRY == 1 ]]; then echo "${cmd[*]}"; ARRAYS+=("<array>"); else ARRAYS+=("$("${cmd[@]}" </dev/null)"); fi
done < <("$PY" -c "import json,sys; [print(h, ','.join(map(str,ids)), sep='\t') for h,ids in json.loads(sys.argv[1])['groups'].items()]" "$PLANS")
PRIOR=$("$PY" -c "import json,sys; print(' '.join(json.loads(sys.argv[1])['prior_arrays']))" "$PLANS")
ALL=(${PRIOR} "${ARRAYS[@]}")
DEP=$(IFS=:; echo "${ALL[*]}"); IDS=$(IFS=,; echo "${ARRAYS[*]}"); ALLIDS=$(IFS=,; echo "${ALL[*]}")
# Stage 2 adjudication is not a dependent Slurm job; it is run with analyze_stage2.sh once all chains have ended.
[[ $DRY == 1 ]] && exit 0; ANALYSIS=none
"$PY" - "$LEDGER" "$PLANS" "$IDS" "$ANALYSIS" "$MODE" <<'EOF'
import json, sys, time
from pathlib import Path
led_p, plans, ids, analysis, mode = Path(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
led = json.loads(led_p.read_text()) if led_p.is_file() else {"submissions": []}
led["submissions"].append({"mode": mode, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "array_job_ids": ids.split(","),
                           "analysis_job_id": analysis, "tasks_by_time_limit_hours": plans["groups"],
                           "allocated_core_hours": plans["new"], "plan_sha256": plans["plan_sha"],
                           "production_locks_unchanged_sha256": plans["locks"]})
led_p.write_text(json.dumps(led, indent=1) + "\n")
print(json.dumps(led["submissions"][-1], indent=1))
EOF
