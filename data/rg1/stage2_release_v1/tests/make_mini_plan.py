"""Miniature Stage 2 plan with the SAME ensemble structure as PLAN.json, for local plumbing tests only.
Q12/Q16/Q24 -> L6, Q32 -> L8 at the Stage 1 mini-plan masses, 8 chains x 3000 (equal to the Stage 1 mini chains,
so L24/L32 pooling is exercised). Never used for science.
  python make_mini_plan.py OUT.json STAGE1_MINI_PLAN.json STAGE1_MINI_ADJ.json"""
import copy
import hashlib
import json
import sys
from pathlib import Path

plan = json.loads((Path(__file__).parents[1] / "PLAN.json").read_text())
sub = {"Q12_lo": (6, -0.41), "Q12_hi": (6, -0.39), "Q16_lo": (6, -0.41), "Q16_hi": (6, -0.39),
       "Q24_lo": (6, -0.41), "Q24_hi": (6, -0.39), "Q32_lo": (8, -0.535), "Q32_hi": (8, -0.525)}
ens = []
for e in plan["ensembles"]:
    e = copy.deepcopy(e); L, m = sub[e["ensemble"]]
    e.update(L=L, m0sq=m, n_meas=3000, n_therm=300, time_limit_hours=0.25, chains=8, seeds=e["seeds"][:8])
    ens.append(e)
tasks = []
for e in ens:
    for c, s in enumerate(e["seeds"]):
        tasks.append({"task": len(tasks), "ensemble": e["ensemble"], "chain": c, "seed": s,
                      "args": [e["L"], repr(e["m0sq"]), repr(e["lambda"]), repr(e["h"]), e["n_therm"], e["n_meas"],
                               e["n_or"], e["meas_every"], s], "zero_mode_gibbs": e["zero_mode_gibbs"],
                      "n_meas": e["n_meas"], "expected_bytes": 32 * e["n_meas"], "time_limit_hours": 0.25})
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan.update(ensembles=ens, tasks=tasks, campaign="RG1_STAGE2_MINI_TEST_NOT_SCIENCE")
plan["stage1"].update(plan_sha256=sha(sys.argv[2]), adjudication_sha256=sha(sys.argv[3]))
Path(sys.argv[1]).write_text(json.dumps(plan, indent=1))
print(len(tasks), "mini tasks")
