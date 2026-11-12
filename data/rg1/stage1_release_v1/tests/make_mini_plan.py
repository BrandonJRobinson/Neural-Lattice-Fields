"""Miniature plan with the SAME ensemble structure as PLAN.json for local end-to-end plumbing tests only.
L24 -> L6, L32 -> L8, L16 -> L6, L12 -> L6 (z ~ 3 masses from the in-session ladder). Never used for science."""
import copy
import json
import sys
from pathlib import Path

plan = json.loads((Path(__file__).parents[1] / "PLAN.json").read_text())
sub = {"P24_lo": (6, -0.41, 3000), "P24_hi": (6, -0.39, 3000), "P32_lo": (8, -0.535, 3000), "P32_hi": (8, -0.525, 3000),
       "C12_repro": (6, -0.4, 3000), "C08_gauss": (4, 0.5, 3000), "C12_local": (6, -0.4, 3000),
       "C16_local": (6, -0.4, 3000), "C16_gibbs": (6, -0.4, 3000), "S24_eta4p5": (6, -0.39, 3000), "S32_eta4p5": (8, -0.525, 3000)}
ens = []
for e in plan["ensembles"]:
    e = copy.deepcopy(e); L, m, n = sub[e["ensemble"]]
    e.update(L=L, m0sq=m, n_meas=n, n_therm=300, time_limit_hours=0.25)
    if e["eta"]:
        e["h"] = e["eta"] / L ** 3
    if e["role"] == "production":
        e["chains"] = 8; e["seeds"] = e["seeds"][:8]
    ens.append(e)
tasks = []
for e in ens:
    for c, s in enumerate(e["seeds"]):
        tasks.append({"task": len(tasks), "ensemble": e["ensemble"], "chain": c, "seed": s,
                      "args": [e["L"], repr(e["m0sq"]), repr(e["lambda"]), repr(e["h"]), e["n_therm"], e["n_meas"],
                               e["n_or"], e["meas_every"], s], "zero_mode_gibbs": e["zero_mode_gibbs"],
                      "n_meas": e["n_meas"], "expected_bytes": 32 * e["n_meas"], "time_limit_hours": 0.25})
plan.update(ensembles=ens, tasks=tasks, campaign="RG1_MINI_TEST_NOT_SCIENCE")
Path(sys.argv[1]).write_text(json.dumps(plan, indent=1))
print(len(tasks), "mini tasks")
