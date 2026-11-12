"""Generate the frozen RG-1 Stage 1 plan (PLAN.json). Run once; the plan is then hashed and immutable.

Design (predeclared before any Stage 1 data exist):
  * Production: zero-source ensembles at two bracketing bare masses per volume, symmetric about the named
    masses (L24 m0^2 = -0.6775, L32 m0^2 = -0.686). Half-width chosen from dz/dm0^2 ~ L^2/(2z) so each pair
    straddles z = 3 by ~+-0.067 (> 4 sigma of the in-session z errors). Equal statistics per mass.
  * Statistics: 2% relative sigma_g at z = 3 at the conservative reference g = 5.4 (one-loop expectation),
    from the measured per-configuration sd and tau_int of the RB kappa_4 influence (results/forecast.json):
    L24 5.91e6 sweeps, L32 7.79e6 sweeps in total over both masses.
  * >= 8 independent chains per mass (L24 10, L32 16), independent seeds, 5,000 thermalization sweeps.
  * Predeclared controls in the same release (section 9 of the verdict): L12 reproduction, lambda = 0
    Gaussian, local-only-kernel cross-check at L12/L16 with a Gibbs-on L16 reference, direct sourced runs
    for the reweighting identity at one production mass per volume.
Sampler call (unchanged binary): L m0sq lambda h n_therm n_meas n_or meas_every seed outfile zero_mode_gibbs
"""
import json
from pathlib import Path

N_OR, MEAS_EVERY, N_THERM = 2, 1, 5000
ens = []


def add(eid, role, L, m0sq, lam, h, n_meas, chains, time_h, zm=1, eta=None, seed0=None):
    ens.append({"ensemble": eid, "role": role, "L": L, "m0sq": m0sq, "lambda": lam, "h": h, "eta": eta,
                "n_therm": N_THERM, "n_meas": n_meas, "n_or": N_OR, "meas_every": MEAS_EVERY,
                "zero_mode_gibbs": zm, "chains": chains, "time_limit_hours": time_h,
                "seeds": [seed0 + c for c in range(chains)]})


# production (seeds: 26102800000 + 10000*k + chain; never used before)
add("P24_lo", "production", 24, -0.6782, 0.4, 0.0, 296000, 10, 6.0, seed0=26102810000)
add("P24_hi", "production", 24, -0.6768, 0.4, 0.0, 296000, 10, 6.0, seed0=26102820000)
add("P32_lo", "production", 32, -0.6864, 0.4, 0.0, 245000, 16, 13.5, seed0=26102830000)
add("P32_hi", "production", 32, -0.6856, 0.4, 0.0, 245000, 16, 13.5, seed0=26102840000)
# controls
add("C12_repro", "control_L12_reproduction", 12, -0.6223, 0.4, 0.0, 150000, 8, 1.0, seed0=26102850000)
add("C08_gauss", "control_lambda0", 8, 0.5, 0.0, 0.0, 100000, 4, 0.5, seed0=26102860000)
add("C12_local", "control_cross_kernel", 12, -0.6223, 0.4, 0.0, 150000, 4, 0.75, zm=0, seed0=26102870000)
add("C16_local", "control_cross_kernel", 16, -0.657, 0.4, 0.0, 100000, 4, 0.75, zm=0, seed0=26102880000)
add("C16_gibbs", "control_cross_kernel_reference", 16, -0.657, 0.4, 0.0, 100000, 4, 0.75, seed0=26102890000)
add("S24_eta4p5", "control_direct_source", 24, -0.6768, 0.4, 4.5 / 24 ** 4 * 24, 60000, 2, 2.0, eta=4.5, seed0=26102900000)
add("S32_eta4p5", "control_direct_source", 32, -0.6856, 0.4, 4.5 / 32 ** 4 * 32, 30000, 2, 3.0, eta=4.5, seed0=26102910000)

tasks = []
for e in ens:
    for c, s in enumerate(e["seeds"]):
        tasks.append({"task": len(tasks), "ensemble": e["ensemble"], "chain": c, "seed": s,
                      "args": [e["L"], repr(e["m0sq"]), repr(e["lambda"]), repr(e["h"]), e["n_therm"], e["n_meas"],
                               e["n_or"], e["meas_every"], s],
                      "zero_mode_gibbs": e["zero_mode_gibbs"], "n_meas": e["n_meas"],
                      "expected_bytes": 32 * e["n_meas"], "time_limit_hours": e["time_limit_hours"]})
ANALYSIS = {"cpus": 8, "time_limit_hours": 1.5}
alloc = sum(t["time_limit_hours"] for t in tasks) + ANALYSIS["cpus"] * ANALYSIS["time_limit_hours"]
plan = {"schema": "rg1-stage1-plan-v1", "campaign": "RG1_STAGE1_20261028",
        "sampler": {"source": "source/src/phi4_local2.cpp",
                    "source_sha256": "492c2aa21baec4c10b3ea32dce3dd4490d87752a71d5977b643f66d0d9ad55c7",
                    "binary": "bin/phi4_local2",
                    "binary_sha256": "fe5601a3a624572c60bd5c72c81f6cd55a164be64d817507879f6e1b477311c9",
                    "package_zip_sha256": "0b833d4591018e6ea7acbf7c348e13ced53b54e36585ab1723c689781ff2c6db",
                    "compiler": "g++ (GCC) 8.5.0 20210514 (Red Hat 8.5.0-29)"},
        "ensembles": ens, "tasks": tasks, "analysis_job": ANALYSIS,
        "resource_ceiling_core_hours": 600.0, "allocated_core_hours": alloc,
        "restart_policy": ("A chain is complete only when its full file (exactly 32*n_meas bytes) and stdout JSON were "
                           "captured and hashed on persistent storage. A failed or timed-out chain leaves no "
                           "completed file and is never analysed. Rerunning a task reuses the same seed and is "
                           "bit-identical; reruns are submitted only within the remaining authorized ceiling. "
                           "Ensembles are analysed with all complete chains if >= 8 are complete (controls: all)."),
        "stage2": "conditional; at most 1,400 additional core-hours; separate gate approval; never automatic"}
out = Path(__file__).with_name("PLAN.json")
out.write_text(json.dumps(plan, indent=1) + "\n")
print(f"{len(tasks)} tasks, allocated ceiling {alloc:.2f} core-h")
