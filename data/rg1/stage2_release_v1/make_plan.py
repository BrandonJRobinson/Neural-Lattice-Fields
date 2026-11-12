"""Generate the frozen RG-1 Stage 2 plan (PLAN.json). Run once, before any Stage 2 data; then hashed and immutable.

Design (contract section 9 / execution order step 4, predeclared allocation; not tuned on Stage 1 slopes):
  * Targets after z = 3 interpolation: L12 0.2%, L16 0.3%, L24 1.35%, L32 1.35% relative sigma_g.
  * Units (per-configuration sd of g, tau_int, seconds per sweep) are MEASURED in Stage 1
    (STAGE1_ADJUDICATION.json, sha pinned below): L24/L32 from the interpolated production error; L12/L16 from
    the larger of the Stage 1 control ensemble (C12_repro, C16_gibbs) and the package forecast.
  * L24/L32: extend the Stage 1 bracketing masses with NEW independent chains of the SAME length (296k / 245k),
    pooled with the Stage 1 chains (equal-length pooling, 8 jackknife blocks per chain). Chains needed =
    ceil((N_target - N_stage1) / n_meas), split equally over the two masses. No extra margin (units measured
    on these very ensembles).
  * L12/L16: NEW z = 3 brackets (two masses each, 8 chains per mass), N_target x 1.15 margin for the
    interpolation weight and the forecast-derived units. Masses from Stage 1 z(m0^2) at L12 (-0.6223 ->
    z = 3.026) and L16 (-0.657 -> z = 2.937) with dz/dm0^2 ~ 0.82 L^2/(2z) (Stage 1 measured 0.83 at L24, 0.81 at L32).
  * Time limit per chain = max(1.4 x expected, expected + 0.5 h) from Stage 1 measured wall time per sweep,
    rounded up to 0.25 h. 5,000 thermalization sweeps per chain. Seeds 26102930000 + 10000 k + chain (unused).
  * One adjudication reservation (8 CPU x 1.5 h). Allocated total <= 1,400 core-hours.
Sampler call (unchanged binary): L m0sq lambda h n_therm n_meas n_or meas_every seed outfile zero_mode_gibbs
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S1_ADJ = ROOT / "stage1_data/adjudication_local_20261029T100334Z/STAGE1_ADJUDICATION.json"
S1_ADJ_SHA = "e679129dc5758e5d7b15776a2b16264d3e9f0a7996890300f168ff3e80455684"
S1_PLAN_SHA = "22912a4a33ba34f7222ec8538bcbbcf227d810380eb2f44ec7d8e81add5edcdf"
assert hashlib.sha256(S1_ADJ.read_bytes()).hexdigest() == S1_ADJ_SHA
s1 = json.loads(S1_ADJ.read_text())
assert s1["plan_sha256"] == S1_PLAN_SHA and s1["stage1_gate"]["stage2_gate"] == "PASS"
fc = json.loads((ROOT / "source/results/forecast.json").read_text())["units"]

TARGET = {12: 0.002, 16: 0.003, 24: 0.0135, 32: 0.0135}
N_OR, MEAS_EVERY, N_THERM, CEILING = 2, 1, 5000, 1400.0
E = s1["ensembles"]; U = s1["stage1_gate"]["projection"]["units"]


def ctl_unit(k):
    e = E[k]
    return e["g_err_used"] * math.sqrt(e["N"] / (2 * e["tau_int"]["infl_rb"])), e["tau_int"]["infl_rb"], e["seconds_per_sweep_mean"]


units = {}
for L, k in ((12, "C12_repro"), (16, "C16_gibbs")):
    sd, tau, sps = ctl_unit(k)
    units[L] = {"sd": max(sd, fc[str(L)]["sd_g_per_config"]), "tau": max(tau, fc[str(L)]["tau_int"]), "sps": sps}
for L in (24, 32):
    u = U[str(L)]; units[L] = {"sd": u["sd_g_per_config"], "tau": u["tau_int"], "sps": u["sec_per_sweep"]}
g_ref = {12: 5.805, 16: 5.5, 24: s1["z3"]["24"]["g_at_z3"][0], 32: s1["z3"]["32"]["g_at_z3"][0]}
have = {24: E["P24_lo"]["N"] + E["P24_hi"]["N"], 32: E["P32_lo"]["N"] + E["P32_hi"]["N"]}
need = {L: 2 * units[L]["tau"] * (units[L]["sd"] / (TARGET[L] * g_ref[L])) ** 2 for L in TARGET}

ens = []


def limit(n_meas, sps):
    exp_h = (n_meas + N_THERM) * sps / 3600
    return exp_h, math.ceil(max(1.4 * exp_h, exp_h + 0.5) * 4) / 4


def add(eid, L, m0sq, n_meas, chains, k, role="production"):
    exp_h, tl = limit(n_meas, units[L]["sps"])
    ens.append({"ensemble": eid, "role": role, "L": L, "m0sq": m0sq, "lambda": 0.4, "h": 0.0, "eta": None,
                "n_therm": N_THERM, "n_meas": n_meas, "n_or": N_OR, "meas_every": MEAS_EVERY, "zero_mode_gibbs": 1,
                "chains": chains, "time_limit_hours": tl, "expected_hours_per_chain": exp_h,
                "seeds": [26102930000 + 10000 * k + c for c in range(chains)]})


for i, (L, lo, hi) in enumerate(((12, -0.6250, -0.6223), (16, -0.6570, -0.6535))):
    n = math.ceil(need[L] * 1.15 / 16 / 1000) * 1000
    add(f"Q{L}_lo", L, lo, n, 8, 2 * i); add(f"Q{L}_hi", L, hi, n, 8, 2 * i + 1)
for i, (L, lo, hi, n) in enumerate(((24, -0.6782, -0.6768, 296000), (32, -0.6864, -0.6856, 245000))):
    per_mass = math.ceil(max(0.0, need[L] - have[L]) / n / 2)
    add(f"Q{L}_lo", L, lo, n, per_mass, 4 + 2 * i); add(f"Q{L}_hi", L, hi, n, per_mass, 5 + 2 * i)

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
expected = sum(e["expected_hours_per_chain"] * e["chains"] for e in ens) + ANALYSIS["cpus"] * ANALYSIS["time_limit_hours"]
assert alloc <= CEILING, alloc
s1plan = json.loads((ROOT / "stage1_release_v1/PLAN.json").read_text())
plan = {"schema": "rg1-stage2-plan-v1", "campaign": "RG1_STAGE2_20261029",
        "sampler": s1plan["sampler"], "stage1": {"plan_sha256": S1_PLAN_SHA, "adjudication_sha256": S1_ADJ_SHA,
                                                 "data_root": "stage1_data", "pooled_with": {"Q24_lo": "P24_lo", "Q24_hi": "P24_hi",
                                                                                            "Q32_lo": "P32_lo", "Q32_hi": "P32_hi"}},
        "design": {"targets_rel": TARGET, "units": units, "g_ref": g_ref, "N_target": need, "N_stage1": have},
        "ensembles": ens, "tasks": tasks, "analysis_job": ANALYSIS,
        "resource_ceiling_core_hours": CEILING, "allocated_core_hours": alloc, "expected_core_hours": expected,
        "restart_policy": s1plan["restart_policy"]}
out = Path(__file__).with_name("PLAN.json")
out.write_text(json.dumps(plan, indent=1) + "\n")
for e in ens:
    print(f"{e['ensemble']:7s} L={e['L']} m0^2={e['m0sq']:+.4f} chains={e['chains']:2d} n_meas={e['n_meas']:8d} "
          f"exp={e['expected_hours_per_chain']:.2f}h limit={e['time_limit_hours']}h")
print(f"{len(tasks)} tasks; expected {expected:.1f}, allocated {alloc:.1f} <= {CEILING} core-h")
