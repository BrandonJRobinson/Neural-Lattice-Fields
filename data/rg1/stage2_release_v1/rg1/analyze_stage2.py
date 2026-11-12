"""Predeclared RG-1 Stage 2 (final) adjudication. Frozen before any Stage 2 data exist; no option changes estimators.

Inputs: Stage 2 PLAN.json + data root, Stage 1 PLAN.json + data root + STAGE1_ADJUDICATION.json (sha-pinned by the
Stage 2 plan), an sacct export. Every number comes from captured, hash-verified, QC-PASS chains only. Ensemble
analysis = the package's analyze_ensemble.py; PT comparison and running fits = its pt_compare.py and
running_fits_g.py (all hash-pinned, unmodified). Per-ensemble rules are exactly Stage 1's (rg1.analyze_stage1).

Rules
  * Ensembles: L12, L16 = Stage 2 brackets Q12/Q16. L24, L32 = Stage 1 P chains pooled with Stage 2 Q chains at
    the same (L, m0^2) (equal chain length, 8 jackknife blocks per chain). Each Stage 2 ensemble needs >= 8
    usable chains and each Stage 1 P ensemble must still verify with >= 8 usable chains.
  * Error = max(jackknife, replica scatter); z = 3 interpolation and bracket rule as Stage 1, at all four volumes.
  * Controls (PASS <= 2 sigma, WARN 2-3, FAIL > 3): Stage 2-only vs Stage 1 at each L24/L32 mass (z and g);
    Q12_hi (m0^2 = -0.6223) vs the in-session L12 reference z = 3.0245(11), g = 5.805(41); Lee-Yang order and
    Lebowitz sign as Stage 1. Stage 1 controls (lambda = 0, cross-kernel, reweighting identity) are carried over.
  * Rows for fits: in-session L6, L8, L10 (scaling_final.json) + the four new z = 3 values. Volumes are
    independent ensembles, so the full covariance is diagonal. Frozen pt_compare.py and running_fits_g.py
    (windows L6-32, L6-12, L12-32, L8-32) + L >= 16 sensitivity (L16-32: M1, MPT, MCUT with 0 dof) using the
    same model formulas and the frozen one-loop kernel running_fits.K.
  * Calibration: 2,000 synthetic campaigns each under alpha = 0 and alpha = 1 (MPT truth with its intercept fitted
    to the data at fixed alpha), Gaussian with the measured per-volume errors, on the primary window L12-32;
    each campaign refit with M1 and MCUT. Reported: false-positive rate and power of "B > 1.96 sigma_B" for M1,
    MCUT and both. RNG seed 20261029.
  * Claim (primary window L12-32, primary model MCUT; no other window or model may be substituted):
      POSITIVE_RUNNING_RESOLVED  iff all prerequisites pass, MCUT and M1 95% intervals for B both exclude 0
                                 from above, and the calibrated joint false-positive rate <= 0.05;
      NEGATIVE_SLOPE_RESOLVED    iff prerequisites pass and the MCUT 95% upper bound on B is < 0;
      NOT_RESOLVED_BOUNDED       otherwise with prerequisites passing (the exact intervals are the result);
      INCONCLUSIVE_PREREQUISITE_FAIL if any production validity, bracket or control FAIL.
    PT: CONSISTENT_WITH_ONE_LOOP iff |alpha_MPT(L12-32) - 1| <= 1.96 sigma, else DEPARTURE_FROM_ONE_LOOP.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .analyze_stage1 import B1, REF_L12, _run_frozen_ensemble, classify, interpolate, inventory, per_chain_g
from .frozen import ANALYSIS, SOURCE, require, sha256

NCAL, CAL_SEED = 2000, 20261029
VOLS = (12, 16, 24, 32)


def _load_K():
    spec = importlib.util.spec_from_file_location("rg1_frozen_running_fits", ANALYSIS / "running_fits.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.K


def fit_g(L, g, eg, KK, Kc, model, fixed=None):
    """Same models and g-space least squares as the frozen running_fits_g.py."""
    x = np.log(L / 12)

    def ymod(p):
        if model == "M1": return p[0] + p[1] * x
        if model == "MCUT": return p[0] + p[1] * x + p[2] * (L ** -2 - 12.0 ** -2)
        a = p[1] if fixed is None else fixed
        return Kc + p[0] + a * (KK - Kc)
    npar = {"M1": 2, "MCUT": 3, "MPT": 2 if fixed is None else 1}[model]
    p0 = [1 / 5.8, B1, 0.0][:npar] if model != "MPT" else ([1 / 5.8, 1.0] if fixed is None else [1 / 5.8])
    rf = least_squares(lambda p: (1 / ymod(p) - g) / eg, p0)
    C = np.linalg.pinv(rf.jac.T @ rf.jac)
    return rf.x, np.sqrt(np.diag(C)), float(np.sum(rf.fun ** 2)), len(L) - npar, ymod


def kernels(rows, K):
    ref = [r for r in rows if r["L"] == 12][0]
    KK = np.array([K(r["L"], r["z_RB"][0], ref["g_RB"][0], ref["z_RB"][0])[0] for r in rows])
    Kc = np.array([K(r["L"], r["z_RB"][0], ref["g_RB"][0], ref["z_RB"][0])[1] for r in rows])
    return KK, Kc


def arrays(rows):
    return (np.array([r["L"] for r in rows], float), np.array([r["g_RB"][0] for r in rows]),
            np.array([r["g_RB"][1] for r in rows]))


def window_fits(rows, ref_rows, K):
    """M1/MCUT/MPT on `rows`, kernels anchored at the L12 row of ref_rows (as the frozen script)."""
    KK, Kc = kernels(ref_rows, K)
    idx = [i for i, r in enumerate(ref_rows) if r in rows]
    L, g, eg = arrays(rows); KK, Kc = KK[idx], Kc[idx]
    out = {"L": [int(v) for v in L]}
    p, e, c2, dof, _ = fit_g(L, g, eg, KK, Kc, "M1"); out["M1"] = {"B": [p[1], e[1]], "alpha_B": [p[1] / B1, e[1] / B1], "chi2": c2, "dof": dof}
    p, e, c2, dof, _ = fit_g(L, g, eg, KK, Kc, "MCUT"); out["MCUT"] = {"B": [p[1], e[1]], "C": [p[2], e[2]], "chi2": c2, "dof": dof}
    p, e, c2, dof, _ = fit_g(L, g, eg, KK, Kc, "MPT"); out["MPT"] = {"alpha": [p[1], e[1]], "chi2": c2, "dof": dof}
    c1 = fit_g(L, g, eg, KK, Kc, "MPT", 1.0)[2]; c0 = fit_g(L, g, eg, KK, Kc, "MPT", 0.0)[2]
    out["MPT"]["dchi2_alpha1"] = c1 - c2; out["MPT"]["dchi2_alpha0"] = c0 - c2
    return out


def calibrate(rows, K, n=NCAL, seed=CAL_SEED):
    KK, Kc = kernels(rows, K); L, g, eg = arrays(rows)
    rng = np.random.default_rng(seed); res = {}
    for alpha in (0.0, 1.0):
        _, _, _, _, ymod = fit_g(L, g, eg, KK, Kc, "MPT", alpha)
        A = fit_g(L, g, eg, KK, Kc, "MPT", alpha)[0]
        gt = 1 / ymod(A)
        hits = {"M1": 0, "MCUT": 0, "both": 0}; zs = {"M1": [], "MCUT": []}
        for _ in range(n):
            gs = gt + eg * rng.standard_normal(len(L))
            z = {}
            for m in ("M1", "MCUT"):
                p, e, *_ = fit_g(L, gs, eg, KK, Kc, m); z[m] = p[1] / e[1]; zs[m].append(z[m])
            hits["M1"] += z["M1"] > 1.96; hits["MCUT"] += z["MCUT"] > 1.96; hits["both"] += (z["M1"] > 1.96 and z["MCUT"] > 1.96)
        key = "alpha0_false_positive_rate" if alpha == 0 else "alpha1_power"
        res[key] = {k: v / n for k, v in hits.items()}
        res[f"alpha{int(alpha)}_median_B_over_sigma"] = {k: float(np.median(v)) for k, v in zs.items()}
        res[f"alpha{int(alpha)}_truth_g"] = gt.tolist()
    res.update(n=n, seed=seed, window=[int(v) for v in L], errors=eg.tolist())
    return res


def summarize(r, chains):
    entry = {"N": r["N"], "z_RB": r["z_RB"], "g_RB": r["g_RB"], "g_raw": r["g_raw"], "U4_RB": r["U4_RB"],
             "tau_int": r["tau_int_sweepsets"], "f": r["f_conditional_variance_fraction"]}
    cg = np.array([c["g"][0] for c in chains]); ce = np.array([c["g"][1] for c in chains])
    w = 1 / ce ** 2; mean = float(np.sum(w * cg) / np.sum(w))
    scatter = float(cg.std(ddof=1) / math.sqrt(len(cg)))
    zs = np.array([c["z"][0] for c in chains])
    entry["replica"] = {"chi2": float(np.sum(w * (cg - mean) ** 2)), "dof": len(cg) - 1, "scatter_error": scatter, "n_chains": len(cg)}
    entry["g_err_used"] = max(r["g_RB"][1], scatter)
    entry["z_err_used"] = max(r["z_RB"][1], float(zs.std(ddof=1) / math.sqrt(len(zs))))
    j = r["joint"]; ly = {}
    for eta in ("4.5", "6.0"):
        gq, gs, g0 = j[f"Gq_{eta}"], j[f"Gs_{eta}"], j["g_RB_tau0"]
        d1 = (gq[0] - gs[0]) / max(math.hypot(gq[1], gs[1]), 1e-300)
        d2 = (g0[0] - gq[0]) / max(math.hypot(g0[1], gq[1]), 1e-300)
        ly[eta] = {"Gs<Gq_sigma": d1, "Gq<g_sigma": d2, "status": "FAIL" if min(d1, d2) < -3 else "PASS"}
    entry["lee_yang_order"] = ly
    entry["lebowitz"] = "FAIL" if r["g_RB"][0] < -3 * r["g_RB"][1] else "PASS"
    return entry


def accounting(sacct_path, ceiling, tpc=2):
    if not sacct_path or not Path(sacct_path).is_file():
        return {"available": False}
    lines = [ln for ln in Path(sacct_path).read_text().splitlines() if ln.strip()]
    hdr = lines[0].split("|"); rows = [dict(zip(hdr, ln.split("|"))) for ln in lines[1:]]
    slurm = 0.0; states = {}
    for r in rows:
        if "." in r["JobID"]:
            continue
        slurm += int(r["AllocCPUS"]) * int(r["ElapsedRaw"]) / 3600
        s = r["State"].split()[0]; states[s] = states.get(s, 0) + 1
    return {"available": True, "slurm_cpu_hours": slurm, "physical_core_hours": slurm / tpc, "threads_per_core": tpc,
            "states": states, "ceiling_physical_core_hours": ceiling, "within_ceiling": slurm / tpc <= ceiling}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--stage1-plan", required=True); ap.add_argument("--stage1-data", required=True)
    ap.add_argument("--stage1-adj", required=True)
    ap.add_argument("--sacct"); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ncal", type=int, default=NCAL, help="tests only; production uses the default 2,000")
    a = ap.parse_args(argv)
    require()
    plan = json.loads(Path(a.plan).read_text()); p1 = json.loads(Path(a.stage1_plan).read_text())
    if sha256(a.stage1_plan) != plan["stage1"]["plan_sha256"] or sha256(a.stage1_adj) != plan["stage1"]["adjudication_sha256"]:
        raise SystemExit("Stage 1 plan/adjudication differ from the hashes pinned in the Stage 2 plan")
    s1 = json.loads(Path(a.stage1_adj).read_text())
    d2, d1 = Path(a.data), Path(a.stage1_data); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pool_map = plan["stage1"]["pooled_with"]
    inv2 = inventory(plan, d2)
    p1 = dict(p1, tasks=[t for t in p1["tasks"] if t["ensemble"] in pool_map.values()])
    inv1 = inventory(p1, d1)
    ens2 = {e["ensemble"]: e for e in plan["ensembles"]}

    def files(inv, plan_, data, k):
        return [str(data / k / f"chain_{inv[t['task']]['chain']:02d}" / "chain.bin")
                for t in plan_["tasks"] if t["ensemble"] == k and inv[t["task"]]["usable"]]
    f2 = {k: files(inv2, plan, d2, k) for k in ens2}
    f1 = {k: files(inv1, p1, d1, k) for k in pool_map.values()}
    status = {k: {"stage": 2, "planned": e["chains"], "usable": len(f2[k]), "valid": len(f2[k]) >= 8} for k, e in ens2.items()}
    status.update({k: {"stage": 1, "planned": 0, "usable": len(v), "valid": len(v) >= 8} for k, v in f1.items()})
    # analysis sets: final (pooled at L24/L32) and Stage 2-only subsets at L24/L32
    sets = {}
    for k, e in ens2.items():
        L = int(k[1:].split("_")[0]); side = k.split("_")[1]          # volume label from the plan name (Q<L>_<side>)
        if k in pool_map:
            if status[k]["valid"] and status[pool_map[k]]["valid"]:
                sets[f"L{L}_{side}"] = (e, f1[pool_map[k]] + f2[k])
            if status[k]["valid"]:
                sets[f"{k}_only"] = (e, f2[k])
        elif status[k]["valid"]:
            sets[f"L{L}_{side}"] = (e, f2[k])
    jobs = {n: (str(out / f"ensemble_{n}.json"), e["L"], e["m0sq"], e["lambda"], 8 * len(fs), fs) for n, (e, fs) in sets.items()}
    allfiles = sorted({f for _, fs in sets.values() for f in fs})
    Lof = {f: e["L"] for e, fs in sets.values() for f in fs}; mof = {f: e["m0sq"] for e, fs in sets.values() for f in fs}
    with ProcessPoolExecutor(a.workers) as pool:
        frozen = dict(zip(jobs, pool.map(_run_frozen_ensemble, jobs.values())))
        pc = dict(zip(allfiles, pool.map(per_chain_g, allfiles, [Lof[f] for f in allfiles], [mof[f] for f in allfiles],
                                         [0.4] * len(allfiles))))
    summ, errors = {}, {}
    for n, r in frozen.items():
        if "error" in r:
            errors[n] = r["error"]; continue
        summ[n] = summarize(r, [pc[f] for f in sets[n][1]])
    res = {"schema": "rg1-stage2-adjudication-v1", "plan_sha256": sha256(a.plan), "stage1_plan_sha256": sha256(a.stage1_plan),
           "stage1_adjudication_sha256": sha256(a.stage1_adj), "ensemble_status": status,
           "inventory_stage2": {k: {kk: v[kk] for kk in ("ensemble", "chain", "captured", "usable")} | {"qc": v.get("qc", {}).get("status")}
                                for k, v in inv2.items()},
           "stage1_pooled_chains_reverified": {k: len(v) for k, v in f1.items()},
           "ensembles": summ, "ensemble_errors": errors}

    def zg(n):
        s = summ[n]; return {"z": [s["z_RB"][0], s["z_err_used"]], "g": [s["g_RB"][0], s["g_err_used"]]}
    interp = {}
    for L in VOLS:
        lo, hi = f"L{L}_lo", f"L{L}_hi"
        if lo in summ and hi in summ:
            interp[L] = interpolate(zg(lo), zg(hi)); interp[L]["rel_err"] = interp[L]["g_at_z3"][1] / interp[L]["g_at_z3"][0]
            interp[L]["target_rel"] = plan["design"]["targets_rel"][str(L)]
        else:
            interp[L] = {"bracket": "FAIL", "reason": "ensemble invalid or analysis error"}
    res["z3"] = interp

    ctl = {}
    for k, p in pool_map.items():
        n = f"{k}_only"
        if n in summ and p in s1["ensembles"]:
            a2, e1 = zg(n), s1["ensembles"][p]
            b1 = {"z": [e1["z_RB"][0], e1["z_err_used"]], "g": [e1["g_RB"][0], e1["g_err_used"]]}
            c = {q: {"stage2": a2[q], "stage1": b1[q], "sigma": (a2[q][0] - b1[q][0]) / math.hypot(a2[q][1], b1[q][1])} for q in ("z", "g")}
            for q in ("z", "g"):
                c[q]["status"] = classify(c[q]["sigma"])
            ctl[f"stage2_vs_stage1_{p}"] = c
    if "L12_hi" in summ:
        s = zg("L12_hi")
        c = {q: {"value": s[q], "ref": REF_L12[q], "sigma": (s[q][0] - REF_L12[q][0]) / math.hypot(s[q][1], REF_L12[q][1])} for q in ("z", "g")}
        for q in ("z", "g"):
            c[q]["status"] = classify(c[q]["sigma"])
        ctl["L12_reproduction_Q12_hi"] = c
    ctl["stage1_controls_carried"] = {k: (v.get("status") or {q: x.get("status") for q, x in v.items() if isinstance(x, dict)})
                                      for k, v in s1["controls"].items()}
    res["controls"] = ctl

    def any_fail(v):
        if isinstance(v, dict):
            return v.get("status") == "FAIL" or any(any_fail(x) for x in v.values())
        return v == "FAIL"
    prereq = {"production_valid": all(s["valid"] for s in status.values()),
              "brackets": {L: interp[L].get("bracket") for L in VOLS},
              "controls_fail": [k for k, v in ctl.items() if any_fail(v)],
              "lee_yang_or_lebowitz_fail": [k for k, s in summ.items() if s["lebowitz"] == "FAIL" or any(x["status"] == "FAIL" for x in s["lee_yang_order"].values())],
              "analysis_errors": sorted(errors)}
    prereq["pass"] = (prereq["production_valid"] and all(v == "PASS" for v in prereq["brackets"].values())
                      and not prereq["controls_fail"] and not prereq["lee_yang_or_lebowitz_fail"] and not errors)
    res["prerequisites"] = prereq

    fits = None
    if all(interp[L].get("bracket") == "PASS" for L in VOLS):
        base = [r for r in json.loads((SOURCE / "results/scaling_final.json").read_text()) if r["L"] <= 10]
        new = [{"L": L, "m0sq": f"interp_z3_L{L}", "N": None, "z_RB": [3.0, 0.0], "g_RB": interp[L]["g_at_z3"]} for L in VOLS]
        rows = base + new
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); (td / "data").mkdir()
            shutil.copy(SOURCE / "results/batch_check.json", td / "data/batch_check.json")
            (td / "rows.json").write_text(json.dumps(rows))
            for script in ("running_fits.py", "pt_compare.py", "running_fits_g.py"):
                shutil.copy(ANALYSIS / script, td / script)
            pt = subprocess.run([sys.executable, "-B", "pt_compare.py", "rows.json", "pt.json"], cwd=td, capture_output=True, text=True)
            rf = subprocess.run([sys.executable, "-B", "running_fits_g.py", "rows.json", "fits.json"], cwd=td, capture_output=True, text=True)
            fits = {"rows": rows,
                    "pt_compare": json.loads((td / "pt.json").read_text()) if pt.returncode == 0 else pt.stderr[-2000:],
                    "running_fits_g": json.loads((td / "fits.json").read_text()) if rf.returncode == 0 else rf.stderr[-2000:]}
        K = _load_K()
        fits["L16-32_sensitivity"] = window_fits([r for r in new if r["L"] >= 16], new, K)
        y12, ey12 = 1 / new[0]["g_RB"][0], new[0]["g_RB"][1] / new[0]["g_RB"][0] ** 2
        y32, ey32 = 1 / new[3]["g_RB"][0], new[3]["g_RB"][1] / new[3]["g_RB"][0] ** 2
        fits["endpoint_B_12_32"] = [(y32 - y12) / math.log(32 / 12), math.hypot(ey12, ey32) / math.log(32 / 12)]
        if isinstance(fits["pt_compare"], list):
            fits["pulls_vs_MPT_scheme"] = {p["L"]: (p["dy_meas"][0] - p["dy_MPT_scheme"]) / p["dy_meas"][1]
                                          for p in fits["pt_compare"] if p["L"] in (16, 24, 32) and p["dy_meas"][1] > 0}
        fits["calibration_L12-32"] = calibrate(new, K, n=a.ncal)
    res["fits"] = fits

    claim = {"primary_window": "L12-32", "primary_model": "MCUT"}
    if fits and isinstance(fits["running_fits_g"], dict):
        w = fits["running_fits_g"]["L12-32"]
        iv = lambda v: [v[0] - 1.96 * v[1], v[0] + 1.96 * v[1]]
        claim.update(B_MCUT=w["MCUT"]["B"], B_MCUT_95=iv(w["MCUT"]["B"]), B_M1=w["M1"]["B"], B_M1_95=iv(w["M1"]["B"]),
                     alpha_MPT=w["MPT"]["alpha"], alpha_MPT_95=iv(w["MPT"]["alpha"]), b1=B1)
        fpr = fits["calibration_L12-32"]["alpha0_false_positive_rate"]["both"]
        if not prereq["pass"]:
            claim["verdict"] = "INCONCLUSIVE_PREREQUISITE_FAIL"
        elif claim["B_MCUT_95"][0] > 0 and claim["B_M1_95"][0] > 0 and fpr <= 0.05:
            claim["verdict"] = "POSITIVE_RUNNING_RESOLVED"
        elif claim["B_MCUT_95"][1] < 0:
            claim["verdict"] = "NEGATIVE_SLOPE_RESOLVED"
        else:
            claim["verdict"] = "NOT_RESOLVED_BOUNDED"
        al = w["MPT"]["alpha"]
        claim["pt"] = "CONSISTENT_WITH_ONE_LOOP" if abs(al[0] - 1) <= 1.96 * al[1] else "DEPARTURE_FROM_ONE_LOOP"
    else:
        claim["verdict"] = "INCONCLUSIVE_PREREQUISITE_FAIL"
    res["claim"] = claim
    res["accounting"] = accounting(a.sacct, plan["resource_ceiling_core_hours"])
    (out / "STAGE2_RESULT.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({"z3": {L: interp[L].get("g_at_z3") for L in VOLS}, "brackets": prereq["brackets"],
                      "controls_fail": prereq["controls_fail"], "claim": claim}, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
