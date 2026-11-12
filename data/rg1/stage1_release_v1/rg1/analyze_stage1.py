"""Predeclared RG-1 Stage 1 adjudication. Frozen before any Stage 1 data exist; no option changes estimators.

Inputs: PLAN.json, the persistent data root, an sacct export. Every number comes from captured, hash-verified,
QC-PASS chains only. Ensemble analysis uses the staged package's own analyze_ensemble.py (hash-pinned,
unmodified); fits use its pt_compare.py and running_fits_g.py (hash-pinned, unmodified).

Rules
  * Production ensemble valid iff >= 8 chains captured, verified and QC PASS. Equal-length chains are pooled;
    jackknife blocks = 8 per chain (never straddling chains).
  * Replica scatter: chi^2 of per-chain g_RB about their weighted mean; reported, and the ensemble error is
    max(jackknife, replica-scatter error).
  * z = 3 interpolation per volume, linear in m0^2 between the two bracketing masses. Bracket PASS iff
    z_lo < 3 < z_hi within 2 sigma, or the nearest endpoint is within 0.3% of 3. Error: linear propagation,
    taking the larger of the uncorrelated and fully-(anti)correlated z-g combinations.
  * Controls: <= 2 sigma PASS, 2-3 sigma WARN, > 3 sigma FAIL (L12 reproduction, cross-kernel L12/L16,
    lambda = 0 exact values, reweighting identity at eta = 4.5). Lee-Yang order G_s < G_q < g and Lebowitz
    g >= 0 FAIL only if violated by > 3 sigma.
  * Stage 2 gate (execution order): projected MCUT sigma_B <= b1/2.5 with b1 = 3/(16 pi^2), from MEASURED
    per-configuration variance, tau_int and cost at L24/L32 (Stage 1) and the in-session L12/L16 units, for
    the order's allocation (L12 0.2%, L16 0.3%, L24/L32 equal relative precision), within 1,400 core-hours
    of additional cost. Failure of any production validity, bracket or control FAIL also blocks Stage 2.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .frozen import ANALYSIS, SOURCE, require, sha256, zm
from .run_chain import verify_capture

B1 = 3 / (16 * math.pi ** 2)
GATE_SIGMA_B = B1 / 2.5
STAGE2_CAP = 1400.0
REF_L12 = {"z": (3.0245, 0.0011), "g": (5.805, 0.041)}


def classify(dev):
    a = abs(dev)
    return "PASS" if a <= 2 else ("WARN" if a <= 3 else "FAIL")


def inventory(plan, data: Path):
    inv = {}
    for t in plan["tasks"]:
        d = data / t["ensemble"] / f"chain_{t['chain']:02d}"
        rec = {"task": t["task"], "ensemble": t["ensemble"], "chain": t["chain"], "captured": d.is_dir()}
        if d.is_dir():
            bad = verify_capture(d)
            rec["integrity"] = bad or "OK"
            q = d.parent / f"{d.name}.qc.json"
            rec["qc"] = json.loads(q.read_text()) if q.is_file() else {"status": "QC_MISSING"}
            rec["meta"] = json.loads((d / "chain.json").read_text())
            rec["usable"] = not bad and rec["qc"].get("status") == "PASS"
        else:
            rec["usable"] = False
        inv[t["task"]] = rec
    return inv


def _run_frozen_ensemble(args):
    out_json, L, m0, lam, nblock, files = args
    cmd = [sys.executable, "-B", str(ANALYSIS / "analyze_ensemble.py"), out_json, str(L), repr(m0), repr(lam),
           str(nblock), *files]
    p = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if p.returncode:
        return {"error": p.stderr[-3000:]}
    return json.loads(Path(out_json).read_text())


def per_chain_g(path, L, m0, lam, nblock=8):
    a = zm.load(path); V = L ** 4
    c = zm.conditional(a["S2"], a["S3"], lam=lam, m0sq=m0, V=V, L=L, eta=0.0)
    r2, r4 = c["mom"][2], c["mom"][4]
    B = lambda x: zm.blocks_mean(x, nblock)

    def obs(b):
        z = zm.second_moment_z(b[0] * L * L, b[2], L)
        return np.array([z, 3 * z ** 4 * (1 - b[1] / (3 * b[0] ** 2))])
    val, err, _ = zm.jackknife(obs, [B(r2), B(r4), B(a["Gp"])])
    return {"z": [float(val[0]), float(err[0])], "g": [float(val[1]), float(err[1])]}


def direct_source(paths, L, eta, nblock=8):
    xs = [L * zm.load(p)["u"] for p in paths]
    n = min(map(len, xs)); X = np.concatenate([x[:n] for x in xs])
    nb = nblock * len(xs)
    bx, bx2 = zm.blocks_mean(X, nb), zm.blocks_mean(X * X, nb)
    val, err, _ = zm.jackknife(lambda b: np.array([b[0] / eta, b[1] - b[0] ** 2]), [bx, bx2])
    return {"q": [float(val[0]), float(err[0])], "s": [float(val[1]), float(err[1])], "N": int(X.size)}


def interpolate(lo, hi):
    (zl, ezl), (gl, egl) = lo["z"], lo["g"]
    (zh, ezh), (gh, egh) = hi["z"], hi["g"]
    t = (3.0 - zl) / (zh - zl)
    g = gl + t * (gh - gl)
    dg_dgl, dg_dgh = 1 - t, t
    dg_dzl = (gh - gl) * (t - 1) / (zh - zl)
    dg_dzh = -(gh - gl) * t / (zh - zl)
    unc = math.sqrt((dg_dgl * egl) ** 2 + (dg_dgh * egh) ** 2 + (dg_dzl * ezl) ** 2 + (dg_dzh * ezh) ** 2)
    corr = max(math.sqrt((abs(dg_dgl * egl) + abs(dg_dzl * ezl)) ** 2 + (abs(dg_dgh * egh) + abs(dg_dzh * ezh)) ** 2), unc)
    straddle = (zl - 2 * ezl < 3.0 < zh + 2 * ezh) and zl < zh
    nearest = min(abs(zl - 3), abs(zh - 3)) / 3
    ok = (0 <= t <= 1 and straddle) or nearest <= 0.003
    return {"g_at_z3": [g, corr], "g_err_uncorrelated": unc, "t": t, "z_lo": lo["z"], "z_hi": hi["z"],
            "bracket": "PASS" if ok else "FAIL", "extrapolation": not (0 <= t <= 1)}


def unit_from(row_g_err, N, tau, sps):
    sd = row_g_err * math.sqrt(N / (2 * tau))
    return sd, tau, sps


def sigma_B(Ls, g, rel, cut=True):
    Ls = np.asarray(Ls, float); g = np.asarray(g, float); sy = np.asarray(rel) / g
    cols = [np.ones_like(Ls), np.log(Ls / 12)] + ([Ls ** -2 - 12.0 ** -2] if cut else [])
    D = np.array(cols).T
    return float(np.sqrt(np.linalg.inv(D.T @ np.diag(1 / sy ** 2) @ D)[1, 1]))


def stage2_projection(units, g, have):
    """Cost of reaching L12 0.2%, L16 0.3%, L24/L32 rel r (r solved for sigma_B <= gate) minus what exists."""
    def hours(L, rel):
        sd, tau, sps = units[L]
        need = 2 * tau * (sd / (rel * g[L])) ** 2          # sweeps for this precision
        return max(0.0, need - have.get(L, 0.0)) * sps / 3600
    lo, hi = 1e-4, 0.2
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if sigma_B([12, 16, 24, 32], [g[12], g[16], g[24], g[32]], [0.002, 0.003, m, m]) > GATE_SIGMA_B:
            hi = m
        else:
            lo = m
    rel = lo
    cost = {L: hours(L, r) for L, r in ((12, 0.002), (16, 0.003), (24, rel), (32, rel))}
    return {"L24_L32_rel_required": rel, "sigma_B_gate": GATE_SIGMA_B,
            "sigma_B_MCUT_at_design": sigma_B([12, 16, 24, 32], [g[12], g[16], g[24], g[32]], [0.002, 0.003, rel, rel]),
            "stage2_core_hours_by_volume": cost, "stage2_core_hours": sum(cost.values()),
            "within_1400": sum(cost.values()) <= STAGE2_CAP}


def accounting(sacct_path, plan):
    if not sacct_path or not Path(sacct_path).is_file():
        return {"available": False}
    lines = [ln for ln in Path(sacct_path).read_text().splitlines() if ln.strip()]
    hdr = lines[0].split("|"); rows = [dict(zip(hdr, ln.split("|"))) for ln in lines[1:]]
    alloc = cpu = 0.0; states = {}
    for r in rows:
        if "." in r["JobID"]:
            continue
        alloc += int(r["AllocCPUS"]) * int(r["ElapsedRaw"]) / 3600
        states[r["State"].split()[0]] = states.get(r["State"].split()[0], 0) + 1
    return {"available": True, "allocated_core_hours_used": alloc, "states": states,
            "ceiling": plan["resource_ceiling_core_hours"], "within_ceiling": alloc <= plan["resource_ceiling_core_hours"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--sacct"); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    require()
    plan = json.loads(Path(a.plan).read_text()); data = Path(a.data); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    inv = inventory(plan, data)
    ens = {e["ensemble"]: e for e in plan["ensembles"]}
    usable = {k: [inv[t["task"]] for t in plan["tasks"] if t["ensemble"] == k and inv[t["task"]]["usable"]] for k in ens}
    status = {}
    jobs = {}
    for k, e in ens.items():
        n = len(usable[k]); need = 8 if e["role"] == "production" else 2
        status[k] = {"planned": e["chains"], "usable": n, "valid": n >= need}
        if status[k]["valid"] and e["h"] == 0.0:
            files = [str(data / k / f"chain_{r['chain']:02d}" / "chain.bin") for r in usable[k]]
            jobs[k] = (str(out / f"ensemble_{k}.json"), e["L"], e["m0sq"], e["lambda"], 8 * n, files)
    with ProcessPoolExecutor(a.workers) as pool:
        frozen = dict(zip(jobs, pool.map(_run_frozen_ensemble, jobs.values())))
        chains = {k: list(pool.map(per_chain_g, [j for j in jobs[k][5]], [ens[k]["L"]] * len(jobs[k][5]),
                                   [ens[k]["m0sq"]] * len(jobs[k][5]), [ens[k]["lambda"]] * len(jobs[k][5])))
                  for k in jobs if ens[k]["lambda"] > 0}
    res = {"schema": "rg1-stage1-adjudication-v1", "plan_sha256": sha256(a.plan), "ensemble_status": status,
           "inventory": {k: {kk: v[kk] for kk in ("ensemble", "chain", "captured", "usable")} | {"qc": v.get("qc", {}).get("status")}
                         for k, v in inv.items()}, "ensembles": {}}
    summ = {}
    for k, r in frozen.items():
        if "error" in r:
            res["ensembles"][k] = r; continue
        e = ens[k]
        entry = {"N": r["N"], "z_RB": r["z_RB"], "g_RB": r["g_RB"], "g_raw": r["g_raw"], "U4_RB": r["U4_RB"],
                 "tau_int": r["tau_int_sweepsets"], "f": r["f_conditional_variance_fraction"]}
        if k in chains:
            cg = np.array([c["g"][0] for c in chains[k]]); ce = np.array([c["g"][1] for c in chains[k]])
            w = 1 / ce ** 2; mean = float(np.sum(w * cg) / np.sum(w)); chi2 = float(np.sum(w * (cg - mean) ** 2))
            scatter = float(cg.std(ddof=1) / math.sqrt(len(cg)))
            entry["replica"] = {"per_chain": chains[k], "chi2": chi2, "dof": len(cg) - 1, "scatter_error": scatter}
            entry["g_err_used"] = max(r["g_RB"][1], scatter)
            zs = np.array([c["z"][0] for c in chains[k]])
            entry["z_err_used"] = max(r["z_RB"][1], float(zs.std(ddof=1) / math.sqrt(len(zs))))
        else:
            entry["g_err_used"] = r["g_RB"][1]; entry["z_err_used"] = r["z_RB"][1]
        j = r["joint"]
        ly = {}
        for eta in ("4.5", "6.0"):
            gq, gs, g0 = j[f"Gq_{eta}"], j[f"Gs_{eta}"], j["g_RB_tau0"]
            d1 = (gq[0] - gs[0]) / max(math.hypot(gq[1], gs[1]), 1e-300)
            d2 = (g0[0] - gq[0]) / max(math.hypot(g0[1], gq[1]), 1e-300)
            ly[eta] = {"Gs<Gq_sigma": d1, "Gq<g_sigma": d2, "status": "FAIL" if min(d1, d2) < -3 else "PASS"}
        entry["lee_yang_order"] = ly
        entry["lebowitz"] = "FAIL" if r["g_RB"][0] < -3 * r["g_RB"][1] else "PASS"
        entry["response_q4.5"] = r["response"].get("4.5", {}).get("q")
        entry["response_s4.5"] = r["response"].get("4.5", {}).get("s")
        meta_sps = [inv[t["task"]]["meta"]["seconds_per_sweepset"] for t in plan["tasks"] if t["ensemble"] == k and inv[t["task"]]["usable"]]
        entry["seconds_per_sweep_mean"] = float(np.mean(meta_sps)) if meta_sps else None
        summ[k] = entry
    res["ensembles"].update(summ)

    def zg(k):
        s = summ[k]
        return {"z": [s["z_RB"][0], s["z_err_used"]], "g": [s["g_RB"][0], s["g_err_used"]]}

    # ---- z = 3 interpolation
    interp = {}
    for L, lo, hi in ((24, "P24_lo", "P24_hi"), (32, "P32_lo", "P32_hi")):
        if lo in summ and hi in summ:
            interp[L] = interpolate(zg(lo), zg(hi)); interp[L]["rel_err"] = interp[L]["g_at_z3"][1] / interp[L]["g_at_z3"][0]
        else:
            interp[L] = {"bracket": "FAIL", "reason": "production ensemble invalid"}
    res["z3"] = interp

    # ---- controls
    ctl = {}
    if "C12_repro" in summ:
        s = zg("C12_repro")
        ctl["L12_reproduction"] = {q: {"value": s[q], "ref": REF_L12[q], "sigma": (s[q][0] - REF_L12[q][0]) / math.hypot(s[q][1], REF_L12[q][1])} for q in ("z", "g")}
        for q in ("z", "g"):
            ctl["L12_reproduction"][q]["status"] = classify(ctl["L12_reproduction"][q]["sigma"])
    for loc, ref in (("C12_local", "C12_repro"), ("C16_local", "C16_gibbs")):
        if loc in summ and ref in summ:
            a1, b1 = zg(loc), zg(ref)
            ctl[f"cross_kernel_{loc}"] = {q: {"local_only": a1[q], "gibbs": b1[q], "sigma": (a1[q][0] - b1[q][0]) / math.hypot(a1[q][1], b1[q][1])} for q in ("z", "g")}
            for q in ("z", "g"):
                ctl[f"cross_kernel_{loc}"][q]["status"] = classify(ctl[f"cross_kernel_{loc}"][q]["sigma"])
    if "C08_gauss" in frozen and "error" not in frozen["C08_gauss"]:
        r = frozen["C08_gauss"]; L, m2 = ens["C08_gauss"]["L"], ens["C08_gauss"]["m0sq"]
        ph2 = 4 * math.sin(math.pi / L) ** 2
        exact_a1 = 1 / (L * L * m2)
        dev_g = r["g_raw"][0] / r["g_raw"][1]
        dev_a = (r["a1_raw"][0] - exact_a1) / r["a1_raw"][1]
        ctl["lambda0"] = {"g_raw": r["g_raw"], "g_sigma_from_0": dev_g, "a1": r["a1_raw"], "a1_exact": exact_a1,
                          "a1_sigma": dev_a, "Gp_exact": 1 / (ph2 + m2),
                          "status": "FAIL" if max(abs(dev_g), abs(dev_a)) > 3 else ("WARN" if max(abs(dev_g), abs(dev_a)) > 2 else "PASS")}
    for S, P in (("S24_eta4p5", "P24_hi"), ("S32_eta4p5", "P32_hi")):
        if status.get(S, {}).get("valid") and P in summ and summ[P]["response_q4.5"]:
            e = ens[S]
            paths = [str(data / S / f"chain_{r['chain']:02d}" / "chain.bin") for r in usable[S]]
            d = direct_source(paths, e["L"], e["eta"])
            rq, rs = summ[P]["response_q4.5"], summ[P]["response_s4.5"]
            sq = (d["q"][0] - rq[0]) / math.hypot(d["q"][1], rq[1]); ss = (d["s"][0] - rs[0]) / math.hypot(d["s"][1], rs[1])
            ctl[f"reweighting_identity_{S}"] = {"direct": d, "reweighted": {"q": rq, "s": rs}, "q_sigma": sq, "s_sigma": ss,
                                                "status": max((classify(sq), classify(ss)), key=["PASS", "WARN", "FAIL"].index)}
    res["controls"] = ctl

    # ---- PT comparison and fits with the frozen scripts (in-session L6-16 rows + new L24/L32 at z = 3)
    fits = None
    if all(interp[L].get("bracket") == "PASS" for L in (24, 32)):
        base = [r for r in json.loads((SOURCE / "results/scaling_final.json").read_text()) if r["L"] <= 16]
        rows = base + [{"L": L, "m0sq": f"interp_z3_L{L}", "N": None, "z_RB": [3.0, 0.0],
                        "g_RB": interp[L]["g_at_z3"]} for L in (24, 32)]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); (td / "data").mkdir()
            shutil.copy(SOURCE / "results/batch_check.json", td / "data/batch_check.json")
            (td / "rows.json").write_text(json.dumps(rows))
            for script in ("running_fits.py", "pt_compare.py", "running_fits_g.py"):
                shutil.copy(ANALYSIS / script, td / script)
            pt = subprocess.run([sys.executable, "-B", "pt_compare.py", "rows.json", "pt.json"], cwd=td, capture_output=True, text=True)
            rf = subprocess.run([sys.executable, "-B", "running_fits_g.py", "rows.json", "fits.json"], cwd=td, capture_output=True, text=True)
            fits = {"pt_compare": json.loads((td / "pt.json").read_text()) if pt.returncode == 0 else pt.stderr[-2000:],
                    "running_fits_g": json.loads((td / "fits.json").read_text()) if rf.returncode == 0 else rf.stderr[-2000:]}
        g12 = [r for r in base if r["L"] == 12][0]["g_RB"]
        y12, ey12 = 1 / g12[0], g12[1] / g12[0] ** 2
        g32 = interp[32]["g_at_z3"]; y32, ey32 = 1 / g32[0], g32[1] / g32[0] ** 2
        fits["endpoint_B_12_32"] = [(y32 - y12) / math.log(32 / 12), math.hypot(ey12, ey32) / math.log(32 / 12)]
        if isinstance(fits["pt_compare"], list):
            fits["pulls_vs_MPT_scheme"] = {p["L"]: (p["dy_meas"][0] - p["dy_MPT_scheme"]) / p["dy_meas"][1]
                                          for p in fits["pt_compare"] if p["L"] in (24, 32) and p["dy_meas"][1] > 0}
    res["fits"] = fits

    # ---- Stage 2 gate with measured units
    gate = {"production_valid": all(status[k]["valid"] for k in ("P24_lo", "P24_hi", "P32_lo", "P32_hi")),
            "brackets": {L: interp[L].get("bracket") for L in (24, 32)},
            "controls_fail": [k for k, v in ctl.items() if (v.get("status") == "FAIL" or any(isinstance(x, dict) and x.get("status") == "FAIL" for x in v.values()))],
            "lee_yang_or_lebowitz_fail": [k for k, s in summ.items() if ens[k]["lambda"] > 0 and (s["lebowitz"] == "FAIL" or any(x["status"] == "FAIL" for x in s["lee_yang_order"].values()))]}
    proj = None
    if gate["production_valid"] and all(v == "PASS" for v in gate["brackets"].values()):
        sc = {r["L"]: r for r in json.loads((SOURCE / "results/scaling_final.json").read_text())}
        fc = json.loads((SOURCE / "results/forecast.json").read_text())["units"]
        units = {L: (fc[str(L)]["sd_g_per_config"], fc[str(L)]["tau_int"], fc[str(L)]["sec_per_sweep"]) for L in (12, 16)}
        have = {12: float(sc[12]["N"]), 16: float(sc[16]["N"])}
        g = {12: sc[12]["g_RB"][0], 16: sc[16]["g_RB"][0]}
        for L, lo, hi in ((24, "P24_lo", "P24_hi"), (32, "P32_lo", "P32_hi")):
            N = summ[lo]["N"] + summ[hi]["N"]; tau = max(summ[lo]["tau_int"]["infl_rb"], summ[hi]["tau_int"]["infl_rb"])
            gz, eg = interp[L]["g_at_z3"]
            units[L] = unit_from(eg, N, tau, float(np.mean([summ[lo]["seconds_per_sweep_mean"], summ[hi]["seconds_per_sweep_mean"]])))
            have[L] = float(N); g[L] = gz
        proj = stage2_projection(units, g, have)
        proj["units"] = {L: dict(zip(("sd_g_per_config", "tau_int", "sec_per_sweep"), u)) for L, u in units.items()}
    gate["projection"] = proj
    gate["accounting"] = accounting(a.sacct, plan)
    gate["stage2_gate"] = "PASS" if (gate["production_valid"] and all(v == "PASS" for v in gate["brackets"].values())
                                     and not gate["controls_fail"] and not gate["lee_yang_or_lebowitz_fail"]
                                     and proj and proj["within_1400"]) else "STOP"
    res["stage1_gate"] = gate
    (out / "STAGE1_ADJUDICATION.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({"z3": {L: interp[L].get("g_at_z3") for L in interp}, "brackets": gate["brackets"],
                      "controls_fail": gate["controls_fail"], "stage2_gate": gate["stage2_gate"],
                      "stage2_cost": proj and proj["stage2_core_hours"]}, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
