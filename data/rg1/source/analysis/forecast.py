"""RG-1 cost and power forecasts from measured per-configuration noise, tau_int and sweep cost [FORECAST].

per-config sd of the g estimator: sd_L = sigma_g * sqrt(N / (2 tau)); sweeps needed: N = 2 tau (sd/sigma)^2.
Reference coupling for relative targets: g ~ in-session values / PT expectation (5.8 at L<=16, 5.4 at L>=24).
"""
import json

import numpy as np

b1 = 3 / (16 * np.pi ** 2)
rows = {}
for r in json.load(open("data/scaling_final.json")):
    if r["L"] not in rows or r["N"] > rows[r["L"]]["N"]:
        rows[r["L"]] = r


def unit(L):
    r = rows[L]
    tau = r["tau_int"]["infl_rb"]
    sd = r["g_RB"][1] * np.sqrt(r["N"] / (2 * tau))
    return sd, tau, r["seconds_per_sweepset"]


def ch(L, rel, gref):
    sd, tau, sps = unit(L)
    sig = rel * gref
    return 2 * tau * (sd / sig) ** 2 * sps / 3600


def sB(Ls, rel, g12=5.8, B=b1, cut=True):
    Ls = np.array(Ls, float); y = 1 / g12 + B * np.log(Ls / 12); g = 1 / y; sy = np.array(rel) / g
    cols = [np.ones_like(Ls), np.log(Ls / 12)] + ([Ls ** -2 - 12.0 ** -2] if cut else [])
    D = np.array(cols).T
    return float(np.sqrt(np.linalg.inv(D.T @ np.diag(1 / sy ** 2) @ D)[1, 1]))


Beff = b1 - 2.27e-4 * 5.8
target = Beff / 2.80
gref = {12: 5.8, 16: 5.6, 24: 5.4, 32: 5.3}
res = {"units": {L: dict(zip(["sd_g_per_config", "tau_int", "sec_per_sweep"], unit(L))) for L in sorted(rows)}}
designs = {}
for small in (0.002, 0.003):
    lo, hi = 1e-4, 0.05
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if sB([12, 16, 24, 32], [small, 1.5 * small, m, m]) > target:
            hi = m
        else:
            lo = m
    cost = ch(12, small, 5.8) + ch(16, 1.5 * small, 5.6) + ch(24, lo, 5.4) + ch(32, lo, 5.3)
    designs[f"L12={small},L16={1.5*small}"] = {"L24_L32_rel": lo, "core_hours": cost,
                                               "per_volume": {L: ch(L, r_, gref[L]) for L, r_ in [(12, small), (16, 1.5 * small), (24, lo), (32, lo)]}}
eq = 0.0
lo, hi = 1e-4, 0.05
for _ in range(60):
    m = 0.5 * (lo + hi)
    if sB([12, 16, 24, 32], [m] * 4) > target:
        hi = m
    else:
        lo = m
designs["equal"] = {"rel": lo, "core_hours": sum(ch(L, lo, gref[L]) for L in (12, 16, 24, 32))}
stage1 = ch(24, 0.02, 5.4) + ch(32, 0.02, 5.3)
res["designs"] = designs
res["stage1_core_hours_single_mass"] = stage1
res["stage1_core_hours_with_25pct_bracketing"] = stage1 * 1.25
# Stage-1 endpoint contrast
r12 = rows[12]
s12 = r12["g_RB"][1] / r12["g_RB"][0] ** 2
s32 = 0.02 / 5.3
sigB = np.hypot(s12, s32) / np.log(32 / 12)
res["stage1_endpoint"] = {"sigma_B": sigB, "zscore_B0.0175": 0.0175 / sigB, "zscore_B0.0166": 0.0166 / sigB}
# Stage-1 projected alpha precision: current ladder + L24, L32 at 2% (MPT shape ~ b1 log for L>=12)
fits = json.load(open("data/running_fits_final.json"))
res["current_alpha"] = fits["Lmin=6"]["MPT_alpha"]["alpha"]
Lfit = [r["L"] for r in sorted(json.load(open("data/scaling_final.json")), key=lambda r: r["L"]) if r["L"] <= 16]
pts = json.load(open("data/pt_compare_final.json"))
K = {p["L"]: p["dy_MPT_scheme"] for p in pts}
sy = {p["L"]: max(p["dy_meas"][1], 1e-9) for p in pts}
Ls = [L for L in sorted(set(Lfit)) if L != 12] + [24, 32]
X = []; W = []
for L in Ls:
    k = K[L] if L in K and L <= 16 else b1 * np.log(L / 12) * 0.93
    s = sy[L] if L <= 16 else np.hypot(0.02 / 5.4, s12)
    X.append(k); W.append(1 / s ** 2)
X = np.array(X); W = np.array(W)
res["stage1_projected_sigma_alpha"] = float(1 / np.sqrt(np.sum(W * X * X)))
json.dump(res, open("data/forecast.json", "w"), indent=1)
print(json.dumps(res, indent=1, default=float))
