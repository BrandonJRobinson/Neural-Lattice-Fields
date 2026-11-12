"""Cross-volume table: pinning law, RB efficiency, autocorrelation and measured cost.

For every h=0 run: z (second moment, project convention), g raw/RB with block-jackknife errors,
f = E[Var(X|psi)]/Var X, Hartree curvature mu_c^2, m_chi^2 = 1/G(0), Sigma = mu_c^2 - m_chi^2,
pinning prediction f_pred = m_chi^2/mu_c^2, per-config RB/raw variance ratio of the kappa_4 influence
and (1-f)^4, integrated autocorrelation times, wall seconds per sweep-set, and the implied
cost (core-seconds) per independent RB measurement and per 1% statistical precision on g.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import zm  # noqa: E402


def analyse(L, m0sq, lam, files, meta, nb=100):
    V = L ** 4
    parts = [zm.load(f) for f in files]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    X = L * d["u"]
    c = zm.conditional(d["S2"], d["S3"], lam=lam, m0sq=m0sq, V=V, L=L)
    r1, r2, r4 = c["mom"][1], c["mom"][2], c["mom"][4]
    v = r2 - r1 ** 2
    a1 = np.mean(r2)
    G0 = L * L * a1
    m_chi2 = 1.0 / G0
    mu_c2 = m0sq + 12 * lam * d["S2"].mean() / V
    f = v.mean() / np.mean(X ** 2)
    infl_raw = X ** 4 - 6 * np.mean(X ** 2) * X ** 2
    infl_rb = r4 - 6 * a1 * r2
    ratio = infl_rb.var() / infl_raw.var()
    t_raw = zm.tau_int(infl_raw)[0]; t_rb = zm.tau_int(infl_rb)[0]; t_gp = zm.tau_int(d["Gp"])[0]
    B = lambda a: zm.blocks_mean(a, nb)
    blist = [B(X ** 2), B(X ** 4), B(r2), B(r4), B(d["Gp"])]

    def obs(a, rb):
        x2, x4 = (a[2], a[3]) if rb else (a[0], a[1])
        z = zm.second_moment_z(x2 * L * L, a[4], L)
        return np.array([z, 3 * z ** 4 * (1 - x4 / (3 * x2 ** 2))])

    vr, er, _ = zm.jackknife(lambda a: obs(a, False), blist)
    vb, eb, _ = zm.jackknife(lambda a: obs(a, True), blist)
    n = X.size
    sps = meta.get("seconds_per_sweepset", float("nan"))
    # cost per 1% on g with RB: N_needed = n * (err/(0.01 g))^2 sweeps-sets
    N_1pct = n * (eb[1] / (0.01 * vb[1])) ** 2
    return {"L": L, "m0sq": m0sq, "N": int(n), "z_RB": [float(vb[0]), float(eb[0])],
            "g_raw": [float(vr[1]), float(er[1])], "g_RB": [float(vb[1]), float(eb[1])],
            "f": float(f), "mu_c2": float(mu_c2), "m_chi2": float(m_chi2), "Sigma": float(mu_c2 - m_chi2),
            "f_pred": float(m_chi2 / mu_c2), "per_config_ratio_RB_raw": float(ratio), "one_minus_f_4": float((1 - f) ** 4),
            "tau_int": {"infl_raw": float(t_raw), "infl_rb": float(t_rb), "Gp": float(t_gp)},
            "seconds_per_sweepset": sps, "core_hours_for_1pct_g_RB": float(N_1pct * sps / 3600.0),
            "rb_variance_gain_on_g": float((er[1] / eb[1]) ** 2)}


def main():
    spec = json.load(open(sys.argv[1]))
    rows = []
    for r in spec["runs"]:
        meta = {}
        try:
            metas = [json.loads(open(j).read().strip().splitlines()[-1]) for j in r["meta"]]
            meta = {"seconds_per_sweepset": float(np.mean([m["seconds_per_sweepset"] for m in metas]))}
        except Exception:
            pass
        rows.append(analyse(r["L"], r["m0sq"], 0.4, r["files"], meta, nb=r.get("nb", 100)))
        print(json.dumps(rows[-1]))
    json.dump(rows, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
