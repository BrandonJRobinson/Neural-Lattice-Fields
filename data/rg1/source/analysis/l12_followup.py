"""Frozen-window follow-up: correlated jackknife for bias ratios, one-atom LY estimator and LP bracket.

Same-ensemble exact reweighting at eta = 4.5, 6 (Z2 mirrored); delete-one-block jackknife.
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import ly_bracket as lb  # noqa: E402
import zm  # noqa: E402

out = sys.argv[1]; L = int(sys.argv[2]); m0sq = float(sys.argv[3]); lam = float(sys.argv[4]); nb = int(sys.argv[5])
files = sys.argv[6:]
V = L ** 4
parts = [zm.load(f) for f in files]
d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
n = d["u"].size
c0 = zm.conditional(d["S2"], d["S3"], lam=lam, m0sq=m0sq, V=V, L=L, kmax=4)
r2, r4 = c0["mom"][2], c0["mom"][4]
S2m = np.concatenate([d["S2"], d["S2"]]); S3m = np.concatenate([d["S3"], -d["S3"]])
c00 = zm.conditional(S2m, S3m, lam=lam, m0sq=m0sq, V=V, L=L, kmax=0)
B = lambda a: zm.blocks_mean(a, nb)
arrs = [B(r2), B(r4), B(d["Gp"])]
ETAS = [4.5, 6.0]
for eta in ETAS:
    ce = zm.conditional(S2m, S3m, lam=lam, m0sq=m0sq, V=V, L=L, eta=eta, kmax=2)
    lw = ce["logZ"] - c00["logZ"]; lw -= lw.max(); w = np.exp(lw)
    arrs += [B(w[:n] + w[n:]), B(w[:n] * ce["mom"][1][:n] + w[n:] * ce["mom"][1][n:]),
             B(w[:n] * ce["mom"][2][:n] + w[n:] * ce["mom"][2][n:])]


def cubic(a1, q, s, var_q, var_s, var_a1):
    D = np.array([[1, 0], [1, ETAS[0] ** 2], [1, 3 * ETAS[0] ** 2], [1, ETAS[1] ** 2], [1, 3 * ETAS[1] ** 2]], float)
    y = np.array([a1, q[0], s[0], q[1], s[1]])
    W = np.diag([1 / var_a1, 1 / var_q[0], 1 / var_s[0], 1 / var_q[1], 1 / var_s[1]])
    return np.linalg.solve(D.T @ W @ D, D.T @ W @ y)


def fun(a):
    a1 = a[0]
    z = zm.second_moment_z(a1 * L * L, a[2], L)
    g = 3 * z ** 4 * (1 - a[1] / (3 * a1 ** 2))
    q, s, Gq, Gs, G1 = [], [], [], [], []
    for i, eta in enumerate(ETAS):
        bw, b1, b2 = a[3 + 3 * i], a[4 + 3 * i], a[5 + 3 * i]
        mu = b1 / bw
        qq = mu / eta; ss = b2 / bw - mu * mu
        q.append(qq); s.append(ss)
        gq = 6 * z ** 4 * (a1 - qq) / (eta * eta * a1 * a1); gs = 3 * z ** 4 * (qq - ss) / (eta * eta * a1 * a1)
        Gq.append(gq); Gs.append(gs); G1.append(gq * gq / gs)
    q = np.array(q); s = np.array(s)
    vq = s / np.array(ETAS) ** 2; vs = 2 * s ** 2
    cub = []
    for va1 in (2 * a1 ** 2, 0.5 * a1 ** 2):
        a1h, a3h = cubic(a1, q, s, vq, vs, va1)
        cub.append(-6 * z ** 4 * a3h / a1h ** 2)
    vals = [z, g, a1, *q, *s, *Gq, *Gs, *G1, *cub]
    vals += [cub[0] / g, cub[1] / g, G1[0] / g, G1[1] / g, Gq[0] / g, Gq[1] / g, Gs[0] / g, Gs[1] / g]
    return np.array(vals)


names = ["z", "g", "a1", "q4.5", "q6", "s4.5", "s6", "Gq4.5", "Gq6", "Gs4.5", "Gs6", "gLY1_4.5", "gLY1_6",
         "cubic_equalN", "cubic_anchor_precise", "ratio_cubic_equalN", "ratio_cubic_anchor_precise",
         "ratio_gLY1_4.5", "ratio_gLY1_6", "ratio_Gq4.5", "ratio_Gq6", "ratio_Gs4.5", "ratio_Gs6"]
val, err, jk = zm.jackknife(fun, arrs)
res = {nm: [float(v), float(e)] for nm, v, e in zip(names, val, err)}
# LP bracket on central data with JK errors as tolerances (both radii, q and s)
a1 = val[2]; z = val[0]
taus = [e * e * a1 / 2 for e in ETAS]
R = [val[3] / a1, val[4] / a1]
Dd = [(val[3] - val[5]) / (ETAS[0] ** 2 * a1 * a1), (val[4] - val[6]) / (ETAS[1] ** 2 * a1 * a1)]
# JK errors of R and D
iR = [names.index("q4.5"), names.index("q6")]
Rj = np.stack([jk[:, iR[0]] / jk[:, 2], jk[:, iR[1]] / jk[:, 2]], 1)
Dj = np.stack([(jk[:, 3] - jk[:, 5]) / (ETAS[0] ** 2 * jk[:, 2] ** 2), (jk[:, 4] - jk[:, 6]) / (ETAS[1] ** 2 * jk[:, 2] ** 2)], 1)
fac = np.sqrt((nb - 1) ** 2 / nb)
sR = (Rj.std(0) * fac).tolist(); sD = (Dj.std(0) * fac).tolist()
exact = lb.lp_bracket(taus, R, Dd)
tol1 = lb.lp_bracket(taus, R, Dd, R_tol=sR, D_tol=sD)
tol2 = lb.lp_bracket(taus, R, Dd, R_tol=[2 * x for x in sR], D_tol=[2 * x for x in sD])
est = lb.ly_estimate(taus, R, Dd, sR, sD)
k = 3 * z ** 4
res["LP_exact_central"] = [k * exact["U4_min"], k * exact["U4_max"]] if exact["feasible"] else exact
res["LP_box_1sigma"] = [k * tol1["U4_min"], k * tol1["U4_max"]]
res["LP_box_2sigma"] = [k * tol2["U4_min"], k * tol2["U4_max"]]
res["LY_estimate"] = {"g_mid": k * est["U_mid"], "g_min": k * est["U_min"], "g_max": k * est["U_max"], "t_star": est["t_star"]}
res["N"] = int(n); res["nblock"] = nb
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
