"""Full zero-mode analysis of one (or several pooled) h=0 ensembles.

Outputs (JSON):
  * z (second-moment, project convention), raw and Rao-Blackwell U4 and g_R = 3 z^4 U4
  * conditional-variance fraction f and the per-configuration RB/raw variance ratio for the
    kappa_4 influence function; integrated autocorrelation times
  * exact same-ensemble source response q(eta), s(eta) on a grid, by exact uniform-mode
    reweighting (Z2-mirrored), with reweighting effective-sample fractions
  * Lee-Yang secant/tangent couplings G_q(eta), G_s(eta)
  * population value of the frozen common-window cubic q+s+anchor functional (|eta|=4.5,6)
    under several declared weighting models
Errors: delete-one-block jackknife over contiguous blocks (blocks >> tau_int).
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import zm  # noqa: E402

ETA_GRID = [0.0, 0.75, 1.5, 2.25, 3.0, 3.75, 4.5, 5.25, 6.0, 6.75, 7.5, 9.0]
WINDOW = (4.5, 6.0)


def cubic_fit(a1, q, s, etas, model):
    """Population value of the frozen cubic q+s+anchor GLS functional.

    model: dict with 'var_q','var_s' arrays (per radius) and 'var_a1' scalar.
    Returns g/(-6 z^4) * ... i.e. returns a3 and a1_hat.
    """
    rows, y, w = [], [], []
    rows.append([1.0, 0.0]); y.append(a1); w.append(1.0 / model["var_a1"])
    for i, e in enumerate(etas):
        rows.append([1.0, e * e]); y.append(q[i]); w.append(1.0 / model["var_q"][i])
        rows.append([1.0, 3 * e * e]); y.append(s[i]); w.append(1.0 / model["var_s"][i])
    D = np.array(rows); y = np.array(y); W = np.diag(w)
    beta = np.linalg.solve(D.T @ W @ D, D.T @ W @ y)
    return beta  # (a1_hat, a3_hat)


def main():
    args = sys.argv[1:]
    out_json = args[0]
    L = int(args[1]); m0sq = float(args[2]); lam = float(args[3]); nblock = int(args[4])
    files = args[5:]
    V = L ** 4
    parts = [zm.load(f) for f in files]
    # equal-length truncation per chain to make blocks per chain contiguous
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n = data["u"].size
    u, S2, S3, Gp = data["u"], data["S2"], data["S3"], data["Gp"]
    X = L * u
    res = {"L": L, "m0sq": m0sq, "lambda": lam, "N": int(n), "files": files, "nblock": nblock}

    # ---- conditional quantities at h=0
    c0 = zm.conditional(S2, S3, lam=lam, m0sq=m0sq, V=V, L=L, eta=0.0)
    r1, r2, r3, r4 = c0["mom"][1:5]
    vcond = r2 - r1 ** 2
    res["f_conditional_variance_fraction"] = float(vcond.mean() / np.mean(X ** 2))
    res["mu_c2_hartree"] = float(m0sq + 12 * lam * S2.mean() / V)
    infl_raw = X ** 4 - 6 * np.mean(X ** 2) * X ** 2
    infl_rb = r4 - 6 * np.mean(r2) * r2
    res["per_config_var_ratio_kappa4_RB_over_raw"] = float(infl_rb.var() / infl_raw.var())
    taus = {}
    for name, s in [("X2", X ** 2), ("r2", r2), ("infl_raw", infl_raw), ("infl_rb", infl_rb), ("Gp", Gp), ("S2", S2), ("S3", S3)]:
        t, wdw = zm.tau_int(s)
        taus[name] = float(t)
    res["tau_int_sweepsets"] = taus

    # ---- block arrays
    B = lambda a: zm.blocks_mean(a, nblock)
    blk = {"X2": B(X ** 2), "X4": B(X ** 4), "r2": B(r2), "r4": B(r4), "Gp": B(Gp)}

    def zfun(G0x, Gpx):
        return zm.second_moment_z(G0x * L * L, Gpx, L)  # G0 = V<u^2> = L^2 <X^2>

    def obs(a, rb):
        x2, x4 = (a[2], a[3]) if rb else (a[0], a[1])
        z = zfun(x2, a[4])
        U4 = 1 - x4 / (3 * x2 ** 2)
        return np.array([z, U4, 3 * z ** 4 * U4, x2])

    blist = [blk["X2"], blk["X4"], blk["r2"], blk["r4"], blk["Gp"]]
    for tag, rb in [("raw", False), ("RB", True)]:
        val, err, _ = zm.jackknife(lambda a: obs(a, rb), blist)
        res[f"z_{tag}"] = [float(val[0]), float(err[0])]
        res[f"U4_{tag}"] = [float(val[1]), float(err[1])]
        res[f"g_{tag}"] = [float(val[2]), float(err[2])]
        res[f"a1_{tag}"] = [float(val[3]), float(err[3])]

    # ---- exact same-ensemble source response, Z2 mirrored (S3 -> -S3, X -> -X)
    S2m = np.concatenate([S2, S2]); S3m = np.concatenate([S3, -S3])
    resp = {}
    q_blocks, s_blocks = {}, {}
    for eta in ETA_GRID:
        if eta == 0.0:
            continue
        ce = zm.conditional(S2m, S3m, lam=lam, m0sq=m0sq, V=V, L=L, eta=eta, kmax=2)
        c00 = zm.conditional(S2m, S3m, lam=lam, m0sq=m0sq, V=V, L=L, eta=0.0, kmax=0)
        lw = ce["logZ"] - c00["logZ"]
        lw -= lw.max()
        w = np.exp(lw)
        ess_frac = float(w.sum() ** 2 / (w * w).sum() / w.size)
        # pair mirrored halves back onto original time index for blocking
        wpair = w[:n] + w[n:]
        m1pair = w[:n] * ce["mom"][1][:n] + w[n:] * ce["mom"][1][n:]
        m2pair = w[:n] * ce["mom"][2][:n] + w[n:] * ce["mom"][2][n:]
        bw, b1, b2 = B(wpair), B(m1pair), B(m2pair)

        def qs(a, eta=eta):
            mu = a[1] / a[0]
            return np.array([mu / eta, a[2] / a[0] - mu * mu])

        val, err, jk = zm.jackknife(qs, [bw, b1, b2])
        resp[str(eta)] = {"q": [float(val[0]), float(err[0])], "s": [float(val[1]), float(err[1])],
                          "reweight_ess_fraction": ess_frac}
        q_blocks[eta] = (bw, b1, b2)
    res["response"] = resp

    # ---- Lee-Yang secant/tangent couplings and frozen cubic functional, jackknifed jointly
    etas_all = [e for e in ETA_GRID if e > 0]

    def ly_and_cubic(a):
        # a = [r2blk, Gpblk, (bw,b1,b2) for each eta ...]
        a1 = a[0]
        z = zfun(a1, a[1])
        out = [z, 3 * z ** 4 * (1 - 0) ]  # placeholder replaced below
        U4 = None
        qv, sv = {}, {}
        k = 3
        for e in etas_all:
            bw, b1, b2 = a[k], a[k + 1], a[k + 2]; k += 3
            mu = b1 / bw
            qv[e] = mu / e
            sv[e] = b2 / bw - mu * mu
        U4 = 1 - a[2] / (3 * a1 ** 2)
        g0 = 3 * z ** 4 * U4
        vals = [z, g0, a1]
        for e in etas_all:
            Gq = 6 * z ** 4 * (a1 - qv[e]) / (e * e * a1 * a1)
            Gs = 3 * z ** 4 * (qv[e] - sv[e]) / (e * e * a1 * a1)
            vals += [Gq, Gs, qv[e], sv[e]]
        # cubic common-window functional under three weighting models
        e1, e2 = WINDOW
        qq = np.array([qv[e1], qv[e2]]); ss = np.array([sv[e1], sv[e2]])
        models = {
            # equal independent-sample counts per signed point, raw-moment variances, anchor same N
            "raw_equalN": {"var_q": ss / (np.array(WINDOW) ** 2), "var_s": 2 * ss ** 2, "var_a1": 2 * a1 ** 2},
            # anchor 4x more precise than each source point (anchor shared/longer)
            "anchor_precise": {"var_q": ss / (np.array(WINDOW) ** 2), "var_s": 2 * ss ** 2, "var_a1": 0.5 * a1 ** 2},
            # s down-weighted x4 (e.g. noisier connected variance)
            "s_downweighted": {"var_q": ss / (np.array(WINDOW) ** 2), "var_s": 8 * ss ** 2, "var_a1": 2 * a1 ** 2},
            # q-only with exact anchor (secant-type)
            "q_only_exact_anchor": {"var_q": ss / (np.array(WINDOW) ** 2), "var_s": 1e12 * ss ** 2, "var_a1": 1e-12 * a1 ** 2},
        }
        for mname in ["raw_equalN", "anchor_precise", "s_downweighted", "q_only_exact_anchor"]:
            a1h, a3h = cubic_fit(a1, qq, ss, WINDOW, models[mname])
            vals.append(-6 * z ** 4 * a3h / a1h ** 2)
        return np.array(vals)

    arrs = [blk["r2"], blk["Gp"], blk["r4"]]
    for e in etas_all:
        arrs += list(q_blocks[e])
    val, err, _ = zm.jackknife(ly_and_cubic, arrs)
    names = ["z", "g_RB_tau0", "a1"]
    for e in etas_all:
        names += [f"Gq_{e}", f"Gs_{e}", f"q_{e}", f"s_{e}"]
    names += ["cubic_raw_equalN", "cubic_anchor_precise", "cubic_s_downweighted", "cubic_q_only_exact_anchor"]
    res["joint"] = {nm: [float(v), float(e)] for nm, v, e in zip(names, val, err)}
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: res[k] for k in res if k not in ("response", "joint")}, indent=1))
    print("joint:")
    for nm in names:
        print(f"  {nm:28s} {res['joint'][nm][0]: .6f} +- {res['joint'][nm][1]:.6f}")
    print("reweight ESS fractions:", {k: round(v['reweight_ess_fraction'], 4) for k, v in resp.items()})


if __name__ == "__main__":
    main()
