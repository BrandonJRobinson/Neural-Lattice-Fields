"""Lee-Yang re-analysis of archived source-response bootstrap draws (NEW prospective analysis).

This does NOT replace the frozen G2 estimator.  It is a separately labelled analysis whose only
physics input is the Lee-Yang/Newman theorem for finite-volume lattice phi^4.

Input (JSON):  {"volumes": [ {"L": 12, "label": "...",
                               "draws": [ {"z": .., "a1": .., "q": {"4.5": .., "6.0": ..},
                                           "s": {"4.5": .., "6.0": ..}}, ... ] }, ... ]}
  a1 = <X^2> at h=0 (exact-Z2 anchor), q(eta) = <X>_eta/eta, s(eta) = Var_eta(X), X = L*M, eta = h L^3.
  One entry per bootstrap draw, all quantities from the SAME draw index (pairing preserved).

Per draw and radius:
  G_q = 6 z^4 (a1 - q)/(eta^2 a1^2)        (secant; rigorous lower bound on g_R)
  G_s = 3 z^4 (q - s)/(eta^2 a1^2)         (tangent; rigorous lower bound, G_s <= G_q)
  g_LY1 = G_q^2 / G_s                      (one-atom Lee-Yang Pade; rigorous lower bound, >= G_q)
Central data additionally get the LP bracket over all Lee-Yang measures (both radii jointly) and the
minimal normalised Lee-Yang violation t* (t* >> 1 => data inconsistent with ANY phi^4 response).

Diagnostics that are theorems (must hold in the population): s < q, G_s < G_q, q decreasing in eta.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import ly_bracket as lb  # noqa: E402


def per_draw(d, etas):
    z, a1 = d["z"], d["a1"]
    out = {}
    for e in etas:
        q, s = d["q"][e], d["s"][e]
        ee = float(e)
        Gq = 6 * z ** 4 * (a1 - q) / (ee * ee * a1 * a1)
        Gs = 3 * z ** 4 * (q - s) / (ee * ee * a1 * a1)
        out[e] = (Gq, Gs, Gq * Gq / Gs if Gs > 0 else np.nan)
    return out


def summarise(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    return {"median": float(np.median(v)), "p2.5": float(np.percentile(v, 2.5)), "p97.5": float(np.percentile(v, 97.5)),
            "n_finite": int(v.size)}


def main(path_in, path_out):
    data = json.load(open(path_in))
    report = {"analysis": "Lee-Yang re-analysis (new, prospective label; frozen G2 unchanged)", "volumes": []}
    for vol in data["volumes"]:
        draws = vol["draws"]
        etas = sorted(draws[0]["q"].keys(), key=float)
        pd = [per_draw(d, etas) for d in draws]
        r = {"L": vol["L"], "label": vol.get("label", ""), "n_draws": len(draws), "radii": etas}
        for e in etas:
            Gq = [p[e][0] for p in pd]; Gs = [p[e][1] for p in pd]; G1 = [p[e][2] for p in pd]
            r[f"Gq_{e}"] = summarise(Gq); r[f"Gs_{e}"] = summarise(Gs); r[f"gLY1_{e}"] = summarise(G1)
            r[f"frac_draws_s_lt_q_{e}"] = float(np.mean([d["s"][e] < d["q"][e] for d in draws]))
            r[f"frac_draws_Gs_lt_Gq_{e}"] = float(np.mean([p[e][1] < p[e][0] for p in pd]))
        # LP on median data with draw-spread as the tolerance scale
        med = lambda k, e: float(np.median([d[k][e] for d in draws]))
        z = float(np.median([d["z"] for d in draws])); a1 = float(np.median([d["a1"] for d in draws]))
        taus, R, D, sR, sD = [], [], [], [], []
        for e in etas:
            ee = float(e); tau = ee * ee * a1 / 2
            Rd = np.array([d["q"][e] / d["a1"] for d in draws])
            Dd = np.array([(d["q"][e] - d["s"][e]) / (ee * ee * d["a1"] * d["a1"]) for d in draws])
            taus.append(tau); R.append(float(np.median(Rd))); D.append(float(np.median(Dd)))
            sR.append(float(np.std(Rd))); sD.append(float(np.std(Dd)))
        est = lb.ly_estimate(taus, R, D, sR, sD)
        r["LP_central"] = {"g_mid": 3 * z ** 4 * est["U_mid"], "g_min": 3 * z ** 4 * est["U_min"],
                           "g_max": 3 * z ** 4 * est["U_max"], "t_star": est["t_star"]}
        # frozen-style cubic q+s+anchor GLS per draw (covariance from the draws; radii independent)
        Y = np.array([[d["a1"]] + sum([[d["q"][e], d["s"][e]] for e in etas], []) for d in draws])
        C = np.zeros((Y.shape[1], Y.shape[1]))
        C[0, 0] = np.var(Y[:, 0])
        for i in range(len(etas)):
            blk = slice(1 + 2 * i, 3 + 2 * i)
            C[blk, blk] = np.cov(Y[:, blk].T)
        Dm = [[1.0, 0.0]] + sum([[[1.0, float(e) ** 2], [1.0, 3 * float(e) ** 2]] for e in etas], [])
        Dm = np.array(Dm); Ci = np.linalg.inv(C)
        P = np.linalg.solve(Dm.T @ Ci @ Dm, Dm.T @ Ci)
        gc = []
        for d, y in zip(draws, Y):
            a1h, a3h = P @ y
            gc.append(-6 * d["z"] ** 4 * a3h / a1h ** 2)
        r["cubic_GLS"] = summarise(gc)
        # per-draw Lee-Yang LP midpoint (statistical distribution of the LY estimator)
        if len(sys.argv) > 3 and sys.argv[3] == "--per-draw-lp":
            gl = []
            for d in draws:
                tt, RR, DD = [], [], []
                for e in etas:
                    ee = float(e); tau = ee * ee * d["a1"] / 2
                    tt.append(tau); RR.append(d["q"][e] / d["a1"]); DD.append((d["q"][e] - d["s"][e]) / (ee * ee * d["a1"] ** 2))
                ee_ = lb.ly_estimate(tt, RR, DD, sR, sD, ngrid=1500)
                gl.append(3 * d["z"] ** 4 * ee_["U_mid"])
            r["LY_LP_per_draw"] = summarise(gl)
        report["volumes"].append(r)
    json.dump(report, open(path_out, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
