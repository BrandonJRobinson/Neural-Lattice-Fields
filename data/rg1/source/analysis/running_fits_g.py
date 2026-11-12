"""Running fits in g-space (errors on g, not linearised reciprocals). NEW prospective analysis on local data.

g_model = 1/y_model with
  M1   : y = A + B log(L/12)
  MCUT : y = A + B log(L/12) + C (L^-2 - 12^-2)
  MPT  : y = Kconv_L + A + alpha (K_L - Kconv_L)   (scheme-matched exact one-loop lattice kernel at measured z)
Errors: block-jackknife; for L >= 16 the larger of jackknife and 8-batch-means error.
"""
import json, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from running_fits import K  # noqa: E402

b1 = 3 / (16 * np.pi ** 2)
rows = sorted(json.load(open(sys.argv[1])), key=lambda r: (r["L"], r["m0sq"]))
batch = json.load(open("data/batch_check.json"))
ref = [r for r in rows if r["L"] == 12][0]
def err(r):
    e = r["g_RB"][1]
    key = f"{r['L']}_{r['m0sq']}"
    if key in batch: e = max(e, batch[key]["batches"]["8"]["batch_err"])
    return e

def fit(sel, model, fixed=None):
    L = np.array([r["L"] for r in sel], float); g = np.array([r["g_RB"][0] for r in sel]); eg = np.array([err(r) for r in sel])
    z = np.array([r["z_RB"][0] for r in sel])
    KK = np.array([K(Li, zi, ref["g_RB"][0], ref["z_RB"][0])[0] for Li, zi in zip(L, z)])
    Kc = np.array([K(Li, zi, ref["g_RB"][0], ref["z_RB"][0])[1] for Li, zi in zip(L, z)])
    x = np.log(L / 12)
    def ymod(p):
        if model == "M1": return p[0] + p[1] * x
        if model == "MCUT": return p[0] + p[1] * x + p[2] * (L ** -2 - 12.0 ** -2)
        a = p[1] if fixed is None else fixed
        return Kc + p[0] + a * (KK - Kc)
    npar = {"M1": 2, "MCUT": 3, "MPT": 2 if fixed is None else 1}[model]
    p0 = [1 / 5.8, b1, 0.0][:npar] if model != "MPT" else ([1 / 5.8, 1.0] if fixed is None else [1 / 5.8])
    rf = least_squares(lambda p: (1 / ymod(p) - g) / eg, p0)
    J = rf.jac; C = np.linalg.inv(J.T @ J)
    return rf.x, np.sqrt(np.diag(C)), float(np.sum(rf.fun ** 2)), len(L) - npar

out = {}
for name, sel in [("L6-32", rows), ("L6-12", [r for r in rows if r["L"] <= 12]), ("L12-32", [r for r in rows if r["L"] >= 12]),
                  ("L8-32", [r for r in rows if r["L"] >= 8])]:
    res = {"L": [r["L"] for r in sel]}
    p, e, c2, dof = fit(sel, "M1"); res["M1"] = {"B": [p[1], e[1]], "alpha_B": [p[1] / b1, e[1] / b1], "chi2": c2, "dof": dof}
    if len(sel) >= 4:
        p, e, c2, dof = fit(sel, "MCUT"); res["MCUT"] = {"B": [p[1], e[1]], "C": [p[2], e[2]], "chi2": c2, "dof": dof}
    p, e, c2, dof = fit(sel, "MPT"); res["MPT"] = {"alpha": [p[1], e[1]], "chi2": c2, "dof": dof}
    _, _, c1, _ = fit(sel, "MPT", fixed=1.0); _, _, c0, _ = fit(sel, "MPT", fixed=0.0)
    res["MPT"]["dchi2_alpha1"] = c1 - c2; res["MPT"]["dchi2_alpha0"] = c0 - c2
    out[name] = res
    print(name, res["L"])
    for m in ("M1", "MCUT", "MPT"):
        if m in res: print("   ", m, {k: ([round(v, 4) for v in vv] if isinstance(vv, list) else round(vv, 2)) for k, vv in res[m].items()})
json.dump(out, open(sys.argv[2], "w"), indent=1)
