"""Direct finite-source ensemble analysis and comparison with exact h=0 reweighting.

For a run at source h (eta = h L^3) we compute q = <X>/eta and s = Var(X), both raw and with exact
uniform-mode conditioning at that source (RB), with block-jackknife errors.  These are compared with the
same-ensemble h=0 reweighted predictions stored by analyze_ensemble.py (an exact identity: agreement
within errors validates sampler + identity; disagreement would falsify one of them).
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import zm  # noqa: E402


def main():
    out_json, L, m0sq, lam, h, nb = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
    files = sys.argv[7:]
    V = L ** 4
    eta = h * L ** 3
    parts = [zm.load(f) for f in files]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    X = L * d["u"]
    c = zm.conditional(d["S2"], d["S3"], lam=lam, m0sq=m0sq, V=V, L=L, eta=eta, kmax=2)
    r1, r2 = c["mom"][1], c["mom"][2]
    B = lambda a: zm.blocks_mean(a, nb)
    fun = lambda a: np.array([a[0] / eta, a[1] - a[0] ** 2])
    raw, eraw, _ = zm.jackknife(fun, [B(X), B(X * X)])
    rb, erb, _ = zm.jackknife(fun, [B(r1), B(r2)])
    taus = {k: zm.tau_int(v)[0] for k, v in [("X", X), ("r1", r1), ("r2", r2)]}
    res = {"L": L, "m0sq": m0sq, "lambda": lam, "h": h, "eta": eta, "N": int(X.size),
           "q_raw": [float(raw[0]), float(eraw[0])], "s_raw": [float(raw[1]), float(eraw[1])],
           "q_RB": [float(rb[0]), float(erb[0])], "s_RB": [float(rb[1]), float(erb[1])],
           "var_ratio_RB_over_raw": {"q": float((erb[0] / eraw[0]) ** 2), "s": float((erb[1] / eraw[1]) ** 2)},
           "tau_int": {k: float(v) for k, v in taus.items()}}
    json.dump(res, open(out_json, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
