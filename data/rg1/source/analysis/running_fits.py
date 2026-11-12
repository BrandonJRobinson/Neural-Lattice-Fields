"""Running fits on the independent local fixed-z ladder (NEW prospective analysis; not the frozen G2 set).

Inputs: scaling table rows (g_RB, z_RB per L).  Volumes are independent ensembles -> diagonal covariance.
Models (y = 1/g_P):
  M1    : y = A + B log(L/12)
  MCUT  : y = A + B log(L/12) + C (L^-2 - 12^-2)
  MPTa  : y = A + alpha * K_L, K_L = scheme-matched exact one-loop finite-lattice prediction for y(L)-y(12)
          evaluated at each run's measured z (anchor-independent shape; alpha=1 is one-loop PT, alpha=0 none)
Reported for several minimum-L cuts.  Statistical errors only; z-mistuning enters K_L but not M1/MCUT.
"""
import json
import sys

import numpy as np

b1 = 3 / (16 * np.pi ** 2)


def I2(L, m):
    p = 2 * np.pi * np.arange(L) / L; ph2 = 4 * np.sin(p / 2) ** 2
    t = ph2[:, None, None, None] + ph2[None, :, None, None] + ph2[None, None, :, None] + ph2[None, None, None, :]
    return np.sum(1 / (t + m * m) ** 2) / L ** 4


def K(L, z, gref, zref, Lref=12):
    mu = 2 * np.sinh(z / (2 * L)); q = ((z / L) / mu) ** 4
    muR = 2 * np.sinh(zref / (2 * Lref)); qR = ((zref / Lref) / muR) ** 4
    # y_P(L) = [qR y_P(ref) + 1.5(I2(L,mu)-I2(ref,muR))]/q ; linearise the alpha-scaled loop term around y_P(ref)
    yref = 1 / gref
    return (qR * yref + 1.5 * (I2(L, mu) - I2(Lref, muR))) / q - yref, (qR / q - 1) * yref


def wls(X, y, s):
    W = 1 / s ** 2
    F = X.T @ (W[:, None] * X)
    C = np.linalg.inv(F)
    beta = C @ (X.T @ (W * y))
    chi2 = float(np.sum(W * (y - X @ beta) ** 2))
    return beta, np.sqrt(np.diag(C)), chi2


def main():
    rows = json.load(open(sys.argv[1]))
    rows = sorted([r for r in rows if np.isfinite(r["g_RB"][1])], key=lambda r: r["L"])
    ref = [r for r in rows if r["L"] == 12][0]
    out = {}
    for Lmin in [6, 8, 10]:
        sel = [r for r in rows if r["L"] >= Lmin]
        L = np.array([r["L"] for r in sel], float)
        g = np.array([r["g_RB"][0] for r in sel]); eg = np.array([r["g_RB"][1] for r in sel])
        z = np.array([r["z_RB"][0] for r in sel])
        y = 1 / g; sy = eg / g ** 2
        x = np.log(L / 12)
        res = {"L": L.tolist()}
        b, e, c2 = wls(np.c_[np.ones_like(x), x], y, sy)
        res["M1"] = {"B": [b[1], e[1]], "alpha=B/b1": [b[1] / b1, e[1] / b1], "chi2": c2, "dof": len(L) - 2}
        if len(L) >= 4:
            b, e, c2 = wls(np.c_[np.ones_like(x), x, L ** -2 - 12.0 ** -2], y, sy)
            res["MCUT"] = {"B": [b[1], e[1]], "alpha=B/b1": [b[1] / b1, e[1] / b1], "C": [b[2], e[2]], "chi2": c2, "dof": len(L) - 3}
        KK = np.array([K(Li, zi, ref["g_RB"][0], ref["z_RB"][0])[0] for Li, zi in zip(L, z)])
        Kconv = np.array([K(Li, zi, ref["g_RB"][0], ref["z_RB"][0])[1] for Li, zi in zip(L, z)])
        # alpha multiplies the loop part only; the pure scheme-conversion piece is kinematic
        b, e, c2 = wls(np.c_[np.ones_like(x), KK - Kconv], y - Kconv, sy)
        r1 = (y - Kconv) - (KK - Kconv)
        r0 = (y - Kconv)
        res["MPT_alpha"] = {"alpha": [b[1], e[1]], "chi2": c2, "dof": len(L) - 2,
                            "chi2_alpha1_refitA": float(np.sum((r1 - np.average(r1, weights=1 / sy ** 2)) ** 2 / sy ** 2)),
                            "chi2_alpha0_refitA": float(np.sum((r0 - np.average(r0, weights=1 / sy ** 2)) ** 2 / sy ** 2))}
        out[f"Lmin={Lmin}"] = res
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
