"""Rigorous Lee-Yang (Newman) bracket on the zero-source coupling from finite-source data.

Theorem chain (finite periodic lattice, S = sum[1/2 (dphi)^2 + 1/2 m0^2 phi^2 + lambda phi^4],
lambda > 0, any real m0^2, uniform source):
  * Single-site measure exp(-a phi^2 - lambda phi^4) is in the Griffiths-Simon / Lee-Yang class
    (Simon-Griffiths 1973; Newman 1974), nearest-neighbour couplings are ferromagnetic.
  * Lee-Yang: Z(eta) = <exp(eta X)>_0 vanishes only for Re eta = 0.
  * Z is entire of order 4/3 < 2 and even, hence (Hadamard) Z(eta) = prod_j (1 + eta^2/alpha_j^2).
Consequently, with a1 = <X^2>_0, w_j = 2/(a1 alpha_j^2) (sum w_j = 1, counted with multiplicity),
tau = eta^2 a1 / 2:
     R(tau)  := q(eta)/a1 = sum_j w_j / (1 + tau w_j)
     D(tau)  := -dR/dtau = (q - s)/(2 tau a1) = sum_j w_j^2/(1 + tau w_j)^2
     U4 = 1 - <X^4>/(3<X^2>^2) = sum_j w_j^2,     g_R = 3 z^4 U4.
Every atom of rho = sum_j w_j delta_{w_j} at location w carries mass n*w >= w (integer
multiplicity), hence max_j w_j <= sqrt(U4).

Given data (exact or interval-valued) for R and D at a set of tau values, the set of U4 values
consistent with SOME such measure is bracketed by a linear program over measures on (0, W_cap].
We iterate W_cap = sqrt(U4_max) to impose the atom constraint (valid because the true measure
has support in [0, sqrt(U4_true)] and U4_true <= U4_max).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def lp_bracket(taus, R, D, *, R_tol=None, D_tol=None, ngrid=4000, wcap=1.0, use_D=True, iterate=True):
    taus = np.asarray(taus, float)
    R = np.asarray(R, float); D = np.asarray(D, float)
    R_tol = np.zeros_like(R) if R_tol is None else np.asarray(R_tol, float)
    D_tol = np.zeros_like(D) if D_tol is None else np.asarray(D_tol, float)
    hist = []
    for it in range(30):
        # geometric grid concentrates resolution at small w, plus linear part
        w = np.unique(np.concatenate([np.geomspace(1e-6, wcap, ngrid // 2), np.linspace(1e-6, wcap, ngrid // 2)]))
        A_eq = [np.ones_like(w)]; b_eq = [1.0]
        A_ub, b_ub = [], []
        for i, t in enumerate(taus):
            rowR = 1.0 / (1.0 + t * w)
            if R_tol[i] == 0:
                A_eq.append(rowR); b_eq.append(R[i])
            else:
                A_ub.append(rowR); b_ub.append(R[i] + R_tol[i])
                A_ub.append(-rowR); b_ub.append(-(R[i] - R_tol[i]))
            if use_D:
                rowD = w / (1.0 + t * w) ** 2
                if D_tol[i] == 0:
                    A_eq.append(rowD); b_eq.append(D[i])
                else:
                    A_ub.append(rowD); b_ub.append(D[i] + D_tol[i])
                    A_ub.append(-rowD); b_ub.append(-(D[i] - D_tol[i]))
        kw = dict(A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=(0, None), method="highs")
        if A_ub:
            kw.update(A_ub=np.array(A_ub), b_ub=np.array(b_ub))
        lo = linprog(w, **kw)
        hi = linprog(-w, **kw)
        if lo.status != 0 or hi.status != 0:
            return {"feasible": False, "status": [lo.status, hi.status], "message": [lo.message, hi.message],
                    "wcap": wcap, "history": hist}
        umin, umax = lo.fun, -hi.fun
        hist.append((wcap, umin, umax))
        if not iterate:
            break
        new_cap = min(1.0, np.sqrt(umax))
        if new_cap >= wcap * (1 - 1e-6):
            break
        wcap = new_cap
    return {"feasible": True, "U4_min": umin, "U4_max": umax, "wcap": wcap, "history": hist}


def analytic_single_radius(Gq, z, tau):
    """Closed-form bracket from one secant: Gq <= g <= Gq (1 + tau sqrt(U4))."""
    Uq = Gq / (3 * z ** 4)
    x = 0.5 * (tau * Uq + np.sqrt((tau * Uq) ** 2 + 4 * Uq))
    return Gq, 3 * z ** 4 * x * x


def lower_bound_q2_over_s(Gq, Gs):
    """g >= Gq^2/Gs  (Cauchy-Schwarz on the second-moment zero measure at one radius)."""
    return Gq * Gq / Gs


def ly_estimate(taus, R, D, sR, sD, *, ngrid=3000, use_D=True, slack_floor=1e-9):
    """Noise-robust LY extraction.

    Step 1: minimal normalised violation t* = min max_i |model_i - data_i|/sigma_i over LY measures
            (support (0, wcap], wcap iterated to sqrt(U4_max)).  t* <= ~2 means data are LY-consistent.
    Step 2: bracket [U_min, U_max] over measures with violation <= max(t*, slack_floor) (+1e-6 rel).
    Returns midpoint estimate, bracket, and t*.
    """
    taus = np.asarray(taus, float); R = np.asarray(R, float); D = np.asarray(D, float)
    sR = np.asarray(sR, float); sD = np.asarray(sD, float)
    wcap = 1.0
    for _ in range(20):
        w = np.unique(np.concatenate([np.geomspace(1e-6, wcap, ngrid // 2), np.linspace(1e-6, wcap, ngrid // 2)]))
        nw = w.size
        # variables: p (nw), t (1)
        rows, rhs = [], []
        for i, tau in enumerate(taus):
            fr = 1.0 / (1.0 + tau * w)
            rows.append(np.concatenate([fr, [-sR[i]]])); rhs.append(R[i])
            rows.append(np.concatenate([-fr, [-sR[i]]])); rhs.append(-R[i])
            if use_D:
                fd = w / (1.0 + tau * w) ** 2
                rows.append(np.concatenate([fd, [-sD[i]]])); rhs.append(D[i])
                rows.append(np.concatenate([-fd, [-sD[i]]])); rhs.append(-D[i])
        A_ub = np.array(rows); b_ub = np.array(rhs)
        A_eq = np.concatenate([np.ones(nw), [0.0]])[None, :]
        c = np.zeros(nw + 1); c[-1] = 1.0
        r0 = None
        for meth in ("highs", "highs-ipm", "highs-ds"):
            r0 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * nw + [(0, None)], method=meth)
            if r0.status == 0:
                break
        if r0.status != 0:
            return {"U_mid": float("nan"), "U_min": float("nan"), "U_max": float("nan"), "t_star": float("nan"), "wcap": wcap, "failed": r0.message}
        tstar = r0.fun
        A2 = A_ub[:, :nw]
        tt = max(tstar * (1 + 1e-4), slack_floor)
        for _k in range(12):
            b2 = b_ub + np.abs(A_ub[:, -1]) * tt
            lo = linprog(w, A_ub=A2, b_ub=b2, A_eq=np.ones((1, nw)), b_eq=[1.0], bounds=(0, None), method="highs")
            hi = linprog(-w, A_ub=A2, b_ub=b2, A_eq=np.ones((1, nw)), b_eq=[1.0], bounds=(0, None), method="highs")
            if lo.status == 0 and hi.status == 0:
                break
            tt = tt * 1.01 + 1e-9  # numerical feasibility margin only
        umin, umax = lo.fun, -hi.fun
        new = min(1.0, np.sqrt(umax))
        if new >= wcap * (1 - 1e-6):
            break
        wcap = new
    return {"U_mid": 0.5 * (umin + umax), "U_min": umin, "U_max": umax, "t_star": tstar, "wcap": wcap}
