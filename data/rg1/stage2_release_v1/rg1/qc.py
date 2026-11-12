"""Predeclared per-chain QC, run only on the captured persistent copy.

PASS requires all of:
  * all 4*n_meas values finite;
  * heat-bath acceptance in [0.90, 1.00] and over-relaxation acceptance in [0.90, 1.00];
  * split-chain stationarity: for X^2, S2 and G(p_min), |mean(first half) - mean(second half)| <= 4 combined
    batch-means SE (20 batches per half);
  * block rule: the chain supplies >= 8 blocks of >= 50 tau_int (Madras-Sokal, frozen zm.tau_int) for
    X^2, the RB-free kappa_4 influence proxy, S2, S3 and G(p_min).
Zero-mode Gibbs skips (exact by construction) are recorded, not gated.
"""
from __future__ import annotations

import numpy as np

from .frozen import zm


def _bm_se(x, nb=20):
    n = (len(x) // nb) * nb
    b = x[:n].reshape(nb, -1).mean(axis=1)
    return b.std(ddof=1) / np.sqrt(nb)


def chain_qc(path, task: dict, meta: dict) -> dict:
    a = np.fromfile(path, dtype=np.float64).reshape(-1, 4)
    L = int(task["args"][0])
    u, S2, S3, Gp = a.T
    X = L * u
    out = {"n": int(len(a)), "reasons": []}
    if not np.all(np.isfinite(a)):
        out["reasons"].append("nonfinite values")
        out["status"] = "FAIL"; return out
    for k in ("acc_hb", "acc_or"):
        v = float(meta[k]); out[k] = v
        if not 0.90 <= v <= 1.0:
            out["reasons"].append(f"{k}={v} outside [0.90, 1.00]")
    out["zm_skipped_meas"] = int(meta.get("zm_skipped_meas", 0))
    X2 = X * X
    series = {"X2": X2, "infl_raw": X ** 4 - 6 * X2.mean() * X2, "S2": S2, "S3": S3, "Gp": Gp}
    taus = {k: float(zm.tau_int(v)[0]) for k, v in series.items()}
    out["tau_int"] = taus
    blk = len(a) / 8
    out["block_length_over_50tau"] = blk / (50 * max(taus.values()))
    if out["block_length_over_50tau"] < 1:
        out["reasons"].append(f"8 blocks of {blk:.0f} < 50 * max tau_int {max(taus.values()):.2f}")
    half = len(a) // 2
    split = {}
    for k in ("X2", "S2", "Gp"):
        x = series[k]
        d = x[:half].mean() - x[half:2 * half].mean()
        se = np.hypot(_bm_se(x[:half]), _bm_se(x[half:2 * half]))
        split[k] = float(d / se) if se > 0 else 0.0
        if abs(split[k]) > 4:
            out["reasons"].append(f"split-chain {k} |z|={abs(split[k]):.2f} > 4")
    out["split_chain_z"] = split
    out["means"] = {"X2": float(X2.mean()), "S2": float(S2.mean()), "Gp": float(Gp.mean())}
    out["status"] = "PASS" if not out["reasons"] else "FAIL"
    return out
