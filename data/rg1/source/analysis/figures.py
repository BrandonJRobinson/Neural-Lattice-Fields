"""Figures for the verdict (static PNG, light surface). Palette: reference categorical slots 1-3 + neutral ink."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "legend.frameon": False, "lines.linewidth": 2})

rows = sorted(json.load(open("data/scaling_final.json")), key=lambda r: r["L"])
ptl = json.load(open("data/pt_compare_final.json"))

# Figure 1: g_R(L) independent ladder vs frozen and October values, with scheme-matched one-loop PT
fig, ax = plt.subplots(figsize=(7.2, 4.4))
L = np.array([r["L"] for r in rows]); g = np.array([r["g_RB"][0] for r in rows]); eg = np.array([r["g_RB"][1] for r in rows])
ax.errorbar(L, g, eg, fmt="o", ms=7, color=BLUE, capsize=3, label="independent exact sampler, zero source (this review)")
ref = [r for r in rows if r["L"] == 12][0]
gpt = np.array([1 / (1 / ref["g_RB"][0] + [p for p in ptl if p["L"] == r["L"] and abs(p["z"] - r["z_RB"][0]) < 1e-9][0]["dy_MPT_scheme"]) for r in rows])
batch = json.load(open("data/batch_check.json"))
eg = np.array([max(r["g_RB"][1], batch.get(f"{r['L']}_{r['m0sq']}", {"batches": {"8": {"batch_err": 0}}})["batches"]["8"]["batch_err"]) for r in rows])
ax.errorbar(L, g, eg, fmt="none", ecolor=BLUE, capsize=3)
ax.plot(L, gpt, "_", color=INK, ms=16, mew=2, label="scheme-matched 1-loop lattice PT at each run's z (anchored at L=12)")
fz = {12: (4.5471, 4.0622, 5.0435), 16: (3.8712, 2.5137, 5.1982), 24: (4.8036, 2.0226, 7.6033), 32: (3.9373, 0.1150, 7.2837)}
fl = np.array(list(fz)); fv = np.array([fz[k][0] for k in fz])
ax.errorbar(fl + 0.6, fv, [fv - np.array([fz[k][1] for k in fz]), np.array([fz[k][2] for k in fz]) - fv], fmt="s", ms=6,
            color=ORANGE, capsize=3, label="frozen G2 common-window cubic fit (statistical 95%)")
ax.errorbar([24 - 0.6, 32 - 0.6], [14.5237, 19.4469], [[14.5237 - 2.71, 19.4469 - 10.35], [24.76 - 14.5237, 29.98 - 19.4469]],
            fmt="^", ms=6, color=AQUA, capsize=3, label="October h=0 direct cumulant (reported 95%)")
ax.set_xlabel("L/a  (fixed z ≈ 3, λ = 0.4)"); ax.set_ylabel("finite-volume coupling g_R")
ax.set_xscale("log"); ax.set_xticks([6, 8, 10, 12, 16, 24, 32]); ax.set_xticklabels([6, 8, 10, 12, 16, 24, 32])
ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_ylim(0, 32); ax.legend(fontsize=8, loc="upper left")
ax.set_title("g_R(L): independent determination vs frozen and October values", color=INK, fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig1_gR_ladder.png", dpi=160)

# Figure 2: L=12 Lee-Yang structure of the frozen source window
r12 = json.load(open("data/L12_h0_analysis.json"))
J = r12["joint"]; etas = [0.75, 1.5, 2.25, 3.0, 3.75, 4.5, 5.25, 6.0, 6.75, 7.5, 9.0]
gq = np.array([J[f"Gq_{e}"][0] for e in etas]); gs = np.array([J[f"Gs_{e}"][0] for e in etas])
g0, e0 = J["g_RB_tau0"]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.axhspan(g0 - e0, g0 + e0, color=GRID, alpha=0.9, lw=0)
ax.axhline(g0, color=INK, lw=1.2, label=f"zero-source g_R(12) = {g0:.3f} ± {e0:.3f}")
ax.plot(etas, gq, "o-", color=BLUE, ms=6, label="secant G_q(η)  (theorem: < g_R)")
ax.plot(etas, gs, "s-", color=ORANGE, ms=6, label="tangent G_s(η)  (theorem: < G_q)")
ax.plot(etas, gq ** 2 / gs, "D-", color=AQUA, ms=6, label="one-atom Lee–Yang G_q²/G_s  (theorem: ≤ g_R)")
ax.axvspan(4.5, 6.0, color=ORANGE, alpha=0.08, lw=0)
ax.errorbar([5.25], [4.8178], [0.0338], fmt="x", color=ORANGE, ms=9, mew=2, label="frozen cubic window functional (population, equal-N GLS)")
ax.set_xlabel("source η = hL³   (shaded: frozen window |η| = 4.5, 6)"); ax.set_ylabel("coupling estimate")
ax.set_title("L = 12: the frozen cubic window is biased low by a theorem-signed 17%", color=INK, fontsize=11)
ax.legend(fontsize=8, loc="lower left"); fig.tight_layout(); fig.savefig("figures/fig2_L12_LeeYang.png", dpi=160)

# Figure 3: pinning law
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6))
f = np.array([r["f"] for r in rows]); fp = np.array([r["f_pred"] for r in rows])
ax = axs[0]; ax.plot(L, f, "o", color=BLUE, ms=7, label="measured f = E[Var(X|ψ)]/Var X")
ax.plot(L, fp, "-", color=INK2, lw=1.5, label="pinning law m_χ²/μ_c²")
ax.set_xscale("log"); ax.set_xticks([6, 8, 12, 16, 24, 32]); ax.set_xticklabels([6, 8, 12, 16, 24, 32])
ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlabel("L/a"); ax.set_ylabel("conditional-variance fraction f"); ax.legend(fontsize=8)
ax = axs[1]; rr = np.array([r["per_config_ratio_RB_raw"] for r in rows]); om = np.array([r["one_minus_f_4"] for r in rows])
ax.plot(L, rr, "o", color=BLUE, ms=7, label="measured RB/raw variance, κ₄ influence")
ax.plot(L, om, "-", color=INK2, lw=1.5, label="(1−f)⁴")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([6, 8, 12, 16, 24, 32]); ax.set_xticklabels([6, 8, 12, 16, 24, 32])
ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlabel("L/a"); ax.set_ylabel("per-configuration variance ratio"); ax.legend(fontsize=8)
fig.suptitle("Exact uniform-mode conditioning loses power as the zero mode is pinned to Σψ³", color=INK, fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig3_pinning.png", dpi=160)
print("ok")
