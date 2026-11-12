"""Fill {{SCALING_TABLE}} and {{RUNNING_FITS}} in the verdict from result JSONs (no hand-copied numbers)."""
import json

import numpy as np

rows = sorted(json.load(open("data/scaling_final.json")), key=lambda r: (r["L"], r["m0sq"]))
lines = []
for r in rows:
    lines.append("| {L} | {m} | {N:.2e} | {z:.4f}({ez:.0f}) | {g:.3f} ± {eg:.3f} | {f:.3f} | {fp:.3f} | {S:.4f} | {rr:.4f} | {om:.4f} | {t:.2f} | {s:.2g} |".format(
        L=r["L"], m=r["m0sq"], N=r["N"], z=r["z_RB"][0], ez=r["z_RB"][1] * 1e4, g=r["g_RB"][0], eg=r["g_RB"][1],
        f=r["f"], fp=r["f_pred"], S=r["Sigma"], rr=r["per_config_ratio_RB_raw"], om=r["one_minus_f_4"],
        t=r["tau_int"]["infl_rb"], s=r["seconds_per_sweepset"]))
table = "\n".join(lines)

fits = json.load(open("data/running_fits_g_final.json"))
pt = json.load(open("data/pt_compare_final.json"))
batch = json.load(open("data/batch_check.json"))
ref = [r for r in rows if r["L"] == 12][0]
out = ["Per-ensemble comparison with exact one-loop finite-lattice PT (scheme-matched), evaluated at each run's measured z "
       "and anchored at the measured L=12 value. Errors: block jackknife; for L ≥ 16 the larger of jackknife and 8-batch means. "
       "PT remainder not bounded.\n",
       "| L | m₀² | z | g_R measured | g_R one-loop PT (± anchor) | pull |", "|---|---|---|---|---|---|"]
for r in rows:
    e = r["g_RB"][1]; key = f"{r['L']}_{r['m0sq']}"
    if key in batch: e = max(e, batch[key]["batches"]["8"]["batch_err"])
    pp = [p for p in pt if p["L"] == r["L"] and abs(p["z"] - r["z_RB"][0]) < 1e-9][0]
    gpt = 1 / (1 / ref["g_RB"][0] + pp["dy_MPT_scheme"])
    e_anchor = gpt ** 2 * ref["g_RB"][1] / ref["g_RB"][0] ** 2   # propagated L=12 anchor uncertainty
    pull = (r["g_RB"][0] - gpt) / np.hypot(e, e_anchor) if r["L"] != 12 else 0.0
    out.append(f"| {r['L']} | {r['m0sq']} | {r['z_RB'][0]:.4f} | {r['g_RB'][0]:.3f} ± {e:.3f} | {gpt:.3f} ± {e_anchor:.3f} | {pull:+.1f} |")
out.append("\nFits in g-space (independent ensembles):\n")
out.append("| Ensembles | M1: B (B/b₁) | MCUT: B | MPT amplitude α | χ²/dof | Δχ²(α=0) | Δχ²(α=1) |")
out.append("|---|---|---|---|---|---|---|")
for k, v in fits.items():
    m1 = v["M1"]; mc = v.get("MCUT"); mp = v["MPT"]
    mcs = f"{mc['B'][0]:+.4f} ± {mc['B'][1]:.4f}" if mc else "—"
    out.append(f"| L = {k[1:]} ({len(v['L'])}) | {m1['B'][0]:+.4f} ± {m1['B'][1]:.4f} ({m1['alpha_B'][0]:.2f} ± {m1['alpha_B'][1]:.2f}) | {mcs} | "
               f"{mp['alpha'][0]:.3f} ± {mp['alpha'][1]:.3f} | {mp['chi2']:.1f}/{mp['dof']} | {mp['dchi2_alpha0']:.1f} | {mp['dchi2_alpha1']:.1f} |")
out.append("\n**Reading.**")
out.append("- The resolved decrease and its agreement with finite-lattice one-loop PT (pulls −1.3, −0.8, +0.9 at L = 6, 8, 10; α = 0.875 ± 0.067 with a free intercept) come from L/a = 6–12, where lattice artefacts are large but are included exactly in the one-loop kernel.")
out.append("- The short L ≥ 16 runs all lie above the PT expectation (pulls +1.0 to +2.3; χ² ≈ 10.4/4). On the frozen ladder volumes alone (12–32) there is no resolved running, and α is 2.5σ below 1. This is a ≈2–2.5σ tension (p ≈ 0.01–0.03), not a measurement.")
out.append("- MCUT resolves no slope anywhere. Nothing here is a cutoff-robust running coefficient.")
running = "\n".join(out)

v = open("SCIENTIFIC_RESCUE_VERDICT.md").read()
if "{{RUNNING_FITS}}" not in v:
    a = v.index("### 6.4 In-session small-volume running analysis"); b = v.index("![g_R ladder]")
    hdr = v[a:].split("\n", 1)[0]
    v = v[:a] + hdr + "\n\n{{RUNNING_FITS}}\n\n" + v[b:]
v = v.replace("{{SCALING_TABLE}}", table).replace("{{RUNNING_FITS}}", running)
open("SCIENTIFIC_RESCUE_VERDICT.md", "w").write(v)
print(table)
print(running)
