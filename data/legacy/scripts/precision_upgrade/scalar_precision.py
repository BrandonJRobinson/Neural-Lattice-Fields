"""Checkpointed 2d/3d scalar precision campaigns and derived artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOWER = ROOT / "experiments_lowerd"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LOWER))

from common import (RESULTS, conservative_seed_combine, runtime_metadata, sha256,
                    write_csv, write_json)  # noqa: E402
from connected_xi import bootstrap_estimators  # noqa: E402
from hmc import LatentHMC, phi4_interaction  # noqa: E402
from latent_lattice import WhiteningOp  # noqa: E402
from observables import FIELD_GHAT_COLS, field_summary_with_ghat, integrated_autocorr_time  # noqa: E402


CONFIG_PATH = HERE / "configs" / "scalar_precision.json"
LEGACY_2D = LOWER / "outputs" / "phi4_ising_2d_critical_refine_summary.json"
SOURCE_FILES = [Path(__file__), HERE / "connected_xi.py", HERE / "common.py",
                CONFIG_PATH, LOWER / "hmc.py", LOWER / "latent_lattice.py",
                LOWER / "observables.py"]


def _finite_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _binder(arrays: dict[str, np.ndarray]) -> float:
    m2 = float(np.mean(arrays["M2"]))
    m4 = float(np.mean(arrays["M4"]))
    return float(1.0 - m4 / (3.0 * m2 * m2)) if m2 > 0 else float("nan")


def _block_bootstrap_scalar(arrays, estimator, seed, n_boot=500, n_blocks=20):
    from common import fixed_block_bootstrap
    return fixed_block_bootstrap(arrays, estimator, n_boot,
                                 np.random.default_rng(seed), n_blocks)


def _chain_tag(dim: int, L: int, m0sq: float, seed: int) -> str:
    mass = f"{m0sq:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
    return f"d{dim}_L{L}_m{mass}_seed{seed}"


def run_chain_task(task: dict) -> dict:
    dim = int(task["dim"])
    L = int(task["L"])
    m0sq = float(task["m0sq"])
    seed = int(task["seed"])
    h = task["hmc"]
    outdir = Path(task["outdir"])
    chains = outdir / "chains"
    chains.mkdir(parents=True, exist_ok=True)
    tag = _chain_tag(dim, L, m0sq, seed)
    json_path = chains / f"{tag}.json"
    npz_path = chains / f"{tag}.npz"
    if json_path.exists() and npz_path.exists() and not task.get("overwrite", False):
        with json_path.open() as handle:
            return json.load(handle)

    m_ref_sq = max(0.05, abs(m0sq))
    W = WhiteningOp(L, m_ref_sq, dim)
    action, grad = phi4_interaction(m0sq, m_ref_sq, float(task["lambda"]))
    sampler = LatentHMC(W, action, grad, step_size=float(h["step_size_init"]),
                        n_leapfrog=int(h["n_leapfrog"]))
    start = time.perf_counter()
    result = sampler.run(
        field_summary_with_ghat,
        n_therm=int(h["n_therm"]), n_measure=int(h["n_measure"]),
        measure_every=int(h.get("measure_every", 1)), seed=seed,
        adapt_step=True, target_acc=float(h.get("target_acc", 0.75)),
    )
    samples = result["samples"]
    arrays = {name: samples[:, i] for i, name in enumerate(FIELD_GHAT_COLS)}
    arrays["dim"] = np.full(samples.shape[0], dim, dtype=np.int16)
    np.savez_compressed(npz_path, **arrays)

    n_boot = int(task["n_boot"])
    n_blocks = int(task["n_blocks"])
    xi = bootstrap_estimators(arrays, L, seed + 22000, n_boot, n_blocks)
    binder = xi["U4_bootstrap"]
    chi0_b = xi["chi0_bootstrap"]
    chia_b = xi["chi_abs_bootstrap"]
    tau_m = integrated_autocorr_time(arrays["M"])
    tau_abs = integrated_autocorr_time(arrays["absM"])
    tau_m2 = integrated_autocorr_time(arrays["M2"])
    summary = {
        "tag": tag, "source": "new_precision_chain", "dim": dim, "L": L,
        "m0sq": m0sq, "lambda": float(task["lambda"]), "m_ref_sq": m_ref_sq,
        "seed": seed, "hmc": dict(h), "acceptance": float(result["acceptance"]),
        "step_size_final": float(result["step_size_final"]),
        "n_leapfrog": int(result["n_leapfrog"]), "dH_stats": result["dH_stats"],
        "nonfinite_rejects": int(result["nonfinite_rejects"]),
        "nonfinite_rejects_therm": int(result["nonfinite_rejects_therm"]),
        "nonfinite_rejects_prod": int(result["nonfinite_rejects_prod"]),
        "nonfinite_measurements": int(result["nonfinite_measurements"]),
        "n_measure_recorded": int(result["n_measure_recorded"]),
        "max_abs_phi": float(result["max_abs_phi"]),
        "max_abs_theta": float(result["max_abs_theta"]),
        "action_stats": result["action_stats"],
        "U4": binder["value"], "U4_se": binder["se"], "U4_bootstrap": binder,
        "chi0": chi0_b["value"], "chi0_se": chi0_b["se"],
        "chi_abs": chia_b["value"], "chi_abs_se": chia_b["se"],
        "absM": float(np.mean(arrays["absM"])),
        "M": float(np.mean(arrays["M"])),
        "M2": float(np.mean(arrays["M2"])),
        "M4": float(np.mean(arrays["M4"])),
        "xi": xi,
        "tau_M": tau_m, "tau_absM": tau_abs, "tau_M2": tau_m2,
        "npz_path": str(npz_path.relative_to(ROOT)), "npz_sha256": sha256(npz_path),
        "runtime_s": float(result["time_s"]),
        "analysis_runtime_s": float(time.perf_counter() - start - result["time_s"]),
        "runtime_metadata": runtime_metadata(SOURCE_FILES),
    }
    summary["diagnostic_pass"] = bool(
        0.55 <= summary["acceptance"] <= 0.90
        and summary["nonfinite_measurements"] == 0
        and summary["nonfinite_rejects"] == 0
        and np.isfinite(summary["U4"])
        and summary["tau_absM"]["N_eff"] >= 50
    )
    write_json(json_path, summary)
    return summary


def load_legacy_2d(cfg: dict) -> list[dict]:
    with LEGACY_2D.open() as handle:
        legacy = json.load(handle)
    masses = {round(float(x), 10) for x in cfg["2d"]["m0sq_grid"]}
    Ls = set(cfg["2d"]["L_list"])
    seeds = set(cfg["2d"]["reuse_seeds"])
    points = []
    for p in legacy["per_seed_points"]:
        if p["L"] not in Ls or p["seed"] not in seeds or round(float(p["m0sq"]), 10) not in masses:
            continue
        points.append({
            "tag": _chain_tag(2, p["L"], p["m0sq"], p["seed"]),
            "source": "legacy_2d_critical_refine", "source_path": str(LEGACY_2D.relative_to(ROOT)),
            "source_sha256": sha256(LEGACY_2D), "dim": 2, "L": int(p["L"]),
            "m0sq": float(p["m0sq"]), "lambda": 1.0, "m_ref_sq": float(p["m_ref_sq"]),
            "seed": int(p["seed"]), "acceptance": float(p["acceptance"]),
            "step_size_final": float(p["step_size_final"]), "n_leapfrog": int(p["n_leapfrog"]),
            "hmc": {"n_therm": int(p["n_therm"]), "n_measure": int(p["n_measure"]),
                    "measure_every": 1, "n_leapfrog": int(p["n_leapfrog"]),
                    "step_size_init": 0.05, "target_acc": 0.75},
            "dH_stats": p["dH_stats"], "nonfinite_rejects": None,
            "nonfinite_measurements": None,
            "nonfinite_note": "not recorded in legacy 2d per-seed summary",
            "n_measure_recorded": int(p["n_measure"]),
            "U4": float(p["U4"]["value"]), "U4_se": float(p["U4"]["err"]),
            "chi0": float(p["chi0"]["value"]), "chi0_se": float(p["chi0"]["err"]),
            "chi_abs": float(p["chi_abs"]["value"]), "chi_abs_se": float(p["chi_abs"]["err"]),
            "absM": float(p["absM"]["value"]), "tau_M": p["tau_M"],
            "tau_absM": p["tau_absM"], "tau_M2": p["tau_M2"],
            "runtime_s": float(p["time_s"]),
            "diagnostic_pass": bool(p["sanity_passed"]
                                    and p["tau_absM"].get("N_eff", 0) >= 50
                                    and 0.55 <= p["acceptance"] <= 0.90),
            "xi": None,
        })
    return points


def load_new_points(outdir: Path, dim: int) -> list[dict]:
    rows = []
    for path in sorted((outdir / "chains").glob(f"d{dim}_*.json")):
        try:
            with path.open() as handle:
                row = json.load(handle)
                row["diagnostic_pass"] = bool(
                    0.55 <= row.get("acceptance", float("nan")) <= 0.90
                    and row.get("nonfinite_measurements", 1) == 0
                    and row.get("nonfinite_rejects", 1) == 0
                    and row.get("tau_absM", {}).get("N_eff", 0) >= 50
                    and np.isfinite(row.get("U4", float("nan")))
                )
                rows.append(row)
        except Exception:
            continue
    return rows


def point_rows(points: list[dict]) -> list[dict]:
    grouped = {}
    for p in points:
        grouped.setdefault((int(p["L"]), float(p["m0sq"])), []).append(p)
    rows = []
    for (L, mass), group in sorted(grouped.items()):
        u4 = conservative_seed_combine(group, "U4", "U4_se")
        chi0 = conservative_seed_combine(group, "chi0", "chi0_se")
        chia = conservative_seed_combine(group, "chi_abs", "chi_abs_se")
        abs_m = conservative_seed_combine(group, "absM")
        nf_rej = [p.get("nonfinite_rejects") for p in group if p.get("nonfinite_rejects") is not None]
        nf_meas = [p.get("nonfinite_measurements") for p in group if p.get("nonfinite_measurements") is not None]
        rows.append({
            "L": L, "m0sq": mass, "n_seeds": len(group),
            "seeds": [int(p["seed"]) for p in group],
            "U4": u4["mean"], "U4_se": u4["se"],
            "chi0": chi0["mean"], "chi0_se": chi0["se"],
            "chi_abs": chia["mean"], "chi_abs_se": chia["se"],
            "absM": abs_m["mean"], "absM_se": abs_m["se"],
            "acceptance_mean": float(np.mean([p["acceptance"] for p in group])),
            "acceptance_min": float(np.min([p["acceptance"] for p in group])),
            "acceptance_max": float(np.max([p["acceptance"] for p in group])),
            "nonfinite_rejects": int(sum(nf_rej)) if nf_rej else None,
            "nonfinite_measurements": int(sum(nf_meas)) if nf_meas else None,
            "nonfinite_coverage_seeds": len(nf_meas),
            "tau_absM_mean": float(np.nanmean([p["tau_absM"]["tau_int"] for p in group])),
            "effective_sample_size_sum": float(np.nansum([p["tau_absM"]["N_eff"] for p in group])),
            "runtime_s": float(sum(p.get("runtime_s", 0.0) for p in group)),
            "all_diagnostics_pass": all(p.get("diagnostic_pass", False) for p in group),
            "failed_diagnostic_seeds": int(sum(not p.get("diagnostic_pass", False) for p in group)),
            "sources": sorted(set(p["source"] for p in group)),
        })
    return rows


def xi_rows(points: list[dict]) -> list[dict]:
    groups = {}
    for p in points:
        if p.get("xi"):
            groups.setdefault((p["dim"], p["L"], float(p["m0sq"])), []).append(p)
    rows = []
    for (dim, L, mass), group in sorted(groups.items()):
        def values(key):
            return [{"v": p["xi"][key], "e": p["xi"][key + "_bootstrap"]["se"]} for p in group]
        raw = conservative_seed_combine(values("xi_raw_over_L"), "v", "e")
        conn = conservative_seed_combine(values("xi_connected_over_L"), "v", "e")
        sec = conservative_seed_combine(values("xi_sector_abs_over_L"), "v", "e")
        invalid_raw = sum(p["xi"]["xi_raw_over_L_bootstrap"]["n_invalid"] for p in group)
        invalid_conn = sum(p["xi"]["xi_connected_over_L_bootstrap"]["n_invalid"] for p in group)
        rows.append({
            "dim": dim, "L": L, "m0sq": mass, "n_seeds": len(group),
            "seeds": [p["seed"] for p in group],
            "xi_raw_over_L": raw["mean"], "xi_raw_over_L_se": raw["se"],
            "xi_connected_over_L": conn["mean"], "xi_connected_over_L_se": conn["se"],
            "xi_sector_abs_over_L": sec["mean"], "xi_sector_abs_over_L_se": sec["se"],
            "connected_policy": "sector_abs" if np.mean([p["U4"] for p in group]) > 0.30 else "connected_M",
            "invalid_bootstrap_raw": invalid_raw, "invalid_bootstrap_connected": invalid_conn,
            "acceptance_mean": float(np.mean([p["acceptance"] for p in group])),
            "nonfinite_rejects": int(sum(p["nonfinite_rejects"] for p in group)),
            "nonfinite_measurements": int(sum(p["nonfinite_measurements"] for p in group)),
        })
    return rows


def all_crossings(masses: list[float], diffs: list[float]):
    roots = []
    for m1, m2, d1, d2 in zip(masses, masses[1:], diffs, diffs[1:]):
        if not all(np.isfinite([d1, d2])) or d1 == d2:
            continue
        if d1 == 0 or d1 * d2 < 0:
            root = m1 if d1 == 0 else m1 - d1 * (m2 - m1) / (d2 - d1)
            roots.append((float(root), (float(m1), float(m2))))
    return roots


def crossing_rows(points: list[dict], pairs: list[tuple[int, int]], masses: list[float],
                  n_boot: int, seed: int) -> list[dict]:
    lookup = {(p["L"], float(p["m0sq"]), p["seed"]): p["U4"] for p in points}
    rng = np.random.default_rng(seed)
    out = []
    for La, Lb in pairs:
        common_seeds = sorted(set(p["seed"] for p in points if p["L"] == La)
                              & set(p["seed"] for p in points if p["L"] == Lb))
        diffs = []
        for mass in masses:
            av = [lookup[(La, mass, s)] for s in common_seeds if (La, mass, s) in lookup]
            bv = [lookup[(Lb, mass, s)] for s in common_seeds if (Lb, mass, s) in lookup]
            diffs.append(float(np.mean(bv) - np.mean(av)) if av and bv else float("nan"))
        roots = all_crossings(masses, diffs)
        for branch, (root, bracket) in enumerate(roots):
            boots = []
            for _ in range(n_boot):
                draw = rng.choice(common_seeds, size=len(common_seeds), replace=True)
                bd = []
                for mass in masses:
                    av = [lookup[(La, mass, int(s))] for s in draw if (La, mass, int(s)) in lookup]
                    bv = [lookup[(Lb, mass, int(s))] for s in draw if (Lb, mass, int(s)) in lookup]
                    bd.append(float(np.mean(bv) - np.mean(av)) if av and bv else float("nan"))
                candidate = [x[0] for x in all_crossings(masses, bd) if x[1] == bracket]
                if candidate:
                    boots.append(candidate[0])
            arr = np.asarray(boots)
            out.append({
                "L_low": La, "L_high": Lb, "branch": branch, "m_cross": root,
                "bracket_low": bracket[0], "bracket_high": bracket[1],
                "bootstrap_se": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
                "ci16": float(np.percentile(arr, 16)) if arr.size else float("nan"),
                "ci84": float(np.percentile(arr, 84)) if arr.size else float("nan"),
                "bootstrap_valid": int(arr.size), "bootstrap_requested": n_boot,
                "n_common_seeds": len(common_seeds), "seeds": common_seeds,
            })
    return out


def hmc_rows(points: list[dict]) -> list[dict]:
    rows = []
    for p in sorted(points, key=lambda x: (x["L"], x["m0sq"], x["seed"])):
        dh = p.get("dH_stats", {})
        rows.append({
            "dim": p["dim"], "L": p["L"], "m0sq": p["m0sq"], "seed": p["seed"],
            "source": p["source"], "n_therm": p["hmc"]["n_therm"],
            "n_measure": p["hmc"]["n_measure"], "measure_every": p["hmc"].get("measure_every", 1),
            "n_leapfrog": p["n_leapfrog"], "step_size_initial": p["hmc"]["step_size_init"],
            "step_size_final": p["step_size_final"], "acceptance": p["acceptance"],
            "dH_mean": dh.get("mean"), "dH_std": dh.get("std"), "dH_max_abs": dh.get("max_abs"),
            "nonfinite_rejects": p.get("nonfinite_rejects", 0),
            "nonfinite_measurements": p.get("nonfinite_measurements", 0),
            "tau_absM": p["tau_absM"]["tau_int"], "N_eff_absM": p["tau_absM"]["N_eff"],
            "runtime_s": p.get("runtime_s"), "diagnostic_pass": p.get("diagnostic_pass"),
        })
    return rows


def make_figures(rows2, cross2, rows3, xi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    fdir = RESULTS / "final_figures"

    def binder_plot(rows, stem):
        fig, ax = plt.subplots(figsize=(5.4, 3.7))
        flagged_label_used = False
        for L in sorted(set(r["L"] for r in rows)):
            rs = sorted([r for r in rows if r["L"] == L], key=lambda x: x["m0sq"])
            ax.errorbar([r["m0sq"] for r in rs], [r["U4"] for r in rs],
                        yerr=[r["U4_se"] for r in rs], marker="o", ms=3, lw=1,
                        capsize=2, label=f"L={L}")
            bad = [r for r in rs if r.get("failed_diagnostic_seeds", 0) >= max(1, math.ceil(0.25 * r["n_seeds"]))]
            if bad:
                ax.scatter([r["m0sq"] for r in bad], [r["U4"] for r in bad],
                           s=45, facecolors="none", edgecolors="crimson", linewidths=1.2,
                           label="diagnostic flag" if not flagged_label_used else None, zorder=5)
                flagged_label_used = True
        ax.set_xlabel(r"$m_0^2$")
        ax.set_ylabel(r"$U_4$")
        ax.legend(ncol=2, frameon=False)
        ax.grid(alpha=0.2)
        for ext in ("png", "pdf"):
            fig.savefig(fdir / f"{stem}.{ext}")
        plt.close(fig)

    if rows2:
        binder_plot(rows2, "fig_2d_refined_binder")
    if rows3:
        binder_plot(rows3, "fig_3d_refined_binder")
    if cross2:
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        labels = [f"{r['L_low']}-{r['L_high']} b{r['branch']+1}" for r in cross2]
        ax.errorbar(np.arange(len(cross2)), [r["m_cross"] for r in cross2],
                    yerr=[r["bootstrap_se"] for r in cross2], fmt="o", capsize=3)
        ax.set_xticks(np.arange(len(cross2)), labels, rotation=25, ha="right")
        ax.set_ylabel(r"crossing $m_0^2$")
        ax.grid(axis="y", alpha=0.2)
        for ext in ("png", "pdf"):
            fig.savefig(fdir / f"fig_2d_refined_crossings.{ext}")
        plt.close(fig)
    if xi:
        for dims, stem in [({2, 3}, "fig_connected_xi_raw_vs_connected"),
                           ({3}, "fig_3d_refined_xi_connected")]:
            use = [r for r in xi if r["dim"] in dims]
            if not use:
                continue
            fig, axes = plt.subplots(1, len(dims), figsize=(5.5 if len(dims) == 1 else 9.0, 3.6), squeeze=False)
            for ax, dim in zip(axes[0], sorted(dims)):
                dr = [r for r in use if r["dim"] == dim]
                for L in sorted(set(r["L"] for r in dr)):
                    rs = sorted([r for r in dr if r["L"] == L], key=lambda x: x["m0sq"])
                    ax.errorbar([r["m0sq"] for r in rs], [r["xi_raw_over_L"] for r in rs],
                                yerr=[r["xi_raw_over_L_se"] for r in rs], ls="--", marker="o", ms=3,
                                alpha=0.55, label=f"raw L={L}")
                    ax.errorbar([r["m0sq"] for r in rs], [r["xi_connected_over_L"] for r in rs],
                                yerr=[r["xi_connected_over_L_se"] for r in rs], ls="-", marker="s", ms=3,
                                label=f"connected L={L}")
                ax.set_xlabel(r"$m_0^2$")
                ax.set_ylabel(r"$\xi/L$")
                ax.set_title(f"{dim}d")
                ax.grid(alpha=0.2)
                ax.legend(frameon=False, ncol=2 if len(dr) > 12 else 1)
            for ext in ("png", "pdf"):
                fig.savefig(fdir / f"{stem}.{ext}")
            plt.close(fig)


def write_reports(cfg, points2, rows2, cross2, points3, rows3, cross3, xi):
    pairs2 = {(r["L_low"], r["L_high"]) for r in cross2}
    multi2 = any(sum((r["L_low"], r["L_high"]) == pair for r in cross2) > 1 for pair in pairs2)
    bad2 = sum(not p.get("diagnostic_pass", False) for p in points2)
    bad3 = sum(not p.get("diagnostic_pass", False) for p in points3)
    status2 = "PASS" if cross2 and not multi2 and bad2 == 0 else "PARTIAL"
    status3 = "PASS" if rows3 and cross3 and bad3 == 0 else "PARTIAL"
    report2 = f"""# 2d refined Binder crossings

Status: **{status2}**

The analysis contains {len(points2)} seed-level chains and {len(rows2)} volume/mass points.
Legacy six-seed measurements were imported with SHA-256 provenance; new measurements retain
per-configuration primary observables. Requested L=96 production is pending the explicit timing gate.

Detected crossing branches: {len(cross2)}. Bootstrap uncertainties resample independent seeds and
retain each branch's interpolation bracket. Multiple branches are reported rather than silently selected.
Chains failing the autocorrelation/ESS diagnostic: {bad2}/{len(points2)}. In particular, legacy ordered-tail
chains with effective sample size below 50 are retained and flagged rather than removed.

No critical exponent is inferred from these data.
"""
    report3 = f"""# 3d refined Wilson--Fisher window

Status: **{status3}**

Completed {len(points3)} seed-level chains and {len(rows3)} volume/mass points on the refined grid.
L=48 is an optional extension and is not included unless explicitly present in the tables.
Binder and connected-correlation-length results are treated as a transition-window diagnostic;
no precision Wilson--Fisher exponent is quoted.
Chains failing the acceptance/nonfinite/ESS diagnostic: {bad3}/{len(points3)}.
"""
    x2 = [r for r in xi if r["dim"] == 2]
    x3 = [r for r in xi if r["dim"] == 3]
    better = sum(np.isfinite(r["xi_connected_over_L"]) for r in xi)
    repx = f"""# Connected xi/L diagnostic

Status: **{'PASS' if xi and better == len(xi) else ('PARTIAL' if xi else 'FAIL')}**

The raw, ordinary-connected, and sector-aligned estimators are computed from identical chains.
The sector estimator uses V(<M^2>-<|M|>^2) and the unchanged nonzero-momentum structure factor;
it is selected for U4>0.30. Invalid square-root bootstrap replicas are counted in the CSV.

- 2d points: {len(x2)}
- 3d points: {len(x3)}
- finite selected connected estimates: {better}/{len(xi)}

Finite values alone do not establish scaling quality; large bootstrap errors or invalid-replicate
fractions keep the interpretation at PARTIAL.
"""
    (RESULTS / "reports" / "2d_refined_crossings_report.md").write_text(report2)
    (RESULTS / "reports" / "3d_refined_wilson_fisher_report.md").write_text(report3)
    (RESULTS / "reports" / "connected_xi_report.md").write_text(repx)


def analyze(cfg: dict) -> None:
    out2 = RESULTS / "2d_refined_crossings"
    out3 = RESULTS / "3d_refined_wilson_fisher"
    points2 = load_legacy_2d(cfg) + load_new_points(out2, 2)
    # Deduplicate with new chains taking precedence.
    p2map = {(p["L"], p["m0sq"], p["seed"]): p for p in points2}
    points2 = list(p2map.values())
    points3 = load_new_points(out3, 3)
    rows2 = point_rows(points2)
    rows3 = point_rows(points3)
    masses2 = [float(x) for x in cfg["2d"]["m0sq_grid"]]
    masses3 = [float(x) for x in cfg["3d"]["m0sq_grid"]]
    cross2 = crossing_rows(points2, [(32, 48), (48, 64), (64, 96)], masses2,
                           int(cfg["bootstrap_replicates"]), 9281)
    cross3 = crossing_rows(points3, [(12, 16), (16, 24), (24, 32), (32, 48)], masses3,
                           int(cfg["bootstrap_replicates"]), 9282)
    xis = xi_rows([p for p in points2 + points3 if p.get("xi")])

    write_csv(out2 / "binder_table.csv", rows2)
    write_csv(out2 / "crossing_table.csv", cross2)
    write_csv(out2 / "hmc_diagnostics.csv", hmc_rows(points2))
    pairs2 = {(r["L_low"], r["L_high"]) for r in cross2}
    multiple_2d_branches = any(sum((r["L_low"], r["L_high"]) == pair for r in cross2) > 1
                               for pair in pairs2)
    bad2 = sum(not p.get("diagnostic_pass", False) for p in points2)
    verdict2 = "PASS" if cross2 and not multiple_2d_branches and bad2 == 0 else "PARTIAL"
    pairs3 = {(r["L_low"], r["L_high"]) for r in cross3}
    multiple_3d_branches = any(sum((r["L_low"], r["L_high"]) == pair for r in cross3) > 1
                               for pair in pairs3)
    bad3 = sum(not p.get("diagnostic_pass", False) for p in points3)
    verdict3 = ("PASS" if len(rows3) >= 28 and cross3 and not multiple_3d_branches and bad3 == 0
                else ("PARTIAL" if rows3 else "FAIL"))
    write_json(out2 / "summary.json", {
        "experiment": "2d_refined_crossings", "status": "complete_lower_volumes_L96_pending",
        "verdict": verdict2,
        "classification_diagnostics": {"multiple_crossing_branches": multiple_2d_branches,
                                       "failed_chain_diagnostics": bad2},
        "config": cfg["2d"], "seed_list": cfg["seeds"], "points": rows2,
        "crossings": cross2, "per_seed": points2,
        "runtime_metadata": runtime_metadata(SOURCE_FILES),
    })
    write_csv(out3 / "binder_table.csv", rows3)
    write_csv(out3 / "crossing_table.csv", cross3)
    write_csv(out3 / "xi_table.csv", [r for r in xis if r["dim"] == 3])
    write_csv(out3 / "hmc_diagnostics.csv", hmc_rows(points3))
    write_json(out3 / "summary.json", {
        "experiment": "3d_refined_wilson_fisher", "status": "in_progress" if len(rows3) < 28 else "complete_lower_volumes_L48_pending",
        "verdict": verdict3,
        "classification_diagnostics": {"multiple_crossing_branches": multiple_3d_branches,
                                       "failed_chain_diagnostics": bad3},
        "config": cfg["3d"], "seed_list": cfg["seeds"], "points": rows3,
        "crossings": cross3, "per_seed": points3,
        "runtime_metadata": runtime_metadata(SOURCE_FILES),
    })
    write_csv(RESULTS / "connected_xi" / "connected_xi_table.csv", xis)
    finite_xi = sum(np.isfinite(r["xi_connected_over_L"]) for r in xis)
    xi_verdict = ("PASS" if xis and finite_xi == len(xis)
                  else ("PARTIAL" if finite_xi else "FAIL"))
    write_json(RESULTS / "connected_xi" / "connected_xi_summary.json", {
        "experiment": "connected_xi", "status": "complete_targeted_campaign",
        "verdict": xi_verdict,
        "classification_diagnostics": {
            "finite_selected_estimates": finite_xi,
            "total_estimates": len(xis),
            "all_selected_estimates_finite": finite_xi == len(xis),
        },
        "estimator_policy": "sector_abs when U4>0.30, connected_M otherwise",
        "seed_lists": {"2d": cfg["2d"]["new_seeds"], "3d": cfg["seeds"]},
        "grid_parameters": {"2d": cfg["2d"], "3d": cfg["3d"]},
        "rows": xis, "runtime_metadata": runtime_metadata(SOURCE_FILES),
    })
    make_figures(rows2, cross2, rows3, xis)
    write_reports(cfg, points2, rows2, cross2, points3, rows3, cross3, xis)


def build_tasks(cfg: dict, campaign: str, overwrite: bool, max_chains: int | None):
    tasks = []
    if campaign in ("2d", "all"):
        c = cfg["2d"]
        out = RESULTS / "2d_refined_crossings"
        for L in c["L_list"]:
            for mass in c["m0sq_grid"]:
                for seed in c["new_seeds"]:
                    tasks.append({"dim": 2, "L": L, "m0sq": mass, "seed": seed,
                                  "lambda": cfg["lambda"], "hmc": c["hmc"],
                                  "n_boot": cfg["bootstrap_replicates"], "n_blocks": cfg["bootstrap_blocks"],
                                  "outdir": str(out), "overwrite": overwrite})
    if campaign in ("3d", "all"):
        c = cfg["3d"]
        out = RESULTS / "3d_refined_wilson_fisher"
        for L in c["L_list"]:
            for mass in c["m0sq_grid"]:
                for seed in cfg["seeds"]:
                    tasks.append({"dim": 3, "L": L, "m0sq": mass, "seed": seed,
                                  "lambda": cfg["lambda"], "hmc": c["hmc"],
                                  "n_boot": cfg["bootstrap_replicates"], "n_blocks": cfg["bootstrap_blocks"],
                                  "outdir": str(out), "overwrite": overwrite})
    if max_chains is not None:
        tasks = tasks[:max_chains]
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--campaign", choices=["2d", "3d", "all", "analyze"], default="analyze")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-chains", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-analyze", action="store_true")
    args = parser.parse_args()
    with args.config.open() as handle:
        cfg = json.load(handle)
    if not (0 <= args.shard_index < args.shard_count):
        parser.error("require 0 <= shard-index < shard-count")
    tasks = build_tasks(cfg, args.campaign, args.overwrite, None)
    tasks = [task for i, task in enumerate(tasks) if i % args.shard_count == args.shard_index]
    if args.max_chains is not None:
        tasks = tasks[:args.max_chains]
    if tasks:
        print(f"running {len(tasks)} scalar chains with {args.workers} worker(s) "
              f"[shard {args.shard_index + 1}/{args.shard_count}]", flush=True)
        if args.workers <= 1:
            for i, task in enumerate(tasks, 1):
                row = run_chain_task(task)
                print(f"[{i}/{len(tasks)}] {row['tag']} acc={row['acceptance']:.3f} t={row['runtime_s']:.1f}s", flush=True)
        else:
            # Threads avoid restricted POSIX semaphore discovery in managed
            # execution environments. NumPy FFTs and vector kernels release
            # the GIL, so independent chains still overlap effectively.
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(run_chain_task, task): task for task in tasks}
                for i, future in enumerate(as_completed(futures), 1):
                    row = future.result()
                    print(f"[{i}/{len(tasks)}] {row['tag']} acc={row['acceptance']:.3f} t={row['runtime_s']:.1f}s", flush=True)
    if not args.skip_analyze:
        analyze(cfg)


if __name__ == "__main__":
    main()
