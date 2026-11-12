#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from lattice_nnft.spectral_prior_architectures import (
    CosRFF_TwoScale,
    _d2_grid,
)


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs"
    / "moment_engineering_p1c_v1_preregistration.json"
)


def amplitude_moments(kind: str):
    if kind == "fixed":
        return 1.0, 1.0, 1.0

    if kind == "gaussian":
        return 1.0, 3.0, 15.0

    if kind == "quartic_null":
        return 1.0, 2.0, 4.0

    raise ValueError(kind)


def analytic_targets(arch, kind: str, separation: int):
    _, m4, m6 = amplitude_moments(kind)

    d2 = _d2_grid(arch.L)
    d = float(d2[0, separation % arch.L])

    c1 = (
        arch.pi1
        * math.exp(-(arch.sigma1 ** 2) * d)
        +
        (1.0 - arch.pi1)
        * math.exp(-(arch.sigma2 ** 2) * d)
    )

    c2 = (
        arch.pi1
        * math.exp(-4.0 * (arch.sigma1 ** 2) * d)
        +
        (1.0 - arch.pi1)
        * math.exp(-4.0 * (arch.sigma2 ** 2) * d)
    )

    s = arch.sigma_c

    g = 0.5 * s * s * c1

    nk22 = s ** 4 * (
        m4 * (0.25 + 0.125 * c2)
        - 0.25
        - 0.5 * c1 * c1
    )

    nk4_local = (
        3.0
        * s ** 4
        / 8.0
        * (m4 - 2.0)
    )

    n2k6_local = (
        5.0
        * s ** 6
        / 16.0
        * (m6 - 9.0 * m4 + 12.0)
    )

    return {
        "d2": d,
        "C1": c1,
        "C2": c2,
        "G": g,
        "N_kappa22": nk22,
        "N_kappa4_local": nk4_local,
        "N2_kappa6_local": n2k6_local,
    }


def jk_error(vals):
    vals = np.asarray(vals, dtype=float)

    if vals.size <= 1:
        return float("nan")

    cen = float(np.mean(vals))

    return float(
        np.sqrt(
            (vals.size - 1)
            / vals.size
            * np.sum((vals - cen) ** 2)
        )
    )


def measure_configs(
    configs: np.ndarray,
    L: int,
    separations: list[int],
    n_blocks: int,
):
    M, V = configs.shape

    if V != L * L:
        raise ValueError(
            f"config volume {V} does not match L={L}"
        )

    blocks = [
        x
        for x in np.array_split(np.arange(M), min(n_blocks, M))
        if x.size
    ]

    B = len(blocks)
    R = len(separations)

    counts = np.zeros(B, dtype=int)

    q1_sums = np.zeros(B)
    q2_sums = np.zeros(B)
    q4_sums = np.zeros(B)
    q6_sums = np.zeros(B)

    corr_sums = np.zeros((B, R))
    q22_sums = np.zeros((B, R))

    power_sums = np.zeros(
        (B, L, L),
        dtype=float,
    )

    for ib, inds in enumerate(blocks):
        f = configs[inds].reshape((-1, L, L))
        nb = f.shape[0]
        counts[ib] = nb

        f2 = f * f

        q1_sums[ib] = np.sum(
            np.mean(f, axis=(1, 2))
        )

        q2_sums[ib] = np.sum(
            np.mean(f2, axis=(1, 2))
        )

        q4_sums[ib] = np.sum(
            np.mean(f2 * f2, axis=(1, 2))
        )

        q6_sums[ib] = np.sum(
            np.mean(f2 * f2 * f2, axis=(1, 2))
        )

        for ir, r in enumerate(separations):
            fy = np.roll(
                f,
                -int(r),
                axis=2,
            )

            f2y = np.roll(
                f2,
                -int(r),
                axis=2,
            )

            corr_sums[ib, ir] = np.sum(
                np.mean(
                    f * fy,
                    axis=(1, 2),
                )
            )

            q22_sums[ib, ir] = np.sum(
                np.mean(
                    f2 * f2y,
                    axis=(1, 2),
                )
            )

        ph = np.fft.fft2(
            f,
            axes=(1, 2),
        )

        power_sums[ib] = np.sum(
            np.abs(ph) ** 2 / V,
            axis=0,
        )

    ntot = int(np.sum(counts))

    q1_total = float(np.sum(q1_sums))
    q2_total = float(np.sum(q2_sums))
    q4_total = float(np.sum(q4_sums))
    q6_total = float(np.sum(q6_sums))

    corr_total = np.sum(
        corr_sums,
        axis=0,
    )

    q22_total = np.sum(
        q22_sums,
        axis=0,
    )

    power_total = np.sum(
        power_sums,
        axis=0,
    )

    q1 = q1_total / ntot
    q2 = q2_total / ntot
    q4 = q4_total / ntot
    q6 = q6_total / ntot

    kappa4 = q4 - 3.0 * q2 * q2

    kappa6 = (
        q6
        - 15.0 * q4 * q2
        + 30.0 * q2 ** 3
    )

    kmom = power_total / ntot

    sep = {}

    for ir, r in enumerate(separations):
        g = corr_total[ir] / ntot
        q22 = q22_total[ir] / ntot

        k22 = (
            q22
            - q2 * q2
            - 2.0 * g * g
        )

        sep[str(r)] = {
            "G": float(g),
            "q22": float(q22),
            "kappa22": float(k22),
        }

    jk_q2 = []
    jk_k4 = []
    jk_k6 = []

    jk_g = {
        str(r): []
        for r in separations
    }

    jk_k22 = {
        str(r): []
        for r in separations
    }

    for ib in range(B):
        nsub = ntot - counts[ib]

        q2b = (
            q2_total
            - q2_sums[ib]
        ) / nsub

        q4b = (
            q4_total
            - q4_sums[ib]
        ) / nsub

        q6b = (
            q6_total
            - q6_sums[ib]
        ) / nsub

        k4b = q4b - 3.0 * q2b * q2b

        k6b = (
            q6b
            - 15.0 * q4b * q2b
            + 30.0 * q2b ** 3
        )

        jk_q2.append(q2b)
        jk_k4.append(k4b)
        jk_k6.append(k6b)

        for ir, r in enumerate(separations):
            gb = (
                corr_total[ir]
                - corr_sums[ib, ir]
            ) / nsub

            q22b = (
                q22_total[ir]
                - q22_sums[ib, ir]
            ) / nsub

            k22b = (
                q22b
                - q2b * q2b
                - 2.0 * gb * gb
            )

            jk_g[str(r)].append(gb)
            jk_k22[str(r)].append(k22b)

    out = {
        "M": ntot,
        "n_blocks": B,
        "field_mean": float(q1),

        "local": {
            "m2": float(q2),
            "m4": float(q4),
            "m6": float(q6),
            "m2_jk_se": jk_error(jk_q2),

            "kappa4": float(kappa4),
            "kappa4_jk_se": jk_error(jk_k4),

            "kappa6": float(kappa6),
            "kappa6_jk_se": jk_error(jk_k6),
        },

        "separated": sep,

        "kernel_mom": kmom.tolist(),
    }

    for r in separations:
        out["separated"][str(r)]["G_jk_se"] = (
            jk_error(jk_g[str(r)])
        )

        out["separated"][str(r)]["kappa22_jk_se"] = (
            jk_error(jk_k22[str(r)])
        )

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kind",
        required=True,
        choices=[
            "fixed",
            "gaussian",
            "quartic_null",
        ],
    )

    parser.add_argument(
        "--width",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--jackknife-blocks",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    cfg = json.loads(
        CONFIG.read_text()
    )

    N = int(args.width)
    seed = int(args.seed)

    M = (
        int(args.samples)
        if args.samples is not None
        else int(cfg["samples_per_replica"])
    )

    n_jk = (
        int(args.jackknife_blocks)
        if args.jackknife_blocks is not None
        else int(cfg["jackknife_blocks"])
    )

    p = cfg["twoscale"]

    arch = CosRFF_TwoScale(
        cfg["L"],
        sigma1=p["sigma1"],
        sigma2=p["sigma2"],
        pi1=p["pi1"],
        sigma_c=p["sigma_c"],
    )

    geometry_seed = (
        seed
        + 10000 * N
    )

    amplitude_seed = (
        seed
        + 100000000
        + 10000 * N
    )

    t0 = time.perf_counter()

    configs = arch.generate(
        N,
        M,
        batch_size=cfg["batch_size"],
        seed=geometry_seed,
        amplitude_law=args.kind,
        amplitude_seed=amplitude_seed,
    )

    nonfinite = int(
        configs.size
        - np.count_nonzero(
            np.isfinite(configs)
        )
    )

    meas = measure_configs(
        configs,
        cfg["L"],
        cfg["separations"],
        n_jk,
    )

    kmom_emp = np.asarray(
        meas.pop("kernel_mom"),
        dtype=float,
    )

    kmom_target = np.asarray(
        arch.kernel_mom(),
        dtype=float,
    )

    diff = kmom_emp - kmom_target

    kernel_l2 = float(
        np.linalg.norm(diff)
        /
        (
            np.linalg.norm(kmom_target)
            + 1e-30
        )
    )

    kernel_rms_rel = float(
        np.sqrt(
            np.mean(
                (
                    diff
                    /
                    (
                        np.abs(kmom_target)
                        + 1e-30
                    )
                ) ** 2
            )
        )
    )

    _, m4, m6 = amplitude_moments(
        args.kind
    )

    for r in cfg["separations"]:
        th = analytic_targets(
            arch,
            args.kind,
            r,
        )

        d = meas["separated"][str(r)]

        d["N_kappa22"] = (
            N * d["kappa22"]
        )

        d["N_kappa22_jk_se"] = (
            N * d["kappa22_jk_se"]
        )

        d["target_G"] = th["G"]
        d["target_N_kappa22"] = (
            th["N_kappa22"]
        )

        d["C1"] = th["C1"]
        d["C2"] = th["C2"]
        d["d2"] = th["d2"]

    th0 = analytic_targets(
        arch,
        args.kind,
        0,
    )

    loc = meas["local"]

    loc["N_kappa4"] = (
        N * loc["kappa4"]
    )

    loc["N_kappa4_jk_se"] = (
        N * loc["kappa4_jk_se"]
    )

    loc["N2_kappa6"] = (
        N * N * loc["kappa6"]
    )

    loc["N2_kappa6_jk_se"] = (
        N * N * loc["kappa6_jk_se"]
    )

    loc["target_N_kappa4"] = (
        th0["N_kappa4_local"]
    )

    loc["target_N2_kappa6"] = (
        th0["N2_kappa6_local"]
    )

    result = {
        "campaign": cfg["campaign"],
        "kind": args.kind,
        "N": N,
        "seed": seed,
        "geometry_seed": geometry_seed,
        "amplitude_seed": amplitude_seed,
        "L": cfg["L"],
        "samples": M,

        "amplitude_moments": {
            "m2": 1.0,
            "m4": m4,
            "m6": m6,
        },

        "twoscale": p,

        "nonfinite_count": nonfinite,

        "kernel": {
            "l2_relative_to_nngp": kernel_l2,
            "rms_relative_existing_metric": kernel_rms_rel,
            "empirical_momentum": kmom_emp.tolist(),
        },

        "measurements": meas,

        "runtime_s": (
            time.perf_counter()
            - t0
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        json.dumps(
            {
                "kind": args.kind,
                "N": N,
                "seed": seed,
                "samples": M,
                "nonfinite": nonfinite,
                "G0": meas["separated"]["0"]["G"],
                "target_G0": meas["separated"]["0"]["target_G"],
                "N_kappa4": loc["N_kappa4"],
                "target_N_kappa4": loc["target_N_kappa4"],
                "N2_kappa6": loc["N2_kappa6"],
                "target_N2_kappa6": loc["target_N2_kappa6"],
                "kernel_l2": kernel_l2,
                "runtime_s": result["runtime_s"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
