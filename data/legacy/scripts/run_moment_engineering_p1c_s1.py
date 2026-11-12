#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lattice_nnft.spectral_prior_architectures import CosRFF_TwoScale


ROOT = Path(__file__).resolve().parents[1]

CFG_PATH = (
    ROOT
    / "configs"
    / "moment_engineering_p1c_s1_preregistration.json"
)


def estimators_from_site_sums(
    S2: np.ndarray,
    S4: np.ndarray,
    S6: np.ndarray,
    n: int,
):
    """
    S2,S4,S6 have shape (V,) and are sums over n independent
    field configurations at each fixed lattice site.
    """
    if n < 3:
        raise ValueError("need n >= 3")

    mu2_site = S2 / n
    mu4_site = S4 / n
    mu6_site = S6 / n

    # -------------------------------------------------------------
    # Historical/global plug-in estimator
    # -------------------------------------------------------------
    mu2_global = float(np.mean(mu2_site))
    mu4_global = float(np.mean(mu4_site))
    mu6_global = float(np.mean(mu6_site))

    legacy_global = (
        mu6_global
        - 15.0 * mu4_global * mu2_global
        + 30.0 * mu2_global ** 3
    )

    # -------------------------------------------------------------
    # Sitewise plug-in estimator
    # -------------------------------------------------------------
    site_plugin_vec = (
        mu6_site
        - 15.0 * mu4_site * mu2_site
        + 30.0 * mu2_site ** 3
    )

    site_plugin = float(
        np.mean(site_plugin_vec)
    )

    # -------------------------------------------------------------
    # Exactly unbiased U-statistic products
    #
    # sum_{i != j} Xi^4 Xj^2
    #     = S4*S2 - S6
    #
    # sum_{i,j,k all distinct} Xi^2 Xj^2 Xk^2
    #     = S2^3 - 3*S2*S4 + 2*S6
    # -------------------------------------------------------------
    u42 = (
        S4 * S2 - S6
    ) / (n * (n - 1))

    u222 = (
        S2 ** 3
        - 3.0 * S2 * S4
        + 2.0 * S6
    ) / (
        n
        * (n - 1)
        * (n - 2)
    )

    unbiased_vec = (
        mu6_site
        - 15.0 * u42
        + 30.0 * u222
    )

    unbiased = float(
        np.mean(unbiased_vec)
    )

    return {
        "legacy_global_plugin": legacy_global,
        "sitewise_plugin": site_plugin,
        "sitewise_unbiased_U": unbiased,

        "moments": {
            "m2": mu2_global,
            "m4": mu4_global,
            "m6": mu6_global,
        },

        "differences": {
            "legacy_minus_U": legacy_global - unbiased,
            "site_plugin_minus_U": site_plugin - unbiased,
        },
    }


def jackknife_from_blocks(
    block_S2,
    block_S4,
    block_S6,
    block_n,
):
    B = len(block_n)

    full_S2 = np.sum(block_S2, axis=0)
    full_S4 = np.sum(block_S4, axis=0)
    full_S6 = np.sum(block_S6, axis=0)

    n_full = int(np.sum(block_n))

    full = estimators_from_site_sums(
        full_S2,
        full_S4,
        full_S6,
        n_full,
    )

    names = [
        "legacy_global_plugin",
        "sitewise_plugin",
        "sitewise_unbiased_U",
    ]

    jk = {
        name: []
        for name in names
    }

    jk_diff = {
        "legacy_minus_U": [],
        "site_plugin_minus_U": [],
    }

    for b in range(B):
        n_sub = n_full - int(block_n[b])

        est = estimators_from_site_sums(
            full_S2 - block_S2[b],
            full_S4 - block_S4[b],
            full_S6 - block_S6[b],
            n_sub,
        )

        for name in names:
            jk[name].append(
                est[name]
            )

        for name in jk_diff:
            jk_diff[name].append(
                est["differences"][name]
            )

    def jk_se(vals):
        vals = np.asarray(
            vals,
            dtype=float,
        )

        if len(vals) <= 1:
            return float("nan")

        cen = float(
            np.mean(vals)
        )

        return float(
            np.sqrt(
                (len(vals) - 1)
                / len(vals)
                * np.sum(
                    (vals - cen) ** 2
                )
            )
        )

    out = {
        "n_samples": n_full,
        "n_blocks": B,
        "estimators": {},
        "differences": {},
        "moments": full["moments"],
    }

    for name in names:
        out["estimators"][name] = {
            "value": float(full[name]),
            "jk_se": jk_se(jk[name]),
        }

    for name in jk_diff:
        out["differences"][name] = {
            "value": float(
                full["differences"][name]
            ),
            "jk_se": jk_se(
                jk_diff[name]
            ),
        }

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--width",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    cfg = json.loads(
        CFG_PATH.read_text()
    )

    N = int(args.width)
    seed = int(args.seed)

    if N not in cfg["widths"]:
        raise ValueError(
            f"N={N} not preregistered"
        )

    M = int(
        cfg["samples_per_replica"]
    )

    B = int(
        cfg["jackknife_blocks"]
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
        + 200000000
        + 10000 * N
    )

    t0 = time.perf_counter()

    configs = arch.generate(
        N,
        M,
        batch_size=cfg["batch_size"],
        seed=geometry_seed,
        amplitude_law="gaussian",
        amplitude_seed=amplitude_seed,
    )

    nonfinite = int(
        configs.size
        - np.count_nonzero(
            np.isfinite(configs)
        )
    )

    if nonfinite:
        raise RuntimeError(
            f"nonfinite generated values: {nonfinite}"
        )

    V = configs.shape[1]

    blocks = [
        x
        for x in np.array_split(
            np.arange(M),
            B,
        )
        if len(x)
    ]

    block_n = np.asarray(
        [len(x) for x in blocks],
        dtype=int,
    )

    block_S2 = np.empty(
        (len(blocks), V),
        dtype=float,
    )

    block_S4 = np.empty_like(
        block_S2
    )

    block_S6 = np.empty_like(
        block_S2
    )

    for ib, inds in enumerate(blocks):
        x = configs[inds]

        x2 = x * x
        x4 = x2 * x2

        block_S2[ib] = np.sum(
            x2,
            axis=0,
        )

        block_S4[ib] = np.sum(
            x4,
            axis=0,
        )

        block_S6[ib] = np.sum(
            x4 * x2,
            axis=0,
        )

    est = jackknife_from_blocks(
        block_S2,
        block_S4,
        block_S6,
        block_n,
    )

    for obj in est["estimators"].values():
        obj["N2_value"] = (
            N * N * obj["value"]
        )

        obj["N2_jk_se"] = (
            N * N * obj["jk_se"]
        )

    for obj in est["differences"].values():
        obj["N2_value"] = (
            N * N * obj["value"]
        )

        obj["N2_jk_se"] = (
            N * N * obj["jk_se"]
        )

    result = {
        "campaign": cfg["campaign"],
        "N": N,
        "seed": seed,
        "samples": M,
        "L": cfg["L"],

        "geometry_seed": geometry_seed,
        "amplitude_seed": amplitude_seed,

        "target_N2_kappa6": 0.0,

        "nonfinite_count": nonfinite,

        "results": est,

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
                "N": N,
                "seed": seed,
                "samples": M,

                "legacy_N2": (
                    est["estimators"]
                    ["legacy_global_plugin"]
                    ["N2_value"]
                ),

                "site_plugin_N2": (
                    est["estimators"]
                    ["sitewise_plugin"]
                    ["N2_value"]
                ),

                "unbiased_U_N2": (
                    est["estimators"]
                    ["sitewise_unbiased_U"]
                    ["N2_value"]
                ),

                "unbiased_U_se": (
                    est["estimators"]
                    ["sitewise_unbiased_U"]
                    ["N2_jk_se"]
                ),

                "legacy_minus_U": (
                    est["differences"]
                    ["legacy_minus_U"]
                    ["N2_value"]
                ),

                "runtime_s": result[
                    "runtime_s"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
