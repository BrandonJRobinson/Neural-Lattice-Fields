#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CFG = json.loads(
    (
        ROOT
        / "configs"
        / "moment_engineering_p1c_s1_preregistration.json"
    ).read_text()
)

RAW = (
    ROOT
    / "results"
    / "moment_engineering_p1c_s1"
    / "raw"
)

OUT = (
    ROOT
    / "results"
    / "moment_engineering_p1c_s1"
    / "analysis"
)


def conservative_seed_summary(
    values,
    within_errors,
):
    x = np.asarray(
        values,
        dtype=float,
    )

    e = np.asarray(
        within_errors,
        dtype=float,
    )

    mean = float(
        np.mean(x)
    )

    between = (
        float(
            np.std(
                x,
                ddof=1,
            )
            / np.sqrt(len(x))
        )
        if len(x) > 1
        else float("nan")
    )

    within = float(
        np.sqrt(
            np.sum(e ** 2)
        )
        / len(e)
    )

    se = max(
        between,
        within,
    )

    z = (
        abs(mean) / se
        if se > 0
        else float("inf")
    )

    return {
        "mean": mean,
        "between_seed_se": between,
        "within_seed_se": within,
        "conservative_se": se,
        "abs_z": z,
        "n_seeds": len(x),
    }


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return

    keys = list(rows[0])

    with path.open(
        "w",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=keys,
        )

        w.writeheader()
        w.writerows(rows)


def main():
    widths = CFG["widths"]
    seeds = CFG["seeds"]

    expected = {
        RAW
        / f"gaussian_N{N}_seed{seed}.json"
        for N in widths
        for seed in seeds
    }

    actual = set(
        RAW.glob("*.json")
    )

    missing = sorted(
        expected - actual
    )

    extra = sorted(
        actual - expected
    )

    print(
        f"expected={len(expected)} "
        f"actual={len(actual)} "
        f"missing={len(missing)} "
        f"extra={len(extra)}"
    )

    if missing or extra:
        raise SystemExit(2)

    data = [
        json.loads(p.read_text())
        for p in sorted(actual)
    ]

    width_rows = []

    estimator_names = [
        "legacy_global_plugin",
        "sitewise_plugin",
        "sitewise_unbiased_U",
    ]

    for N in widths:
        ds = [
            d
            for d in data
            if int(d["N"]) == int(N)
        ]

        for name in estimator_names:
            vals = [
                d["results"]
                ["estimators"]
                [name]
                ["N2_value"]
                for d in ds
            ]

            errs = [
                d["results"]
                ["estimators"]
                [name]
                ["N2_jk_se"]
                for d in ds
            ]

            s = conservative_seed_summary(
                vals,
                errs,
            )

            width_rows.append({
                "N": N,
                "estimator": name,
                **s,
                "target": 0.0,
            })

    bias_rows = []

    for N in widths:
        ds = [
            d
            for d in data
            if int(d["N"]) == int(N)
        ]

        for name in [
            "legacy_minus_U",
            "site_plugin_minus_U",
        ]:
            vals = [
                d["results"]
                ["differences"]
                [name]
                ["N2_value"]
                for d in ds
            ]

            errs = [
                d["results"]
                ["differences"]
                [name]
                ["N2_jk_se"]
                for d in ds
            ]

            s = conservative_seed_summary(
                vals,
                errs,
            )

            bias_rows.append({
                "N": N,
                "difference": name,
                **s,
            })

    primary = [
        x
        for x in width_rows
        if x["estimator"]
        == "sitewise_unbiased_U"
    ]

    per_width_pass = all(
        x["abs_z"]
        <= CFG[
            "primary_analysis"
        ]["per_width_abs_z_max"]
        for x in primary
    )

    vals = np.asarray(
        [x["mean"] for x in primary]
    )

    errs = np.asarray(
        [
            x["conservative_se"]
            for x in primary
        ]
    )

    good = (
        np.isfinite(vals)
        &
        np.isfinite(errs)
        &
        (errs > 0)
    )

    if np.count_nonzero(good) > 1:
        chi2 = float(
            np.sum(
                (
                    vals[good]
                    / errs[good]
                ) ** 2
            )
        )

        dof = (
            np.count_nonzero(good)
            - 1
        )

        chi2_dof = (
            chi2 / dof
        )
    else:
        chi2_dof = float("nan")

    chi_pass = (
        not np.isfinite(chi2_dof)
        or
        chi2_dof
        <= CFG[
            "primary_analysis"
        ]["width_chi2_dof_max"]
    )

    aggregate_pass = (
        per_width_pass
        and chi_pass
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        OUT / "width_summary.csv",
        width_rows,
    )

    write_csv(
        OUT / "estimator_bias.csv",
        bias_rows,
    )

    verdict = {
        "campaign": CFG["campaign"],

        "artifacts": {
            "expected": len(expected),
            "actual": len(actual),
            "missing": len(missing),
            "extra": len(extra),
        },

        "primary_estimator": (
            "sitewise_unbiased_U"
        ),

        "per_width_pass": per_width_pass,
        "width_chi2_dof": chi2_dof,
        "chi2_pass": chi_pass,
        "aggregate_pass": aggregate_pass,

        "interpretation_if_pass": (
            "P1c Gaussian sextic anomaly is not reproduced "
            "by the exactly unbiased U-statistic estimator; "
            "original miss is attributable to finite-sample "
            "plug-in estimation."
        ),
    }

    (
        OUT / "verdict.json"
    ).write_text(
        json.dumps(
            verdict,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("========================================")
    print("P1c-S1 WIDTH SUMMARY")
    print("========================================")

    for x in width_rows:
        print(
            f"N={int(x['N']):2d} "
            f"{x['estimator']:24s} "
            f"N^2*k6={x['mean']:+.8f} "
            f"+/- {x['conservative_se']:.8f} "
            f"z={x['abs_z']:.2f}"
        )

    print()
    print("========================================")
    print("ESTIMATOR DIFFERENCES")
    print("========================================")

    for x in bias_rows:
        print(
            f"N={int(x['N']):2d} "
            f"{x['difference']:20s} "
            f"delta={x['mean']:+.8f} "
            f"+/- {x['conservative_se']:.8f} "
            f"z={x['abs_z']:.2f}"
        )

    print()
    print("========================================")
    print("VERDICT")
    print("========================================")
    print(
        json.dumps(
            verdict,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
