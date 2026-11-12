#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs"
    / "moment_engineering_p1c_v1_preregistration.json"
)

RAW = (
    ROOT
    / "results"
    / "moment_engineering_p1c_v1"
    / "raw"
)

OUT = (
    ROOT
    / "results"
    / "moment_engineering_p1c_v1"
    / "analysis"
)


def combine(values, errors):
    values = np.asarray(
        values,
        dtype=float,
    )

    errors = np.asarray(
        errors,
        dtype=float,
    )

    good = (
        np.isfinite(values)
        &
        np.isfinite(errors)
        &
        (errors > 0)
    )

    if not np.any(good):
        goodv = values[
            np.isfinite(values)
        ]

        if goodv.size == 0:
            return {
                "mean": float("nan"),
                "within_se": float("nan"),
                "between_se": float("nan"),
                "se": float("nan"),
                "n": 0,
            }

        mean = float(
            np.mean(goodv)
        )

        between = (
            float(
                np.std(
                    goodv,
                    ddof=1,
                )
                / np.sqrt(goodv.size)
            )
            if goodv.size > 1
            else float("nan")
        )

        return {
            "mean": mean,
            "within_se": float("nan"),
            "between_se": between,
            "se": between,
            "n": int(goodv.size),
        }

    v = values[good]
    e = errors[good]

    w = 1.0 / (e * e)

    mean = float(
        np.sum(w * v)
        / np.sum(w)
    )

    within = float(
        1.0
        / np.sqrt(
            np.sum(w)
        )
    )

    between = (
        float(
            np.std(
                v,
                ddof=1,
            )
            / np.sqrt(v.size)
        )
        if v.size > 1
        else 0.0
    )

    se = max(
        within,
        between,
    )

    return {
        "mean": mean,
        "within_se": within,
        "between_se": between,
        "se": se,
        "n": int(v.size),
    }


def zscore(mean, target, se):
    if not (
        np.isfinite(mean)
        and np.isfinite(target)
        and np.isfinite(se)
    ):
        return float("inf")

    if se == 0:
        return (
            0.0
            if mean == target
            else float("inf")
        )

    return abs(mean - target) / se


def chi2_dof(entries, target):
    vals = np.asarray(
        [x["mean"] for x in entries],
        dtype=float,
    )

    errs = np.asarray(
        [x["se"] for x in entries],
        dtype=float,
    )

    good = (
        np.isfinite(vals)
        &
        np.isfinite(errs)
        &
        (errs > 0)
    )

    if np.count_nonzero(good) <= 1:
        return float("nan")

    chi2 = float(
        np.sum(
            (
                (vals[good] - target)
                / errs[good]
            ) ** 2
        )
    )

    dof = max(
        int(np.count_nonzero(good)) - 1,
        1,
    )

    return chi2 / dof


def pool_widths(entries, target):
    c = combine(
        [x["mean"] for x in entries],
        [x["se"] for x in entries],
    )

    c["target"] = float(target)

    c["abs_z"] = zscore(
        c["mean"],
        target,
        c["se"],
    )

    c["max_width_abs_z"] = max(
        (
            zscore(
                x["mean"],
                target,
                x["se"],
            )
            for x in entries
        ),
        default=float("inf"),
    )

    c["chi2_dof"] = chi2_dof(
        entries,
        target,
    )

    return c


def write_csv(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text("")
        return

    keys = []

    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)

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
    cfg = json.loads(
        CONFIG.read_text()
    )

    kinds = list(
        cfg["amplitude_laws"]
    )

    widths = [
        int(x)
        for x in cfg["widths"]
    ]

    seeds = [
        int(x)
        for x in cfg["seeds"]
    ]

    seps = [
        int(x)
        for x in cfg["separations"]
    ]

    expected = {
        RAW
        / f"{kind}_N{N}_seed{seed}.json"
        for kind in kinds
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
        print("ARTIFACT GATE: FAIL")

        if missing:
            print("missing:")
            for x in missing:
                print(" ", x)

        if extra:
            print("extra:")
            for x in extra:
                print(" ", x)

        raise SystemExit(2)

    data = []

    for path in sorted(actual):
        d = json.loads(
            path.read_text()
        )

        d["_path"] = str(path)
        data.append(d)

    if any(
        int(d["nonfinite_count"]) != 0
        for d in data
    ):
        raise SystemExit(
            "ERROR: nonfinite production values found"
        )

    # -------------------------------------------------------------
    # Width-level covariance and vertex summaries
    # -------------------------------------------------------------

    covariance_width = []
    vertex_width = []
    sextic_width = []

    for kind in kinds:
        for N in widths:
            rows = [
                d
                for d in data
                if d["kind"] == kind
                and int(d["N"]) == N
            ]

            loc_vals = [
                d["measurements"]["local"]["N2_kappa6"]
                for d in rows
            ]

            loc_errs = [
                d["measurements"]["local"]["N2_kappa6_jk_se"]
                for d in rows
            ]

            sx = combine(
                loc_vals,
                loc_errs,
            )

            target_sx = float(
                rows[0]["measurements"]["local"]["target_N2_kappa6"]
            )

            sextic_width.append({
                "kind": kind,
                "N": N,
                **sx,
                "target": target_sx,
                "abs_z": zscore(
                    sx["mean"],
                    target_sx,
                    sx["se"],
                ),
            })

            for r in seps:
                rr = str(r)

                g = combine(
                    [
                        d["measurements"]["separated"][rr]["G"]
                        for d in rows
                    ],
                    [
                        d["measurements"]["separated"][rr]["G_jk_se"]
                        for d in rows
                    ],
                )

                tg = float(
                    rows[0]["measurements"]["separated"][rr]["target_G"]
                )

                covariance_width.append({
                    "kind": kind,
                    "N": N,
                    "separation": r,
                    **g,
                    "target": tg,
                    "abs_z": zscore(
                        g["mean"],
                        tg,
                        g["se"],
                    ),
                })

                v = combine(
                    [
                        d["measurements"]["separated"][rr]["N_kappa22"]
                        for d in rows
                    ],
                    [
                        d["measurements"]["separated"][rr]["N_kappa22_jk_se"]
                        for d in rows
                    ],
                )

                tv = float(
                    rows[0]["measurements"]["separated"][rr]["target_N_kappa22"]
                )

                vertex_width.append({
                    "kind": kind,
                    "N": N,
                    "separation": r,
                    **v,
                    "target": tv,
                    "abs_z": zscore(
                        v["mean"],
                        tv,
                        v["se"],
                    ),
                })

    # -------------------------------------------------------------
    # Paired covariance law differences
    # -------------------------------------------------------------

    paired_width = []

    for a, b in combinations(kinds, 2):
        for N in widths:
            for r in seps:
                rr = str(r)

                diffs = []
                diff_errs = []

                for seed in seeds:
                    da = next(
                        d
                        for d in data
                        if d["kind"] == a
                        and int(d["N"]) == N
                        and int(d["seed"]) == seed
                    )

                    db = next(
                        d
                        for d in data
                        if d["kind"] == b
                        and int(d["N"]) == N
                        and int(d["seed"]) == seed
                    )

                    xa = da["measurements"]["separated"][rr]
                    xb = db["measurements"]["separated"][rr]

                    diffs.append(
                        float(xa["G"])
                        - float(xb["G"])
                    )

                    # Conservative because the common-geometry covariance
                    # is ignored in this within-replica error.
                    diff_errs.append(
                        math.sqrt(
                            float(xa["G_jk_se"]) ** 2
                            +
                            float(xb["G_jk_se"]) ** 2
                        )
                    )

                c = combine(
                    diffs,
                    diff_errs,
                )

                paired_width.append({
                    "law_a": a,
                    "law_b": b,
                    "N": N,
                    "separation": r,
                    **c,
                    "target": 0.0,
                    "abs_z": zscore(
                        c["mean"],
                        0.0,
                        c["se"],
                    ),
                })

    # -------------------------------------------------------------
    # Pool across widths
    # -------------------------------------------------------------

    covariance_pooled = []
    vertex_pooled = []
    paired_pooled = []
    sextic_pooled = []

    gcfg = cfg["primary_gates"]

    for kind in kinds:
        for r in seps:
            xs = [
                x
                for x in covariance_width
                if x["kind"] == kind
                and x["separation"] == r
            ]

            target = float(
                xs[0]["target"]
            )

            p = pool_widths(
                xs,
                target,
            )

            passed = (
                p["abs_z"]
                <= gcfg["covariance_against_same_nngp"]["pooled_abs_z_max"]
                and
                p["max_width_abs_z"]
                <= gcfg["covariance_against_same_nngp"]["individual_width_abs_z_max"]
            )

            covariance_pooled.append({
                "kind": kind,
                "separation": r,
                **p,
                "pass": bool(passed),
            })

            ys = [
                x
                for x in vertex_width
                if x["kind"] == kind
                and x["separation"] == r
            ]

            target = float(
                ys[0]["target"]
            )

            p = pool_widths(
                ys,
                target,
            )

            passed = (
                p["abs_z"]
                <= gcfg["spatial_connected_quartic"]["pooled_abs_z_max"]
                and
                p["max_width_abs_z"]
                <= gcfg["spatial_connected_quartic"]["individual_width_abs_z_max"]
                and
                (
                    not np.isfinite(p["chi2_dof"])
                    or
                    p["chi2_dof"]
                    <= gcfg["spatial_connected_quartic"]["width_chi2_dof_max"]
                )
            )

            vertex_pooled.append({
                "kind": kind,
                "separation": r,
                **p,
                "pass": bool(passed),
            })

    for a, b in combinations(kinds, 2):
        for r in seps:
            xs = [
                x
                for x in paired_width
                if x["law_a"] == a
                and x["law_b"] == b
                and x["separation"] == r
            ]

            p = pool_widths(
                xs,
                0.0,
            )

            passed = (
                p["abs_z"]
                <= gcfg["paired_covariance_between_amplitude_laws"]["pooled_abs_z_max"]
                and
                p["max_width_abs_z"]
                <= gcfg["paired_covariance_between_amplitude_laws"]["individual_width_abs_z_max"]
            )

            paired_pooled.append({
                "law_a": a,
                "law_b": b,
                "separation": r,
                **p,
                "pass": bool(passed),
            })

    primary_sx_widths = set(
        gcfg["local_sextic_primary_widths"]["widths"]
    )

    for kind in kinds:
        xs = [
            x
            for x in sextic_width
            if x["kind"] == kind
            and x["N"] in primary_sx_widths
        ]

        target = float(
            xs[0]["target"]
        )

        p = pool_widths(
            xs,
            target,
        )

        passed = (
            p["abs_z"]
            <= gcfg["local_sextic_primary_widths"]["pooled_abs_z_max"]
            and
            p["max_width_abs_z"]
            <= gcfg["local_sextic_primary_widths"]["individual_width_abs_z_max"]
            and
            (
                not np.isfinite(p["chi2_dof"])
                or
                p["chi2_dof"]
                <= gcfg["local_sextic_primary_widths"]["width_chi2_dof_max"]
            )
        )

        sextic_pooled.append({
            "kind": kind,
            "primary_widths": sorted(primary_sx_widths),
            **p,
            "pass": bool(passed),
        })

    # -------------------------------------------------------------
    # Whole-kernel descriptive diagnostics
    # -------------------------------------------------------------

    kernel_rows = []

    for kind in kinds:
        for N in widths:
            xs = [
                d
                for d in data
                if d["kind"] == kind
                and int(d["N"]) == N
            ]

            vals = [
                float(
                    d["kernel"]["l2_relative_to_nngp"]
                )
                for d in xs
            ]

            old_metric = [
                float(
                    d["kernel"]["rms_relative_existing_metric"]
                )
                for d in xs
            ]

            kernel_rows.append({
                "kind": kind,
                "N": N,
                "mean_l2_relative_to_nngp": float(np.mean(vals)),
                "se_l2_relative_to_nngp": (
                    float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                    if len(vals) > 1
                    else float("nan")
                ),
                "mean_rms_relative_existing_metric": float(np.mean(old_metric)),
            })

    # -------------------------------------------------------------
    # Final gate
    # -------------------------------------------------------------

    cov_pass = all(
        x["pass"]
        for x in covariance_pooled
    )

    pair_pass = all(
        x["pass"]
        for x in paired_pooled
    )

    vertex_pass = all(
        x["pass"]
        for x in vertex_pooled
    )

    sextic_pass = all(
        x["pass"]
        for x in sextic_pooled
    )

    overall = (
        cov_pass
        and pair_pass
        and vertex_pass
        and sextic_pass
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        OUT / "covariance_width.csv",
        covariance_width,
    )

    write_csv(
        OUT / "covariance_pooled.csv",
        covariance_pooled,
    )

    write_csv(
        OUT / "paired_covariance_width.csv",
        paired_width,
    )

    write_csv(
        OUT / "paired_covariance_pooled.csv",
        paired_pooled,
    )

    write_csv(
        OUT / "vertex_width.csv",
        vertex_width,
    )

    write_csv(
        OUT / "vertex_pooled.csv",
        vertex_pooled,
    )

    write_csv(
        OUT / "sextic_width.csv",
        sextic_width,
    )

    write_csv(
        OUT / "sextic_pooled.csv",
        sextic_pooled,
    )

    write_csv(
        OUT / "kernel_diagnostics.csv",
        kernel_rows,
    )

    verdict = {
        "campaign": cfg["campaign"],
        "artifacts": {
            "expected": len(expected),
            "actual": len(actual),
            "missing": len(missing),
            "extra": len(extra),
        },
        "gates": {
            "same_nngp_covariance": cov_pass,
            "paired_amplitude_covariance": pair_pass,
            "spatial_connected_quartic": vertex_pass,
            "local_sextic": sextic_pass,
        },
        "aggregate_pass": overall,
        "counts": {
            "covariance_gates": [
                sum(x["pass"] for x in covariance_pooled),
                len(covariance_pooled),
            ],
            "paired_covariance_gates": [
                sum(x["pass"] for x in paired_pooled),
                len(paired_pooled),
            ],
            "vertex_gates": [
                sum(x["pass"] for x in vertex_pooled),
                len(vertex_pooled),
            ],
            "sextic_gates": [
                sum(x["pass"] for x in sextic_pooled),
                len(sextic_pooled),
            ],
        },
        "interpretation": (
            "quartic_null is a full-quartic cancellation only if "
            "the separated spatial targets themselves vanish; "
            "otherwise it is a local/contact cancellation."
        ),
    }

    (
        OUT
        / "verdict.json"
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
    print("P1c ANALYSIS")
    print("========================================")
    print(json.dumps(verdict, indent=2))

    print()
    print("SPATIAL VERTEX POOLED:")
    for x in vertex_pooled:
        print(
            f"{x['kind']:13s} "
            f"r={x['separation']:2d} "
            f"N*k22={x['mean']:+.8f} "
            f"+/- {x['se']:.8f} "
            f"target={x['target']:+.8f} "
            f"z={x['abs_z']:.2f} "
            f"chi2/dof={x['chi2_dof']:.2f} "
            f"PASS={x['pass']}"
        )

    print()
    print("LOCAL SEXTIC:")
    for x in sextic_pooled:
        print(
            f"{x['kind']:13s} "
            f"N^2*k6={x['mean']:+.8f} "
            f"+/- {x['se']:.8f} "
            f"target={x['target']:+.8f} "
            f"z={x['abs_z']:.2f} "
            f"PASS={x['pass']}"
        )


if __name__ == "__main__":
    main()
