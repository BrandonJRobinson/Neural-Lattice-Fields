#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def combine(values, errors):
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    good = (
        np.isfinite(values)
        & np.isfinite(errors)
        & (errors > 0)
    )

    values = values[good]
    errors = errors[good]

    if len(values) == 0:
        return (float("nan"),) * 5

    weights = 1.0 / (errors * errors)

    mean = float(
        np.sum(weights * values)
        / np.sum(weights)
    )

    within = float(
        1.0 / math.sqrt(np.sum(weights))
    )

    if len(values) > 1:
        between = float(
            np.std(values, ddof=1)
            / math.sqrt(len(values))
        )

        chi2dof = float(
            np.sum(
                ((values - mean) / errors) ** 2
            )
            / (len(values) - 1)
        )
    else:
        between = float("nan")
        chi2dof = float("nan")

    conservative = (
        max(within, between)
        if np.isfinite(between)
        else within
    )

    return (
        mean,
        within,
        between,
        conservative,
        chi2dof,
    )


def plateau(rows, key, error_key, widths=None):
    selected = [
        row
        for row in rows
        if widths is None
        or row["width"] in widths
    ]

    return combine(
        [row[key] for row in selected],
        [row[error_key] for row in selected],
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input-dir",
        required=True,
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    args = ap.parse_args()

    files = sorted(
        glob.glob(
            str(
                Path(args.input_dir)
                / "*.json"
            )
        )
    )

    if not files:
        raise SystemExit(
            "No production JSON files found"
        )

    data = []

    for filename in files:
        with open(filename) as handle:
            data.append(json.load(handle))

    groups = defaultdict(list)

    for row in data:
        groups[
            (
                row["kind"],
                int(row["width"]),
            )
        ].append(row)

    summary_rows = []

    for (kind, width), rows in sorted(groups.items()):
        variance = combine(
            [r["variance"] for r in rows],
            [r["variance_se"] for r in rows],
        )

        k4 = combine(
            [r["N_kappa4"] for r in rows],
            [r["N_kappa4_se"] for r in rows],
        )

        k6 = combine(
            [r["N2_kappa6"] for r in rows],
            [r["N2_kappa6_se"] for r in rows],
        )

        summary_rows.append(
            {
                "kind": kind,
                "width": width,
                "n_replicas": len(rows),

                "variance": variance[0],
                "variance_err": variance[3],
                "variance_chi2dof": variance[4],

                "N_kappa4": k4[0],
                "N_kappa4_err": k4[3],
                "N_kappa4_chi2dof": k4[4],

                "N2_kappa6": k6[0],
                "N2_kappa6_err": k6[3],
                "N2_kappa6_chi2dof": k6[4],

                "target_variance":
                    rows[0]["target_variance"],

                "target_N_kappa4":
                    rows[0]["target_N_kappa4"],

                "target_N2_kappa6":
                    rows[0]["target_N2_kappa6"],
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        summary_rows[0].keys()
    )

    with open(
        output_dir / "summary.csv",
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("MOMENT ENGINEERING V1")
    print()

    print(
        f"{'kind':14s} "
        f"{'N':>4s} "
        f"{'N*k4':>20s} "
        f"{'target':>12s} "
        f"{'N^2*k6':>20s} "
        f"{'target':>12s}"
    )

    for row in summary_rows:
        print(
            f"{row['kind']:14s} "
            f"{row['width']:4d} "
            f"{row['N_kappa4']:+.6f} "
            f"+/- {row['N_kappa4_err']:.6f} "
            f"{row['target_N_kappa4']:+.6f} "
            f"{row['N2_kappa6']:+.6f} "
            f"+/- {row['N2_kappa6_err']:.6f} "
            f"{row['target_N2_kappa6']:+.6f}"
        )

    verdict = {}

    kinds = sorted(
        {
            row["kind"]
            for row in summary_rows
        }
    )

    for kind in kinds:
        rows = [
            row
            for row in summary_rows
            if row["kind"] == kind
        ]

        quartic = plateau(
            rows,
            "N_kappa4",
            "N_kappa4_err",
        )

        # Pre-registered primary sextic window.
        # N=32 remains a stress-test point.
        sextic = plateau(
            rows,
            "N2_kappa6",
            "N2_kappa6_err",
            widths={2, 4, 8, 16},
        )

        target4 = rows[0][
            "target_N_kappa4"
        ]

        target6 = rows[0][
            "target_N2_kappa6"
        ]

        z4 = (
            (quartic[0] - target4)
            / quartic[3]
            if quartic[3] > 0
            else float("nan")
        )

        z6 = (
            (sextic[0] - target6)
            / sextic[3]
            if sextic[3] > 0
            else float("nan")
        )

        max_var_rel = max(
            abs(
                row["variance"]
                - row["target_variance"]
            )
            / row["target_variance"]
            for row in rows
        )

        verdict[kind] = {
            "quartic_plateau":
                quartic[0],

            "quartic_err":
                quartic[3],

            "quartic_target":
                target4,

            "quartic_z":
                z4,

            "sextic_plateau_Nle16":
                sextic[0],

            "sextic_err":
                sextic[3],

            "sextic_target":
                target6,

            "sextic_z":
                z6,

            "max_variance_relative_error":
                max_var_rel,

            "pass_quartic":
                bool(abs(z4) <= 3.0),

            "pass_sextic":
                bool(abs(z6) <= 3.0),

            "pass_variance":
                bool(max_var_rel <= 0.01),
        }

    with open(
        output_dir / "verdict.json",
        "w",
    ) as handle:
        json.dump(
            verdict,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print()
    print("PREREGISTERED AGGREGATE TESTS")

    for kind, result in verdict.items():
        print(
            kind,
            json.dumps(
                result,
                sort_keys=True,
            ),
        )

    expected = 120
    missing = expected - len(data)

    print()
    print(
        f"Raw files: {len(data)}/{expected}; "
        f"missing={missing}"
    )

    if missing != 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
