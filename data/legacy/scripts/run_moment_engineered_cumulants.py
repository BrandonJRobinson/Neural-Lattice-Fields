#!/usr/bin/env python3

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np


AMP_MOMENTS = {
    "fixed": (1.0, 1.0, 1.0),
    "gaussian": (1.0, 3.0, 15.0),
    "quartic_null": (1.0, 2.0, 4.0),
}


def draw_c(rng, kind, shape):
    if kind == "fixed":
        # Fixed amplitude. Uniform phase already makes the neuron centered.
        return None

    if kind == "gaussian":
        return rng.normal(size=shape)

    if kind == "quartic_null":
        # c = 0 with probability 1/2,
        # c = +/- sqrt(2) with probability 1/4 each.
        #
        # E[c^2] = 1
        # E[c^4] = 2  => kappa_4 = 0
        # E[c^6] = 4  => kappa_6 != 0
        u = rng.random(size=shape)

        c = np.zeros(shape, dtype=np.float64)
        nz = u >= 0.5

        signs = np.where(
            rng.random(size=shape) < 0.5,
            -1.0,
            1.0,
        )
        c[nz] = math.sqrt(2.0) * signs[nz]
        return c

    raise ValueError(f"Unknown amplitude law: {kind}")


def cumulants_from_sums(n, s2, s4, s6):
    m2 = s2 / n
    m4 = s4 / n
    m6 = s6 / n

    k4 = m4 - 3.0 * m2 * m2

    k6 = (
        m6
        - 15.0 * m4 * m2
        + 30.0 * m2**3
    )

    return m2, m4, m6, k4, k6


def jackknife(blocks, field_index):
    """
    Delete-one-block jackknife.

    field_index:
        0 -> variance
        1 -> kappa_4
        2 -> kappa_6
    """
    b = len(blocks)

    if b < 3:
        return float("nan")

    arr = np.asarray(blocks, dtype=np.float64)
    totals = np.sum(arr, axis=0)

    values = []

    for row in arr:
        rem = totals - row

        n, s1, s2, s4, s6 = rem

        m2, m4, m6, k4, k6 = cumulants_from_sums(
            n,
            s2,
            s4,
            s6,
        )

        values.append(
            (m2, k4, k6)[field_index]
        )

    values = np.asarray(values, dtype=np.float64)
    mean = np.mean(values)

    return math.sqrt(
        (b - 1.0) / b
        * np.sum((values - mean) ** 2)
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--kind",
        choices=AMP_MOMENTS,
        required=True,
    )

    ap.add_argument(
        "--width",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--samples",
        type=int,
        default=4_000_000,
    )

    ap.add_argument(
        "--block-size",
        type=int,
        default=10_000,
    )

    ap.add_argument(
        "--sigma-c",
        type=float,
        default=0.7127,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    N = args.width
    sigma = args.sigma_c

    blocks = []

    # columns:
    # n, sum(phi), sum(phi^2), sum(phi^4), sum(phi^6)
    total = np.zeros(5, dtype=np.float64)

    t0 = time.time()

    remaining = args.samples

    while remaining > 0:
        m = min(args.block_size, remaining)

        # At a coincident point:
        #
        # z_a = sigma_c c_a cos(B_a)
        #
        # with B uniform on [0,2pi).
        #
        # The spectral momentum k_a cancels from the marginal exactly.
        B = rng.uniform(
            0.0,
            2.0 * np.pi,
            size=(m, N),
        )

        y = np.cos(B)

        c = draw_c(
            rng,
            args.kind,
            (m, N),
        )

        if c is not None:
            y *= c

        phi = (
            sigma
            * np.sum(y, axis=1)
            / math.sqrt(N)
        )

        p2 = phi * phi

        row = np.array(
            [
                m,
                np.sum(phi, dtype=np.float64),
                np.sum(p2, dtype=np.float64),
                np.sum(p2 * p2, dtype=np.float64),
                np.sum(p2 * p2 * p2, dtype=np.float64),
            ],
            dtype=np.float64,
        )

        blocks.append(row)
        total += row

        remaining -= m

    n, s1, s2, s4, s6 = total

    m2, m4, m6, k4, k6 = cumulants_from_sums(
        n,
        s2,
        s4,
        s6,
    )

    m2_se = jackknife(blocks, 0)
    k4_se = jackknife(blocks, 1)
    k6_se = jackknife(blocks, 2)

    c2, c4, c6 = AMP_MOMENTS[args.kind]

    target_variance = sigma**2 / 2.0

    target_k4 = (
        (3.0 * sigma**4 / 8.0)
        * (c4 - 2.0)
        / N
    )

    target_k6 = (
        (5.0 * sigma**6 / 16.0)
        * (c6 - 9.0 * c4 + 12.0)
        / (N * N)
    )

    out = {
        "campaign": "moment_engineering_v1",
        "kind": args.kind,
        "width": N,
        "seed": args.seed,
        "samples": args.samples,
        "block_size": args.block_size,
        "sigma_c": sigma,
        "sample_mean": float(s1 / n),

        "variance": float(m2),
        "variance_se": float(m2_se),

        "kappa4": float(k4),
        "kappa4_se": float(k4_se),

        "kappa6": float(k6),
        "kappa6_se": float(k6_se),

        "N_kappa4": float(N * k4),
        "N_kappa4_se": float(N * k4_se),

        "N2_kappa6": float(N * N * k6),
        "N2_kappa6_se": float(N * N * k6_se),

        "target_variance": float(target_variance),
        "target_N_kappa4": float(N * target_k4),
        "target_N2_kappa6": float(N * N * target_k6),

        "runtime_sec": float(time.time() - t0),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tmp = output.with_suffix(
        output.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            out,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    os.replace(tmp, output)

    print(
        json.dumps(
            out,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
