#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from p5_fermionic_common import (
    LatentFermionAuxiliary,
    positive_sequence_tau
)


ROOT = Path(__file__).resolve().parents[1]

CFG = json.loads(
    (
        ROOT
        / "configs"
        / "p5_fermionic_full_measure_v1.json"
    ).read_text()
)


def leapfrog(
    theta,
    momentum,
    *,
    model,
    eps,
    nsteps
):
    q = theta.copy()
    p = momentum.copy()

    S,g = model.action_grad(
        q
    )

    if (
        not np.isfinite(
            S
        )
        or not np.all(
            np.isfinite(
                g
            )
        )
    ):
        return (
            theta,
            momentum,
            np.inf,
            False
        )

    p -= (
        0.5*eps*g
    )

    for step in range(
        nsteps
    ):
        q += (
            eps*p
        )

        S,g = model.action_grad(
            q
        )

        if (
            not np.isfinite(
                S
            )
            or not np.all(
                np.isfinite(
                    q
                )
            )
            or not np.all(
                np.isfinite(
                    g
                )
            )
        ):
            return (
                theta,
                momentum,
                np.inf,
                False
            )

        if step != nsteps-1:
            p -= (
                eps*g
            )

    p -= (
        0.5*eps*g
    )

    p = -p

    return (
        q,
        p,
        S,
        True
    )


def hmc_step(
    theta,
    *,
    model,
    eps,
    trajectory,
    rng
):
    p0 = rng.standard_normal(
        theta.shape
    )

    S0 = model.action(
        theta
    )

    H0 = float(
        S0
        + 0.5*np.sum(
            p0*p0
        )
    )

    nsteps = max(
        2,
        int(
            math.ceil(
                trajectory/eps
            )
        )
    )

    actual_eps = (
        trajectory/nsteps
    )

    q1,p1,S1,finite = leapfrog(
        theta,
        p0,
        model=model,
        eps=actual_eps,
        nsteps=nsteps
    )

    if not finite:
        return (
            theta,
            False,
            np.inf,
            nsteps,
            actual_eps,
            True
        )

    H1 = float(
        S1
        + 0.5*np.sum(
            p1*p1
        )
    )

    dH = float(
        H1-H0
    )

    if not np.isfinite(
        dH
    ):
        return (
            theta,
            False,
            np.inf,
            nsteps,
            actual_eps,
            True
        )

    if (
        dH <= 0
        or math.log(
            rng.random()
        ) < -dH
    ):
        return (
            q1,
            True,
            dH,
            nsteps,
            actual_eps,
            False
        )

    return (
        theta,
        False,
        dH,
        nsteps,
        actual_eps,
        False
    )


def global_shift(
    theta,
    *,
    model,
    std,
    rng
):
    delta = float(
        std
        * rng.standard_normal()
    )

    dsigma = np.full(
        (
            model.L,
            model.L
        ),
        delta,
        dtype=float
    )

    dtheta = model.W.sqrt_apply(
        dsigma
    )

    candidate = (
        theta+dtheta
    )

    Sold = model.action(
        theta
    )

    Snew = model.action(
        candidate
    )

    if not np.isfinite(
        Snew
    ):
        return (
            theta,
            False
        )

    dS = float(
        Snew-Sold
    )

    if (
        dS <= 0
        or math.log(
            rng.random()
        ) < -dS
    ):
        return (
            candidate,
            True
        )

    return (
        theta,
        False
    )


def derived_from_blocks(
    blocks,
    V
):
    b = np.asarray(
        blocks,
        dtype=float
    )

    mean_sigma = float(
        np.mean(
            b[
                :,
                0
            ]
        )
    )

    sigma2 = float(
        np.mean(
            b[
                :,
                1
            ]
        )
    )

    bar2 = float(
        np.mean(
            b[
                :,
                2
            ]
        )
    )

    condensate = float(
        np.mean(
            b[
                :,
                3
            ]
        )
    )

    return {
        "sigma_mean":mean_sigma,

        "sigma2":sigma2,

        "chi_sigma":float(
            V
            * (
                bar2
                - mean_sigma*mean_sigma
            )
        ),

        "condensate":condensate
    }


def jackknife_blocks(
    blocks,
    V
):
    B = np.asarray(
        blocks,
        dtype=float
    )

    n = B.shape[
        0
    ]

    full = derived_from_blocks(
        B,
        V
    )

    total = np.sum(
        B,
        axis=0
    )

    vals = {
        k:[]
        for k in full
    }

    for i in range(
        n
    ):
        leave = (
            total-B[
                i
            ]
        )/(n-1)

        d = derived_from_blocks(
            leave[
                None,
                :
            ],
            V
        )

        for k in vals:
            vals[
                k
            ].append(
                d[
                    k
                ]
            )

    out = {}

    for k in full:
        arr = np.asarray(
            vals[
                k
            ],
            float
        )

        m = float(
            np.mean(
                arr
            )
        )

        se = float(
            np.sqrt(
                (n-1)/n
                * np.sum(
                    (
                        arr-m
                    )**2
                )
            )
        )

        out[
            k
        ] = {
            "value":float(
                full[
                    k
                ]
            ),

            "jk_se":se
        }

    return out


def run(
    g,
    seed
):
    L = int(
        CFG[
            "L"
        ]
    )

    V = L*L

    model = LatentFermionAuxiliary(
        L=L,

        Nf=int(
            CFG[
                "fermion"
            ][
                "Nf"
            ]
        ),

        mf=float(
            CFG[
                "fermion"
            ][
                "mf"
            ]
        ),

        r=float(
            CFG[
                "fermion"
            ][
                "wilson_r"
            ]
        ),

        msigma_sq=float(
            CFG[
                "auxiliary"
            ][
                "msigma_sq"
            ]
        ),

        g=float(
            g
        )
    )

    hc = CFG[
        "hmc"
    ]

    warmup = int(
        hc[
            "warmup"
        ]
    )

    production = int(
        hc[
            "production"
        ]
    )

    nblocks = int(
        CFG[
            "jackknife_blocks"
        ]
    )

    if production % nblocks != 0:
        raise RuntimeError(
            "production must divide jackknife blocks"
        )

    block_size = (
        production//nblocks
    )

    eff_seed = (
        int(
            seed
        )
        + int(
            round(
                100000
                * float(
                    g
                )
            )
        )
        + 1000000*L
    )

    rng = np.random.default_rng(
        eff_seed
    )

    theta = rng.standard_normal(
        (
            L,
            L
        )
    )

    eps = float(
        hc[
            "initial_step_size"
        ]
    )

    trajectory = float(
        hc[
            "trajectory_length"
        ]
    )

    shift_std = float(
        hc[
            "global_shift_std"
        ]
    )

    recent_acc = 0
    recent_n = 0

    warm_acc = 0
    warm_shift_acc = 0
    warm_nf = 0

    # =============================================================
    # Warmup
    # =============================================================

    for sweep in range(
        warmup
    ):
        (
            theta,
            accepted,
            dH,
            nsteps,
            actual_eps,
            nonfinite
        ) = hmc_step(
            theta,
            model=model,
            eps=eps,
            trajectory=trajectory,
            rng=rng
        )

        warm_acc += int(
            accepted
        )

        warm_nf += int(
            nonfinite
        )

        recent_acc += int(
            accepted
        )

        recent_n += 1

        (
            theta,
            shacc
        ) = global_shift(
            theta,
            model=model,
            std=shift_std,
            rng=rng
        )

        warm_shift_acc += int(
            shacc
        )

        if (
            (
                sweep+1
            )
            % int(
                hc[
                    "adapt_interval"
                ]
            )
            == 0
        ):
            rate = (
                recent_acc
                / max(
                    recent_n,
                    1
                )
            )

            if (
                rate
                > float(
                    hc[
                        "acceptance_target_high"
                    ]
                )
            ):
                eps *= float(
                    hc[
                        "adapt_up"
                    ]
                )

            elif (
                rate
                < float(
                    hc[
                        "acceptance_target_low"
                    ]
                )
            ):
                eps *= float(
                    hc[
                        "adapt_down"
                    ]
                )

            eps = float(
                np.clip(
                    eps,
                    float(
                        hc[
                            "step_size_min"
                        ]
                    ),
                    float(
                        hc[
                            "step_size_max"
                        ]
                    )
                )
            )

            recent_acc = 0
            recent_n = 0

    # =============================================================
    # Production
    # =============================================================

    blocks = np.zeros(
        (
            nblocks,
            4
        ),
        dtype=float
    )

    series_mean = np.empty(
        production,
        dtype=float
    )

    series_sigma2 = np.empty(
        production,
        dtype=float
    )

    series_cond = np.empty(
        production,
        dtype=float
    )

    accepted_count = 0
    shift_accepted = 0
    nonfinite_count = 0

    total_steps = 0
    total_grad = 0

    actual_eps_values = []

    t0 = time.perf_counter()

    for sweep in range(
        production
    ):
        (
            theta,
            accepted,
            dH,
            nsteps,
            actual_eps,
            nonfinite
        ) = hmc_step(
            theta,
            model=model,
            eps=eps,
            trajectory=trajectory,
            rng=rng
        )

        accepted_count += int(
            accepted
        )

        nonfinite_count += int(
            nonfinite
        )

        total_steps += int(
            nsteps
        )

        total_grad += int(
            nsteps+1
        )

        actual_eps_values.append(
            actual_eps
        )

        (
            theta,
            shacc
        ) = global_shift(
            theta,
            model=model,
            std=shift_std,
            rng=rng
        )

        shift_accepted += int(
            shacc
        )

        obs = model.observables(
            theta
        )

        series_mean[
            sweep
        ] = obs[
            "sigma_mean"
        ]

        series_sigma2[
            sweep
        ] = obs[
            "sigma2"
        ]

        series_cond[
            sweep
        ] = obs[
            "condensate"
        ]

        b = (
            sweep//block_size
        )

        blocks[
            b,
            0
        ] += obs[
            "sigma_mean"
        ]

        blocks[
            b,
            1
        ] += obs[
            "sigma2"
        ]

        blocks[
            b,
            2
        ] += obs[
            "sigma_bar2"
        ]

        blocks[
            b,
            3
        ] += obs[
            "condensate"
        ]

    elapsed = (
        time.perf_counter()
        - t0
    )

    blocks /= block_size

    estimates = jackknife_blocks(
        blocks,
        V
    )

    tau_mean = positive_sequence_tau(
        series_mean
    )

    tau_sigma2 = positive_sequence_tau(
        series_sigma2
    )

    tau_cond = positive_sequence_tau(
        series_cond
    )

    tau_worst = max(
        tau_mean,
        tau_sigma2,
        tau_cond
    )

    Neff = (
        production
        / (
            2.0*tau_worst
        )
    )

    return {
        "campaign":CFG[
            "campaign"
        ],

        "L":L,

        "V":V,

        "g":float(
            g
        ),

        "seed":int(
            seed
        ),

        "effective_seed":int(
            eff_seed
        ),

        "estimates":estimates,

        "diagnostics":{
            "acceptance":float(
                accepted_count
                / production
            ),

            "global_shift_acceptance":float(
                shift_accepted
                / production
            ),

            "warmup_acceptance":float(
                warm_acc
                / warmup
            ),

            "warmup_shift_acceptance":float(
                warm_shift_acc
                / warmup
            ),

            "tau_sigma_mean":float(
                tau_mean
            ),

            "tau_sigma2":float(
                tau_sigma2
            ),

            "tau_condensate":float(
                tau_cond
            ),

            "tau_worst":float(
                tau_worst
            ),

            "Neff_worst":float(
                Neff
            ),

            "final_nominal_step_size":float(
                eps
            ),

            "mean_actual_step_size":float(
                np.mean(
                    actual_eps_values
                )
            ),

            "mean_leapfrog_steps":float(
                total_steps
                / production
            ),

            "ESS_per_1000_grad":float(
                1000.0
                * Neff
                / max(
                    total_grad,
                    1
                )
            ),

            "ESS_per_sec":float(
                Neff
                / max(
                    elapsed,
                    1e-12
                )
            ),

            "nonfinite_warmup":int(
                warm_nf
            ),

            "nonfinite_production":int(
                nonfinite_count
            ),

            "production_seconds":float(
                elapsed
            )
        }
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--g",
        required=True,
        type=float
    )

    ap.add_argument(
        "--seed",
        required=True,
        type=int
    )

    ap.add_argument(
        "--output",
        required=True,
        type=Path
    )

    args = ap.parse_args()

    out = run(
        float(
            args.g
        ),
        int(
            args.seed
        )
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    args.output.write_text(
        json.dumps(
            out,
            indent=2,
            sort_keys=True
        )
        + "\n"
    )

    print(
        f"g={out['g']:.3f} "
        f"seed={out['seed']} "
        f"acc={out['diagnostics']['acceptance']:.3f} "
        f"shift={out['diagnostics']['global_shift_acceptance']:.3f} "
        f"Neff={out['diagnostics']['Neff_worst']:.1f} "
        f"sigma2={out['estimates']['sigma2']['value']:.8f} "
        f"chi={out['estimates']['chi_sigma']['value']:.8f} "
        f"Cpsi={out['estimates']['condensate']['value']:.8f}",
        flush=True
    )


if __name__ == "__main__":
    main()
