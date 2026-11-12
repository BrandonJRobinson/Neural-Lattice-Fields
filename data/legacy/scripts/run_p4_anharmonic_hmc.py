#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from p4_anharmonic_common import (
    LatentOscillator,
    circular_correlator,
    jackknife_gap,
    jackknife_scalar,
    positive_sequence_tau
)


ROOT = Path(__file__).resolve().parents[1]

CFG = json.loads(
    (
        ROOT
        / "configs"
        / "p4_anharmonic_oscillator_v1.json"
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

    for j in range(
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

        if j != nsteps-1:
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
    trajectory_length,
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
                trajectory_length/eps
            )
        )
    )

    actual_eps = (
        trajectory_length/nsteps
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


def run(
    Nt,
    seed
):
    h = CFG[
        "hamiltonian"
    ]

    e = CFG[
        "euclidean"
    ]

    l = CFG[
        "latent"
    ]

    hc = CFG[
        "hmc"
    ]

    beta = float(
        e[
            "beta"
        ]
    )

    a = (
        beta/Nt
    )

    model = LatentOscillator(
        Nt=Nt,
        beta=beta,
        omega=float(
            h[
                "omega"
            ]
        ),
        g=float(
            h[
                "g"
            ]
        ),
        omega_ref=float(
            l[
                "omega_ref"
            ]
        )
    )

    eff_seed = (
        int(seed)
        + 1000000*Nt
    )

    rng = np.random.default_rng(
        eff_seed
    )

    theta = rng.standard_normal(
        Nt
    )

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

    trajectory = float(
        hc[
            "trajectory_length"
        ]
    )

    eps = float(
        hc[
            "initial_step_size"
        ]
    )

    adapt_interval = int(
        hc[
            "adapt_interval"
        ]
    )

    recent_acc = 0
    recent_n = 0

    warm_acc = 0
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
            trajectory_length=trajectory,
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

        if (
            rng.random()
            < float(
                hc[
                    "global_z2_probability"
                ]
            )
        ):
            theta = -theta

        if (
            (sweep+1)
            % adapt_interval
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

    nblocks = int(
        CFG[
            "jackknife_blocks"
        ]
    )

    if production % nblocks != 0:
        raise RuntimeError(
            "production not divisible by jackknife blocks"
        )

    block_size = (
        production//nblocks
    )

    corr_stride = int(
        hc[
            "correlator_stride"
        ]
    )

    if block_size % corr_stride != 0:
        raise RuntimeError(
            "JK block size not divisible by correlator stride"
        )

    block_x2 = np.zeros(
        nblocks,
        float
    )

    block_x4 = np.zeros(
        nblocks,
        float
    )

    block_C = np.zeros(
        (
            nblocks,
            Nt
        ),
        float
    )

    block_C_count = np.zeros(
        nblocks,
        int
    )

    series_x2 = np.empty(
        production,
        float
    )

    series_x4 = np.empty(
        production,
        float
    )

    accepted_count = 0
    nonfinite_count = 0

    total_grad = 0
    total_steps = 0

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
            trajectory_length=trajectory,
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

        if (
            rng.random()
            < float(
                hc[
                    "global_z2_probability"
                ]
            )
        ):
            theta = -theta

        x = model.field(
            theta
        )

        x2 = float(
            np.mean(
                x*x
            )
        )

        x4 = float(
            np.mean(
                x**4
            )
        )

        series_x2[
            sweep
        ] = x2

        series_x4[
            sweep
        ] = x4

        b = (
            sweep//block_size
        )

        block_x2[
            b
        ] += x2

        block_x4[
            b
        ] += x4

        if (
            sweep
            % corr_stride
            == 0
        ):
            block_C[
                b
            ] += circular_correlator(
                x
            )

            block_C_count[
                b
            ] += 1

    elapsed = (
        time.perf_counter()
        - t0
    )

    block_x2 /= block_size
    block_x4 /= block_size

    for b in range(
        nblocks
    ):
        if block_C_count[
            b
        ] <= 0:
            raise RuntimeError(
                "empty correlator block"
            )

        block_C[
            b
        ] /= block_C_count[
            b
        ]

    estimates = {
        "x2": jackknife_scalar(
            block_x2
        ),

        "x4": jackknife_scalar(
            block_x4
        ),

        "gap_fit": jackknife_gap(
            block_C,
            a=a,
            beta=beta,
            gap_cfg=CFG[
                "gap_fit"
            ]
        )
    }

    tau_x2 = positive_sequence_tau(
        series_x2
    )

    tau_x4 = positive_sequence_tau(
        series_x4
    )

    tau_worst = max(
        tau_x2,
        tau_x4
    )

    Neff = (
        production
        / (
            2.0*tau_worst
        )
    )

    return {
        "campaign": CFG[
            "campaign"
        ],

        "Nt": int(
            Nt
        ),

        "beta": beta,

        "a": float(
            a
        ),

        "seed": int(
            seed
        ),

        "effective_seed": int(
            eff_seed
        ),

        "estimates": estimates,

        "diagnostics": {
            "acceptance": float(
                accepted_count/production
            ),

            "warmup_acceptance": float(
                warm_acc/warmup
            ),

            "tau_x2": float(
                tau_x2
            ),

            "tau_x4": float(
                tau_x4
            ),

            "tau_worst": float(
                tau_worst
            ),

            "Neff_worst": float(
                Neff
            ),

            "final_nominal_step_size": float(
                eps
            ),

            "mean_actual_step_size": float(
                np.mean(
                    actual_eps_values
                )
            ),

            "mean_leapfrog_steps": float(
                total_steps/production
            ),

            "total_grad_evals": int(
                total_grad
            ),

            "ESS_per_1000_grad": float(
                1000.0*Neff
                / total_grad
            ),

            "ESS_per_sec": float(
                Neff/max(
                    elapsed,
                    1e-12
                )
            ),

            "production_seconds": float(
                elapsed
            ),

            "nonfinite_warmup": int(
                warm_nf
            ),

            "nonfinite_production": int(
                nonfinite_count
            )
        }
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--Nt",
        required=True,
        type=int
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
        int(
            args.Nt
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
        f"Nt={out['Nt']} "
        f"a={out['a']:.6f} "
        f"seed={out['seed']} "
        f"acc={out['diagnostics']['acceptance']:.3f} "
        f"Neff={out['diagnostics']['Neff_worst']:.1f} "
        f"x2={out['estimates']['x2']['value']:.8f} "
        f"x4={out['estimates']['x4']['value']:.8f} "
        f"gap={out['estimates']['gap_fit']['value']:.8f}",
        flush=True
    )


if __name__ == "__main__":
    main()
