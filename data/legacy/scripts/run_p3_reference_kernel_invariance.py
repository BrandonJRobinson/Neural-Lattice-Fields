#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CFG = json.loads(
    (
        ROOT
        / "configs"
        / "p3_reference_kernel_invariance_v1.json"
    ).read_text()
)


def phat_sq(L):
    p = (
        2.0
        * np.pi
        * np.fft.fftfreq(L)
    )

    out = np.zeros(
        (L,L),
        dtype=float
    )

    for mu in range(2):
        if mu == 0:
            out += (
                4.0
                * np.sin(
                    p[:,None]/2.0
                )**2
            )
        else:
            out += (
                4.0
                * np.sin(
                    p[None,:]/2.0
                )**2
            )

    return out


class Whitening:
    def __init__(
        self,
        L,
        mref
    ):
        self.L = int(L)

        self.K = (
            phat_sq(L)
            + float(mref)
        )

        if np.min(self.K) <= 0:
            raise ValueError(
                "reference kernel must be positive"
            )

        self.inv_sqrt = (
            1.0
            / np.sqrt(
                self.K
            )
        )

        self.sqrt = np.sqrt(
            self.K
        )

    def apply_inv_sqrt(
        self,
        theta
    ):
        ft = np.fft.fftn(
            theta,
            norm="ortho"
        )

        phi = np.fft.ifftn(
            ft
            * self.inv_sqrt,
            norm="ortho"
        ).real

        return phi

    def apply_inv_sqrt_T(
        self,
        x
    ):
        # Symmetric operator.
        return self.apply_inv_sqrt(
            x
        )

    def apply_sqrt(
        self,
        phi
    ):
        ft = np.fft.fftn(
            phi,
            norm="ortho"
        )

        theta = np.fft.ifftn(
            ft
            * self.sqrt,
            norm="ortho"
        ).real

        return theta


class LatentPhi4:
    def __init__(
        self,
        L,
        m0_sq,
        lam,
        mref
    ):
        self.L = int(L)
        self.V = int(L*L)

        self.m0_sq = float(
            m0_sq
        )

        self.lam = float(
            lam
        )

        self.mref = float(
            mref
        )

        self.dm = (
            self.m0_sq
            - self.mref
        )

        self.W = Whitening(
            L,
            mref
        )

    def field(
        self,
        theta
    ):
        return self.W.apply_inv_sqrt(
            theta
        )

    def action(
        self,
        theta
    ):
        phi = self.field(
            theta
        )

        val = (
            0.5
            * np.sum(
                theta*theta
            )
            + 0.5
            * self.dm
            * np.sum(
                phi*phi
            )
            + self.lam
            * np.sum(
                phi**4
            )
        )

        return float(
            val
        )

    def action_grad(
        self,
        theta
    ):
        phi = self.field(
            theta
        )

        val = (
            0.5
            * np.sum(
                theta*theta
            )
            + 0.5
            * self.dm
            * np.sum(
                phi*phi
            )
            + self.lam
            * np.sum(
                phi**4
            )
        )

        dphi = (
            self.dm
            * phi
            + 4.0
            * self.lam
            * phi**3
        )

        grad = (
            theta
            + self.W.apply_inv_sqrt_T(
                dphi
            )
        )

        return (
            float(
                val
            ),
            grad
        )


def physical_action(
    phi,
    *,
    m0_sq,
    lam
):
    kinetic = 0.0

    for axis in (
        0,
        1
    ):
        d = (
            np.roll(
                phi,
                -1,
                axis=axis
            )
            - phi
        )

        kinetic += (
            0.5
            * np.sum(
                d*d
            )
        )

    return float(
        kinetic
        + 0.5
        * m0_sq
        * np.sum(
            phi*phi
        )
        + lam
        * np.sum(
            phi**4
        )
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

    S, grad = model.action_grad(
        q
    )

    if (
        not np.isfinite(S)
        or not np.all(
            np.isfinite(
                grad
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
        0.5
        * eps
        * grad
    )

    for step in range(
        nsteps
    ):
        q += (
            eps*p
        )

        S, grad = model.action_grad(
            q
        )

        if (
            not np.isfinite(S)
            or not np.all(
                np.isfinite(
                    grad
                )
            )
            or not np.all(
                np.isfinite(
                    q
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
                eps*grad
            )

    p -= (
        0.5
        * eps
        * grad
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

    H0 = (
        S0
        + 0.5
        * np.sum(
            p0*p0
        )
    )

    nsteps = max(
        2,
        int(
            math.ceil(
                trajectory_length
                / eps
            )
        )
    )

    # Keep the trajectory length approximately fixed.
    actual_eps = (
        trajectory_length
        / nsteps
    )

    (
        q1,
        p1,
        S1,
        finite
    ) = leapfrog(
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

    H1 = (
        S1
        + 0.5
        * np.sum(
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
        )
        < -dH
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


def primitive_row(
    phi
):
    L = int(
        phi.shape[
            0
        ]
    )

    V = (
        L*L
    )

    M = float(
        np.mean(
            phi
        )
    )

    M2 = M*M
    M4 = M2*M2

    phi2 = float(
        np.mean(
            phi*phi
        )
    )

    F = np.fft.fftn(
        phi
    )

    S0 = float(
        np.abs(
            F[
                0,
                0
            ]
        )**2
        / V
    )

    Sx = float(
        np.abs(
            F[
                1,
                0
            ]
        )**2
        / V
    )

    Sy = float(
        np.abs(
            F[
                0,
                1
            ]
        )**2
        / V
    )

    Spmin = (
        0.5
        * (
            Sx+Sy
        )
    )

    return np.asarray(
        [
            phi2,
            abs(M),
            M2,
            M4,
            S0,
            Spmin
        ],
        dtype=float
    )


def derived(
    sums,
    n,
    L
):
    (
        phi2,
        absM,
        M2,
        M4,
        S0,
        Spmin
    ) = (
        sums/n
    )

    V = (
        L*L
    )

    chi_abs = (
        V
        * (
            M2
            - absM*absM
        )
    )

    U4 = (
        1.0
        - M4
        / (
            3.0
            * M2*M2
        )
    )

    ratio = (
        S0
        / Spmin
        - 1.0
    )

    xi = (
        1.0
        / (
            2.0
            * np.sin(
                np.pi/L
            )
        )
        * np.sqrt(
            max(
                ratio,
                0.0
            )
        )
    )

    return {
        "phi2": float(
            phi2
        ),

        "abs_M": float(
            absM
        ),

        "chi_abs": float(
            chi_abs
        ),

        "binder_U4": float(
            U4
        ),

        "xi_over_L": float(
            xi/L
        ),

        "S_pmin": float(
            Spmin
        ),

        "chi0": float(
            V*M2
        )
    }


def block_jackknife(
    rows,
    *,
    L,
    nblocks
):
    rows = np.asarray(
        rows,
        dtype=float
    )

    n = len(
        rows
    )

    block = (
        n//nblocks
    )

    if block < 2:
        raise RuntimeError(
            "insufficient samples for jackknife"
        )

    use = (
        block*nblocks
    )

    rows = rows[
        :use
    ]

    bs = np.asarray(
        [
            np.sum(
                rows[
                    i*block:
                    (i+1)*block
                ],
                axis=0
            )
            for i in range(
                nblocks
            )
        ]
    )

    total = np.sum(
        bs,
        axis=0
    )

    full = derived(
        total,
        use,
        L
    )

    out = {}

    for key in full:
        vals = []

        for i in range(
            nblocks
        ):
            d = derived(
                total-bs[i],
                use-block,
                L
            )

            vals.append(
                d[key]
            )

        vals = np.asarray(
            vals,
            dtype=float
        )

        vm = np.mean(
            vals
        )

        se = np.sqrt(
            (
                nblocks-1
            )
            / nblocks
            * np.sum(
                (
                    vals-vm
                )**2
            )
        )

        out[key] = {
            "value": float(
                full[key]
            ),

            "jk_se": float(
                se
            )
        }

    return out


def autocorr_time(
    x,
    max_lag=1500
):
    x = np.asarray(
        x,
        dtype=float
    )

    n = len(x)

    x = (
        x
        - np.mean(x)
    )

    var = float(
        np.dot(
            x,
            x
        )
        / n
    )

    if (
        n < 20
        or var <= 0
    ):
        return 0.5

    tau = 0.5

    max_lag = min(
        max_lag,
        n-1
    )

    for lag in range(
        1,
        max_lag+1
    ):
        c = float(
            np.dot(
                x[:-lag],
                x[lag:]
            )
            / (
                n-lag
            )
        )

        rho = (
            c/var
        )

        if rho <= 0:
            break

        tau += rho

        if lag > 6.0*tau:
            break

    return float(
        tau
    )


def run_chain(
    *,
    L,
    mref,
    seed
):
    m0 = float(
        CFG[
            "m0_sq"
        ]
    )

    lam = float(
        CFG[
            "lambda"
        ]
    )

    hcfg = CFG[
        "hmc"
    ]

    warmup = int(
        hcfg[
            "warmup"
        ]
    )

    production = int(
        hcfg[
            "production"
        ]
    )

    traj = float(
        hcfg[
            "trajectory_length"
        ]
    )

    eps = float(
        hcfg[
            "initial_step_size"
        ]
    )

    rng = np.random.default_rng(
        int(seed)
    )

    model = LatentPhi4(
        L,
        m0,
        lam,
        mref
    )

    # ---------------------------------------------------------
    # Start at a modest physical field drawn by pushing forward
    # a standard latent Gaussian.
    # ---------------------------------------------------------

    theta = rng.standard_normal(
        (
            L,
            L
        )
    )

    recent_att = 0
    recent_acc = 0

    warm_accept = 0
    warm_nonfinite = 0

    # ---------------------------------------------------------
    # Warmup with only step-size adaptation.
    # ---------------------------------------------------------

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
            trajectory_length=traj,
            rng=rng
        )

        recent_att += 1

        if accepted:
            recent_acc += 1
            warm_accept += 1

        if nonfinite:
            warm_nonfinite += 1

        if (
            rng.random()
            < float(
                hcfg[
                    "global_z2_probability"
                ]
            )
        ):
            theta = -theta

        if (
            (
                sweep+1
            )
            % int(
                hcfg[
                    "adapt_interval"
                ]
            )
            == 0
        ):
            rate = (
                recent_acc
                / max(
                    recent_att,
                    1
                )
            )

            if (
                rate
                > float(
                    hcfg[
                        "acceptance_target_high"
                    ]
                )
            ):
                eps *= float(
                    hcfg[
                        "adapt_up"
                    ]
                )

            elif (
                rate
                < float(
                    hcfg[
                        "acceptance_target_low"
                    ]
                )
            ):
                eps *= float(
                    hcfg[
                        "adapt_down"
                    ]
                )

            eps = float(
                np.clip(
                    eps,
                    float(
                        hcfg[
                            "step_size_min"
                        ]
                    ),
                    float(
                        hcfg[
                            "step_size_max"
                        ]
                    )
                )
            )

            recent_att = 0
            recent_acc = 0

    # ---------------------------------------------------------
    # Production.
    # ---------------------------------------------------------

    rows = np.empty(
        (
            production,
            6
        ),
        dtype=float
    )

    accepted_count = 0
    nonfinite_count = 0

    dH_values = []

    total_steps = 0
    total_grad_evals = 0

    step_sizes = []

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
            trajectory_length=traj,
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

        # action_grad at each leapfrog position:
        # approximately nsteps+1 evaluations.
        total_grad_evals += int(
            nsteps+1
        )

        step_sizes.append(
            actual_eps
        )

        if np.isfinite(
            dH
        ):
            dH_values.append(
                dH
            )

        if (
            rng.random()
            < float(
                hcfg[
                    "global_z2_probability"
                ]
            )
        ):
            theta = -theta

        phi = model.field(
            theta
        )

        rows[
            sweep
        ] = primitive_row(
            phi
        )

    elapsed = (
        time.perf_counter()
        - t0
    )

    estimates = block_jackknife(
        rows,
        L=L,
        nblocks=int(
            CFG[
                "jackknife_blocks"
            ]
        )
    )

    # Primitive slow-mode diagnostics.
    tau_phi2 = autocorr_time(
        rows[
            :,
            0
        ]
    )

    tau_absM = autocorr_time(
        rows[
            :,
            1
        ]
    )

    tau_worst = max(
        tau_phi2,
        tau_absM
    )

    Neff = (
        production
        / (
            2.0
            * max(
                tau_worst,
                0.5
            )
        )
    )

    ESS_per_grad = (
        Neff
        / max(
            total_grad_evals,
            1
        )
    )

    ESS_per_sec = (
        Neff
        / max(
            elapsed,
            1e-12
        )
    )

    return {
        "estimates": estimates,

        "diagnostics": {
            "acceptance": float(
                accepted_count
                / production
            ),

            "warmup_acceptance": float(
                warm_accept
                / warmup
            ),

            "final_nominal_step_size": float(
                eps
            ),

            "mean_actual_step_size": float(
                np.mean(
                    step_sizes
                )
            ),

            "mean_leapfrog_steps": float(
                total_steps
                / production
            ),

            "tau_phi2": float(
                tau_phi2
            ),

            "tau_absM": float(
                tau_absM
            ),

            "tau_worst": float(
                tau_worst
            ),

            "Neff_worst": float(
                Neff
            ),

            "ESS_per_grad": float(
                ESS_per_grad
            ),

            "ESS_per_1000_grad": float(
                1000.0
                * ESS_per_grad
            ),

            "ESS_per_sec": float(
                ESS_per_sec
            ),

            "production_seconds": float(
                elapsed
            ),

            "total_grad_evals": int(
                total_grad_evals
            ),

            "nonfinite_warmup": int(
                warm_nonfinite
            ),

            "nonfinite_production": int(
                nonfinite_count
            ),

            "mean_abs_dH": float(
                np.mean(
                    np.abs(
                        dH_values
                    )
                )
                if dH_values
                else np.nan
            )
        }
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--L",
        required=True,
        type=int
    )

    ap.add_argument(
        "--mref",
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

    # Distinct but deterministic stream per point.
    eff_seed = (
        int(
            args.seed
        )
        + 1000000
        * int(
            args.L
        )
        + int(
            round(
                10000
                * float(
                    args.mref
                )
            )
        )
    )

    result = run_chain(
        L=int(
            args.L
        ),
        mref=float(
            args.mref
        ),
        seed=eff_seed
    )

    out = {
        "campaign": CFG[
            "campaign"
        ],

        "dimension": 2,

        "L": int(
            args.L
        ),

        "V": int(
            args.L*args.L
        ),

        "m0_sq": float(
            CFG[
                "m0_sq"
            ]
        ),

        "lambda": float(
            CFG[
                "lambda"
            ]
        ),

        "m_ref_sq": float(
            args.mref
        ),

        "seed": int(
            args.seed
        ),

        "effective_seed": int(
            eff_seed
        ),

        "estimates": result[
            "estimates"
        ],

        "diagnostics": result[
            "diagnostics"
        ]
    }

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
        f"L={args.L} "
        f"mref={args.mref:g} "
        f"seed={args.seed} "
        f"acc={result['diagnostics']['acceptance']:.3f} "
        f"eps={result['diagnostics']['mean_actual_step_size']:.5f} "
        f"steps={result['diagnostics']['mean_leapfrog_steps']:.1f} "
        f"Neff={result['diagnostics']['Neff_worst']:.1f} "
        f"ESS/1kgrad="
        f"{result['diagnostics']['ESS_per_1000_grad']:.4f} "
        f"U4="
        f"{result['estimates']['binder_U4']['value']:+.5f} "
        f"xi/L="
        f"{result['estimates']['xi_over_L']['value']:.5f}",
        flush=True
    )


if __name__ == "__main__":
    main()
