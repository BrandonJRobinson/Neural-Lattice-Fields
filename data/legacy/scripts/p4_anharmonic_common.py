#!/usr/bin/env python3

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar


def lattice_phat_kernel(
    Nt,
    a,
    omega
):
    k = np.arange(
        Nt,
        dtype=float
    )

    return (
        4.0/a
        * np.sin(
            np.pi*k/Nt
        )**2
        + a*omega*omega
    )


class Whitening:
    def __init__(
        self,
        Nt,
        a,
        omega_ref
    ):
        self.Nt = int(
            Nt
        )

        self.K = lattice_phat_kernel(
            Nt,
            a,
            omega_ref
        )

        if np.min(
            self.K
        ) <= 0:
            raise ValueError(
                "non-positive reference kernel"
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

    def inv(
        self,
        theta
    ):
        ft = np.fft.fft(
            theta,
            norm="ortho"
        )

        return np.fft.ifft(
            ft*self.inv_sqrt,
            norm="ortho"
        ).real

    def sqrt_apply(
        self,
        x
    ):
        ft = np.fft.fft(
            x,
            norm="ortho"
        )

        return np.fft.ifft(
            ft*self.sqrt,
            norm="ortho"
        ).real


class LatentOscillator:
    def __init__(
        self,
        Nt,
        beta,
        omega,
        g,
        omega_ref
    ):
        self.Nt = int(
            Nt
        )

        self.beta = float(
            beta
        )

        self.a = (
            self.beta/self.Nt
        )

        self.omega = float(
            omega
        )

        self.g = float(
            g
        )

        self.omega_ref = float(
            omega_ref
        )

        self.W = Whitening(
            Nt,
            self.a,
            omega_ref
        )

        self.delta_omega2 = (
            self.omega*self.omega
            - self.omega_ref*self.omega_ref
        )

    def field(
        self,
        theta
    ):
        return self.W.inv(
            theta
        )

    def action(
        self,
        theta
    ):
        x = self.field(
            theta
        )

        return float(
            0.5*np.sum(
                theta*theta
            )
            + self.a
            * np.sum(
                0.5
                * self.delta_omega2
                * x*x
                + self.g*x**4
            )
        )

    def action_grad(
        self,
        theta
    ):
        x = self.field(
            theta
        )

        value = (
            0.5*np.sum(
                theta*theta
            )
            + self.a
            * np.sum(
                0.5
                * self.delta_omega2
                * x*x
                + self.g*x**4
            )
        )

        dx = (
            self.a
            * (
                self.delta_omega2*x
                + 4.0*self.g*x**3
            )
        )

        grad = (
            theta
            + self.W.inv(
                dx
            )
        )

        return (
            float(
                value
            ),
            grad
        )


def physical_lattice_action(
    x,
    *,
    beta,
    omega,
    g
):
    Nt = len(
        x
    )

    a = (
        beta/Nt
    )

    dx = (
        np.roll(
            x,
            -1
        )
        - x
    )

    return float(
        np.sum(
            0.5/a*dx*dx
            + a
            * (
                0.5
                * omega*omega
                * x*x
                + g*x**4
            )
        )
    )


def circular_correlator(
    x
):
    Nt = len(
        x
    )

    F = np.fft.fft(
        x
    )

    C = np.fft.ifft(
        np.abs(F)**2
    ).real/Nt

    return C


def fit_gap(
    C,
    *,
    a,
    beta,
    tau_min,
    tau_max,
    mass_lower,
    mass_upper
):
    C = np.asarray(
        C,
        dtype=float
    )

    Nt = len(
        C
    )

    jmin = max(
        1,
        int(
            math.ceil(
                tau_min/a
                - 1e-12
            )
        )
    )

    jmax = min(
        Nt//2-1,
        int(
            math.floor(
                tau_max/a
                + 1e-12
            )
        )
    )

    if jmax-jmin < 3:
        raise RuntimeError(
            "insufficient correlator points for gap fit"
        )

    js = np.arange(
        jmin,
        jmax+1
    )

    tau = (
        a*js
    )

    y = C[
        js
    ]

    if not np.all(
        np.isfinite(
            y
        )
    ):
        raise RuntimeError(
            "nonfinite correlator"
        )

    def objective(
        m
    ):
        f = (
            np.exp(
                -m*tau
            )
            + np.exp(
                -m*(
                    beta-tau
                )
            )
        )

        den = float(
            np.dot(
                f,
                f
            )
        )

        if den <= 0:
            return np.inf

        amp = float(
            np.dot(
                y,
                f
            )
            / den
        )

        r = (
            y
            - amp*f
        )

        return float(
            np.dot(
                r,
                r
            )
        )

    opt = minimize_scalar(
        objective,
        bounds=(
            mass_lower,
            mass_upper
        ),
        method="bounded",
        options={
            "xatol":1e-10,
            "maxiter":200
        }
    )

    return float(
        opt.x
    )


def positive_sequence_tau(
    x,
    max_lag=5000
):
    x = np.asarray(
        x,
        dtype=float
    )

    n = len(
        x
    )

    x = (
        x
        - np.mean(
            x
        )
    )

    var = float(
        np.dot(
            x,
            x
        )/n
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
        rho = float(
            np.dot(
                x[:-lag],
                x[lag:]
            )
            / (
                (n-lag)*var
            )
        )

        if rho <= 0:
            break

        tau += rho

        if lag > 6.0*tau:
            break

    return float(
        tau
    )


def jackknife_scalar(
    values
):
    values = np.asarray(
        values,
        dtype=float
    )

    n = len(
        values
    )

    full = float(
        np.mean(
            values
        )
    )

    total = float(
        np.sum(
            values
        )
    )

    jk = (
        total-values
    )/(n-1)

    jm = float(
        np.mean(
            jk
        )
    )

    se = float(
        np.sqrt(
            (n-1)/n
            * np.sum(
                (
                    jk-jm
                )**2
            )
        )
    )

    return {
        "value": full,
        "jk_se": se
    }


def jackknife_gap(
    block_C,
    *,
    a,
    beta,
    gap_cfg
):
    B = np.asarray(
        block_C,
        dtype=float
    )

    n = B.shape[
        0
    ]

    full_C = np.mean(
        B,
        axis=0
    )

    full = fit_gap(
        full_C,
        a=a,
        beta=beta,
        tau_min=float(
            gap_cfg[
                "tau_min"
            ]
        ),
        tau_max=float(
            gap_cfg[
                "tau_max"
            ]
        ),
        mass_lower=float(
            gap_cfg[
                "mass_lower"
            ]
        ),
        mass_upper=float(
            gap_cfg[
                "mass_upper"
            ]
        )
    )

    total = np.sum(
        B,
        axis=0
    )

    vals = []

    for i in range(
        n
    ):
        C = (
            total-B[i]
        )/(n-1)

        vals.append(
            fit_gap(
                C,
                a=a,
                beta=beta,
                tau_min=float(
                    gap_cfg[
                        "tau_min"
                    ]
                ),
                tau_max=float(
                    gap_cfg[
                        "tau_max"
                    ]
                ),
                mass_lower=float(
                    gap_cfg[
                        "mass_lower"
                    ]
                ),
                mass_upper=float(
                    gap_cfg[
                        "mass_upper"
                    ]
                )
            )
        )

    vals = np.asarray(
        vals,
        float
    )

    vm = float(
        np.mean(
            vals
        )
    )

    se = float(
        np.sqrt(
            (n-1)/n
            * np.sum(
                (
                    vals-vm
                )**2
            )
        )
    )

    return {
        "value": float(
            full
        ),
        "jk_se": se
    }
