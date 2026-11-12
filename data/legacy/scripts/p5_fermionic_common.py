#!/usr/bin/env python3

from __future__ import annotations

import math
import numpy as np


SIGMA1 = np.asarray(
    [
        [0.0, 1.0],
        [1.0, 0.0]
    ],
    dtype=np.complex128
)

SIGMA2 = np.asarray(
    [
        [0.0, -1.0j],
        [1.0j, 0.0]
    ],
    dtype=np.complex128
)

GAMMAS = [
    SIGMA1,
    SIGMA2
]

I2 = np.eye(
    2,
    dtype=np.complex128
)


def site_index(
    x,
    y,
    L
):
    return int(
        x*L+y
    )


def wilson_dense(
    L,
    mf,
    r=1.0
):
    L = int(L)

    V = L*L
    nd = 2*V

    D = np.zeros(
        (
            nd,
            nd
        ),
        dtype=np.complex128
    )

    for x in range(L):
        for y in range(L):
            s = site_index(
                x,
                y,
                L
            )

            sl = slice(
                2*s,
                2*s+2
            )

            D[
                sl,
                sl
            ] += (
                mf+2.0*r
            )*I2

            neighbors = [
                (
                    ((x+1) % L, y),
                    ((x-1) % L, y)
                ),
                (
                    (x, (y+1) % L),
                    (x, (y-1) % L)
                )
            ]

            for mu in range(2):
                plus,minus = neighbors[
                    mu
                ]

                sp = site_index(
                    plus[0],
                    plus[1],
                    L
                )

                sm = site_index(
                    minus[0],
                    minus[1],
                    L
                )

                spl = slice(
                    2*sp,
                    2*sp+2
                )

                sml = slice(
                    2*sm,
                    2*sm+2
                )

                D[
                    sl,
                    spl
                ] += (
                    -0.5
                    * (
                        r*I2
                        - GAMMAS[
                            mu
                        ]
                    )
                )

                D[
                    sl,
                    sml
                ] += (
                    -0.5
                    * (
                        r*I2
                        + GAMMAS[
                            mu
                        ]
                    )
                )

    return D


def wilson_background(
    D0,
    sigma,
    g
):
    sigma = np.asarray(
        sigma,
        dtype=float
    ).reshape(-1)

    V = sigma.size

    D = D0.copy()

    for s in range(V):
        for a in range(2):
            i = 2*s+a

            D[
                i,
                i
            ] += (
                g*sigma[
                    s
                ]
            )

    return D


def wilson_background_batch(
    D0,
    sigma,
    g
):
    sigma = np.asarray(
        sigma,
        dtype=float
    )

    B,V = sigma.shape

    nd = D0.shape[
        0
    ]

    D = np.broadcast_to(
        D0,
        (
            B,
            nd,
            nd
        )
    ).copy()

    for s in range(V):
        for a in range(2):
            i = 2*s+a

            D[
                :,
                i,
                i
            ] += (
                g*sigma[
                    :,
                    s
                ]
            )

    return D


def free_logdet_analytic(
    L,
    mf,
    r=1.0
):
    total = 0.0

    for n1 in range(
        L
    ):
        p1 = (
            2.0
            * np.pi
            * n1/L
        )

        for n2 in range(
            L
        ):
            p2 = (
                2.0
                * np.pi
                * n2/L
            )

            A = (
                mf
                + r
                * (
                    2.0
                    - np.cos(
                        p1
                    )
                    - np.cos(
                        p2
                    )
                )
            )

            s2 = (
                np.sin(
                    p1
                )**2
                + np.sin(
                    p2
                )**2
            )

            det_block = (
                A*A+s2
            )

            if det_block <= 0:
                return -np.inf

            total += math.log(
                det_block
            )

    return float(
        total
    )


def fermion_logdet(
    D
):
    sign,logabs = np.linalg.slogdet(
        D
    )

    if (
        not np.isfinite(
            logabs
        )
    ):
        return -np.inf

    return float(
        logabs
    )


def fermion_logdet_force_condensate(
    D,
    *,
    g,
    V
):
    sign,logabs = np.linalg.slogdet(
        D
    )

    if (
        not np.isfinite(
            logabs
        )
    ):
        return (
            -np.inf,
            None,
            np.nan
        )

    try:
        inv = np.linalg.inv(
            D
        )

    except np.linalg.LinAlgError:
        return (
            -np.inf,
            None,
            np.nan
        )

    force = np.zeros(
        V,
        dtype=float
    )

    for s in range(
        V
    ):
        block = inv[
            2*s:2*s+2,
            2*s:2*s+2
        ]

        force[
            s
        ] = (
            g
            * float(
                np.real(
                    np.trace(
                        block
                    )
                )
            )
        )

    condensate = float(
        np.real(
            np.trace(
                inv
            )
        )/V
    )

    return (
        float(
            logabs
        ),
        force,
        condensate
    )


def phat_sq_2d(
    L
):
    p = (
        2.0
        * np.pi
        * np.fft.fftfreq(
            L
        )
    )

    return (
        4.0
        * np.sin(
            p[:,None]/2.0
        )**2
        + 4.0
        * np.sin(
            p[None,:]/2.0
        )**2
    )


class AuxiliaryWhitening:
    def __init__(
        self,
        L,
        msigma_sq
    ):
        self.L = int(
            L
        )

        self.V = (
            self.L*self.L
        )

        self.K = (
            phat_sq_2d(
                self.L
            )
            + float(
                msigma_sq
            )
        )

        if np.min(
            self.K
        ) <= 0:
            raise ValueError(
                "auxiliary kernel is not positive"
            )

        self.sqrtK = np.sqrt(
            self.K
        )

        self.inv_sqrtK = (
            1.0/self.sqrtK
        )

    def inv(
        self,
        theta
    ):
        theta = np.asarray(
            theta,
            dtype=float
        ).reshape(
            self.L,
            self.L
        )

        ft = np.fft.fftn(
            theta,
            norm="ortho"
        )

        sigma = np.fft.ifftn(
            ft*self.inv_sqrtK,
            norm="ortho"
        ).real

        return sigma

    def sqrt_apply(
        self,
        sigma
    ):
        sigma = np.asarray(
            sigma,
            dtype=float
        ).reshape(
            self.L,
            self.L
        )

        ft = np.fft.fftn(
            sigma,
            norm="ortho"
        )

        theta = np.fft.ifftn(
            ft*self.sqrtK,
            norm="ortho"
        ).real

        return theta

    def inv_matrix(
        self
    ):
        M = np.empty(
            (
                self.V,
                self.V
            ),
            dtype=float
        )

        for j in range(
            self.V
        ):
            e = np.zeros(
                self.V,
                dtype=float
            )

            e[
                j
            ] = 1.0

            M[
                :,
                j
            ] = self.inv(
                e
            ).reshape(-1)

        return M


class LatentFermionAuxiliary:
    def __init__(
        self,
        *,
        L,
        Nf,
        mf,
        r,
        msigma_sq,
        g
    ):
        self.L = int(
            L
        )

        self.V = (
            self.L*self.L
        )

        self.Nf = int(
            Nf
        )

        self.mf = float(
            mf
        )

        self.r = float(
            r
        )

        self.msigma_sq = float(
            msigma_sq
        )

        self.g = float(
            g
        )

        self.W = AuxiliaryWhitening(
            self.L,
            self.msigma_sq
        )

        self.D0 = wilson_dense(
            self.L,
            self.mf,
            self.r
        )

        self.logdet0 = fermion_logdet(
            self.D0
        )

    def sigma(
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
        theta = np.asarray(
            theta,
            dtype=float
        )

        sigma = self.sigma(
            theta
        )

        D = wilson_background(
            self.D0,
            sigma,
            self.g
        )

        logabs = fermion_logdet(
            D
        )

        if not np.isfinite(
            logabs
        ):
            return np.inf

        # Constant logdet0 subtraction does not change the measure.
        return float(
            0.5
            * np.sum(
                theta*theta
            )
            - self.Nf
            * (
                logabs
                - self.logdet0
            )
        )

    def action_grad(
        self,
        theta
    ):
        theta = np.asarray(
            theta,
            dtype=float
        )

        sigma = self.sigma(
            theta
        )

        D = wilson_background(
            self.D0,
            sigma,
            self.g
        )

        (
            logabs,
            force_sigma,
            condensate
        ) = fermion_logdet_force_condensate(
            D,
            g=self.g,
            V=self.V
        )

        if (
            not np.isfinite(
                logabs
            )
            or force_sigma is None
            or not np.all(
                np.isfinite(
                    force_sigma
                )
            )
        ):
            return (
                np.inf,
                np.full_like(
                    theta,
                    np.nan
                )
            )

        # dS/dsigma = -Nf * d log|det|/dsigma.
        # theta-gradient gets symmetric K^{-1/2} pullback.
        grad = (
            theta
            - self.Nf
            * self.W.inv(
                force_sigma
            )
        )

        value = float(
            0.5
            * np.sum(
                theta*theta
            )
            - self.Nf
            * (
                logabs
                - self.logdet0
            )
        )

        return (
            value,
            grad
        )

    def observables(
        self,
        theta
    ):
        sigma = self.sigma(
            theta
        )

        D = wilson_background(
            self.D0,
            sigma,
            self.g
        )

        (
            logabs,
            force,
            condensate
        ) = fermion_logdet_force_condensate(
            D,
            g=self.g,
            V=self.V
        )

        if (
            not np.isfinite(
                logabs
            )
            or not np.isfinite(
                condensate
            )
        ):
            raise FloatingPointError(
                "nonfinite fermion observable"
            )

        bar = float(
            np.mean(
                sigma
            )
        )

        return {
            "sigma_mean": bar,

            "sigma2": float(
                np.mean(
                    sigma*sigma
                )
            ),

            "sigma_bar2": float(
                bar*bar
            ),

            "condensate": float(
                condensate
            ),

            "logdet_density": float(
                logabs/self.V
            )
        }


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
        int(
            max_lag
        ),
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
                (n-lag)
                * var
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
