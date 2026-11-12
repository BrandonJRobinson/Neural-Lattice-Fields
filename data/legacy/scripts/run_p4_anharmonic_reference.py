#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh

from p4_anharmonic_common import fit_gap


ROOT = Path(__file__).resolve().parents[1]

CFG = json.loads(
    (
        ROOT
        / "configs"
        / "p4_anharmonic_oscillator_v1.json"
    ).read_text()
)

OUT = (
    ROOT
    / "results"
    / "p4_anharmonic_oscillator_v1"
    / "reference"
)


# =====================================================================
# HAMILTONIAN DIAGONALIZATION
# =====================================================================

def hamiltonian_reference(
    dim,
    basis_omega,
    physical_omega,
    g
):
    n = np.arange(
        1,
        dim,
        dtype=float
    )

    annih = np.diag(
        np.sqrt(
            n
        ),
        1
    )

    create = annih.T

    X = (
        annih+create
    )/math.sqrt(
        2.0*basis_omega
    )

    P = (
        1j
        * math.sqrt(
            basis_omega/2.0
        )
        * (
            create-annih
        )
    )

    X2 = (
        X@X
    )

    X4 = (
        X2@X2
    )

    H = (
        0.5
        * (
            P@P
        ).real
        + 0.5
        * physical_omega**2
        * X2
        + g*X4
    )

    E,U = eigh(
        H
    )

    psi0 = U[
        :,
        0
    ]

    x2 = float(
        psi0
        @ (
            X2@psi0
        )
    )

    x4 = float(
        psi0
        @ (
            X4@psi0
        )
    )

    return {
        "E0": float(
            E[0]
        ),

        "E1": float(
            E[1]
        ),

        "gap": float(
            E[1]-E[0]
        ),

        "x2": x2,

        "x4": x4
    }


# =====================================================================
# FINITE-LATTICE TRANSFER MATRIX
# =====================================================================

def transfer_reference(
    *,
    Nt,
    beta,
    omega,
    g,
    nq,
    xmax,
    gap_cfg
):
    a = (
        beta/Nt
    )

    z,w = leggauss(
        int(
            nq
        )
    )

    x = (
        xmax*z
    )

    w = (
        xmax*w
    )

    V = (
        0.5
        * omega*omega
        * x*x
        + g*x**4
    )

    dx = (
        x[:,None]
        - x[None,:]
    )

    exponent = (
        -dx*dx/(2.0*a)
        -0.5*a
        * (
            V[:,None]
            + V[None,:]
        )
    )

    kernel = (
        np.exp(
            exponent
        )
        / math.sqrt(
            2.0*math.pi*a
        )
    )

    sw = np.sqrt(
        w
    )

    B = (
        sw[:,None]
        * kernel
        * sw[None,:]
    )

    lam,U = eigh(
        B
    )

    idx = np.argsort(
        lam
    )[::-1]

    lam = lam[
        idx
    ]

    U = U[
        :,
        idx
    ]

    if lam[0] <= 0:
        raise RuntimeError(
            "non-positive leading transfer eigenvalue"
        )

    r = np.maximum(
        lam/lam[0],
        0.0
    )

    thermal = (
        r**Nt
    )

    Z = float(
        np.sum(
            thermal
        )
    )

    X2diag = np.sum(
        U
        * (
            x*x
        )[:,None]
        * U,
        axis=0
    )

    X4diag = np.sum(
        U
        * (
            x**4
        )[:,None]
        * U,
        axis=0
    )

    x2 = float(
        np.dot(
            thermal,
            X2diag
        )/Z
    )

    x4 = float(
        np.dot(
            thermal,
            X4diag
        )/Z
    )

    Xmat = (
        U.T
        @ (
            x[:,None]*U
        )
    )

    Xabs2 = (
        Xmat*Xmat
    )

    C = np.empty(
        Nt,
        dtype=float
    )

    for j in range(
        Nt
    ):
        left = (
            r**(
                Nt-j
            )
        )

        right = (
            r**j
        )

        C[j] = float(
            np.sum(
                left[:,None]
                * right[None,:]
                * Xabs2
            )/Z
        )

    gap_spectral = float(
        -math.log(
            r[1]
        )/a
    )

    gap_fit = fit_gap(
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

    return {
        "Nt": int(
            Nt
        ),

        "a": float(
            a
        ),

        "x2": x2,

        "x4": x4,

        "gap_spectral": gap_spectral,

        "gap_fit": float(
            gap_fit
        ),

        "correlator": [
            float(v)
            for v in C
        ]
    }


def relative(
    a,
    b
):
    return abs(
        a-b
    )/max(
        abs(b),
        1e-30
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    hamcfg = CFG[
        "hamiltonian"
    ]

    href = CFG[
        "hamiltonian_reference"
    ]

    beta = float(
        CFG[
            "euclidean"
        ][
            "beta"
        ]
    )

    omega = float(
        hamcfg[
            "omega"
        ]
    )

    g = float(
        hamcfg[
            "g"
        ]
    )

    hprimary = hamiltonian_reference(
        int(
            href[
                "dim_primary"
            ]
        ),
        float(
            href[
                "basis_omega"
            ]
        ),
        omega,
        g
    )

    hcheck = hamiltonian_reference(
        int(
            href[
                "dim_check"
            ]
        ),
        float(
            href[
                "basis_omega"
            ]
        ),
        omega,
        g
    )

    hconv = {
        key: relative(
            hprimary[
                key
            ],
            hcheck[
                key
            ]
        )
        for key in [
            "E0",
            "gap",
            "x2",
            "x4"
        ]
    }

    hpass = all(
        v
        <= float(
            href[
                "relative_convergence_max"
            ]
        )
        for v in hconv.values()
    )

    print(
        "Hamiltonian primary:",
        hprimary,
        flush=True
    )

    print(
        "Hamiltonian convergence:",
        hconv,
        "PASS" if hpass else "FAIL",
        flush=True
    )

    tcfg = CFG[
        "transfer_reference"
    ]

    gapcfg = CFG[
        "gap_fit"
    ]

    transfer = {}

    transfer_checks = {}

    transfer_pass = True

    for Nt in CFG[
        "euclidean"
    ][
        "Nt_values"
    ]:
        print(
            f"transfer Nt={Nt}",
            flush=True
        )

        primary = transfer_reference(
            Nt=int(
                Nt
            ),
            beta=beta,
            omega=omega,
            g=g,
            nq=int(
                tcfg[
                    "quadrature_primary"
                ]
            ),
            xmax=float(
                tcfg[
                    "x_max"
                ]
            ),
            gap_cfg=gapcfg
        )

        qcheck = transfer_reference(
            Nt=int(
                Nt
            ),
            beta=beta,
            omega=omega,
            g=g,
            nq=int(
                tcfg[
                    "quadrature_check"
                ]
            ),
            xmax=float(
                tcfg[
                    "x_max"
                ]
            ),
            gap_cfg=gapcfg
        )

        dcheck = transfer_reference(
            Nt=int(
                Nt
            ),
            beta=beta,
            omega=omega,
            g=g,
            nq=int(
                tcfg[
                    "quadrature_primary"
                ]
            ),
            xmax=float(
                tcfg[
                    "x_max_domain_check"
                ]
            ),
            gap_cfg=gapcfg
        )

        checks = {}

        for key in [
            "x2",
            "x4",
            "gap_spectral",
            "gap_fit"
        ]:
            checks[
                key
            ] = {
                "quadrature_relative": relative(
                    qcheck[
                        key
                    ],
                    primary[
                        key
                    ]
                ),

                "domain_relative": relative(
                    dcheck[
                        key
                    ],
                    primary[
                        key
                    ]
                )
            }

            transfer_pass &= (
                checks[
                    key
                ][
                    "quadrature_relative"
                ]
                <= float(
                    tcfg[
                        "relative_convergence_max"
                    ]
                )
            )

            transfer_pass &= (
                checks[
                    key
                ][
                    "domain_relative"
                ]
                <= float(
                    tcfg[
                        "relative_convergence_max"
                    ]
                )
            )

        transfer[
            str(
                Nt
            )
        ] = primary

        transfer_checks[
            str(
                Nt
            )
        ] = checks

        print(
            f"  x2={primary['x2']:.10f} "
            f"x4={primary['x4']:.10f} "
            f"gap={primary['gap_spectral']:.10f} "
            f"gap_fit={primary['gap_fit']:.10f}",
            flush=True
        )

    # =============================================================
    # Deterministic continuum checks.
    #
    # 1. quadratic polynomial in a^2 over all five points.
    # 2. linear in a^2 over finest four.
    #
    # Both must agree with independent Hamiltonian diagonalization.
    # =============================================================

    continuum = {}

    cgate = CFG[
        "continuum_reference_gates"
    ]

    continuum_pass = True

    for key,href_key in [
        (
            "x2",
            "x2"
        ),
        (
            "x4",
            "x4"
        ),
        (
            "gap_spectral",
            "gap"
        )
    ]:
        rr = [
            transfer[
                str(
                    Nt
                )
            ]
            for Nt in CFG[
                "euclidean"
            ][
                "Nt_values"
            ]
        ]

        aa = np.asarray(
            [
                r[
                    "a"
                ]
                for r in rr
            ],
            float
        )

        yy = np.asarray(
            [
                r[
                    key
                ]
                for r in rr
            ],
            float
        )

        x2a = (
            aa*aa
        )

        # all-five: y = c0+c1 a^2+c2 a^4
        X = np.column_stack(
            [
                np.ones_like(
                    x2a
                ),
                x2a,
                x2a*x2a
            ]
        )

        coef = np.linalg.lstsq(
            X,
            yy,
            rcond=None
        )[0]

        c0_all = float(
            coef[
                0
            ]
        )

        # finest four: y = c0+c1 a^2
        xf = x2a[
            1:
        ]

        yf = yy[
            1:
        ]

        Xf = np.column_stack(
            [
                np.ones_like(
                    xf
                ),
                xf
            ]
        )

        coeff = np.linalg.lstsq(
            Xf,
            yf,
            rcond=None
        )[0]

        c0_fine = float(
            coeff[
                0
            ]
        )

        ref = float(
            hprimary[
                href_key
            ]
        )

        err_all = relative(
            c0_all,
            ref
        )

        err_fine = relative(
            c0_fine,
            ref
        )

        stability = relative(
            c0_all,
            c0_fine
        )

        passed = (
            err_all
            <= float(
                cgate[
                    "relative_error_max"
                ]
            )
            and err_fine
            <= float(
                cgate[
                    "relative_error_max"
                ]
            )
            and stability
            <= float(
                cgate[
                    "fit_stability_relative_max"
                ]
            )
        )

        continuum_pass &= passed

        continuum[
            key
        ] = {
            "hamiltonian_reference": ref,

            "continuum_all5": c0_all,

            "continuum_fine4": c0_fine,

            "relative_error_all5": err_all,

            "relative_error_fine4": err_fine,

            "fit_stability_relative": stability,

            "pass": bool(
                passed
            )
        }

        print(
            f"continuum {key}: "
            f"exact={ref:.10f} "
            f"all5={c0_all:.10f} "
            f"fine4={c0_fine:.10f} "
            f"err={err_all:.3e}/{err_fine:.3e} "
            f"stable={stability:.3e} "
            f"{'PASS' if passed else 'FAIL'}",
            flush=True
        )

    reference_pass = bool(
        hpass
        and transfer_pass
        and continuum_pass
    )

    out = {
        "campaign": CFG[
            "campaign"
        ],

        "hamiltonian": {
            "primary": hprimary,
            "check": hcheck,
            "relative_convergence": hconv,
            "pass": bool(
                hpass
            )
        },

        "transfer": transfer,

        "transfer_checks": transfer_checks,

        "transfer_pass": bool(
            transfer_pass
        ),

        "continuum": continuum,

        "continuum_pass": bool(
            continuum_pass
        ),

        "reference_pass": bool(
            reference_pass
        )
    }

    (
        OUT
        / "reference.json"
    ).write_text(
        json.dumps(
            out,
            indent=2,
            sort_keys=True
        )
        + "\n"
    )

    print()
    print(
        "P4 DETERMINISTIC REFERENCE GATE =",
        reference_pass
    )


if __name__ == "__main__":
    main()
