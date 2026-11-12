#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

from p5_fermionic_common import (
    AuxiliaryWhitening,
    free_logdet_analytic,
    fermion_logdet,
    wilson_background,
    wilson_background_batch,
    wilson_dense
)


ROOT = Path(__file__).resolve().parents[1]

CFG = json.loads(
    (
        ROOT
        / "configs"
        / "p5_fermionic_full_measure_v1.json"
    ).read_text()
)

OUT = (
    ROOT
    / "results"
    / "p5_fermionic_full_measure_v1"
    / "reference"
)


L = int(
    CFG[
        "L"
    ]
)

V = L*L

Nf = int(
    CFG[
        "fermion"
    ][
        "Nf"
    ]
)

r = float(
    CFG[
        "fermion"
    ][
        "wilson_r"
    ]
)

mf0 = float(
    CFG[
        "fermion"
    ][
        "mf"
    ]
)

msigma_sq = float(
    CFG[
        "auxiliary"
    ][
        "msigma_sq"
    ]
)

RCFG = CFG[
    "reference"
]

W = AuxiliaryWhitening(
    L,
    msigma_sq
)

WMAT = W.inv_matrix()


def relative(
    a,
    b
):
    return abs(
        a-b
    )/max(
        abs(
            b
        ),
        1e-30
    )


def gh_grid(
    nq
):
    nodes,weights = hermgauss(
        int(
            nq
        )
    )

    mesh = np.meshgrid(
        *(
            [nodes]*V
        ),
        indexing="ij"
    )

    wmesh = np.meshgrid(
        *(
            [weights]*V
        ),
        indexing="ij"
    )

    theta = np.stack(
        [
            m.reshape(-1)
            for m in mesh
        ],
        axis=1
    )

    theta *= math.sqrt(
        2.0
    )

    logq = np.zeros(
        theta.shape[
            0
        ],
        dtype=float
    )

    for wm in wmesh:
        logq += np.log(
            wm.reshape(
                -1
            )
        )

    logq -= (
        0.5
        * V
        * math.log(
            math.pi
        )
    )

    sigma = (
        theta
        @ WMAT.T
    )

    return (
        theta,
        sigma,
        logq
    )


def batch_logdet_and_cond(
    sigma,
    *,
    mf,
    g,
    need_condensate
):
    D0 = wilson_dense(
        L,
        mf,
        r
    )

    n = sigma.shape[
        0
    ]

    bs = int(
        RCFG[
            "batch_size"
        ]
    )

    logdet = np.empty(
        n,
        dtype=float
    )

    cond = (
        np.empty(
            n,
            dtype=float
        )
        if need_condensate
        else None
    )

    for start in range(
        0,
        n,
        bs
    ):
        end = min(
            start+bs,
            n
        )

        D = wilson_background_batch(
            D0,
            sigma[
                start:end
            ],
            g
        )

        sign,ld = np.linalg.slogdet(
            D
        )

        logdet[
            start:end
        ] = ld

        if need_condensate:
            inv = np.linalg.inv(
                D
            )

            tr = np.trace(
                inv,
                axis1=1,
                axis2=2
            )

            cond[
                start:end
            ] = (
                np.real(
                    tr
                )/V
            )

    return (
        logdet,
        cond
    )


def logZ_only(
    sigma,
    logq,
    *,
    mf,
    g
):
    ld,_ = batch_logdet_and_cond(
        sigma,
        mf=mf,
        g=g,
        need_condensate=False
    )

    return float(
        logsumexp(
            logq
            + Nf*ld
        )
    )


def quadrature(
    nq,
    g
):
    (
        theta,
        sigma,
        logq
    ) = gh_grid(
        nq
    )

    ld,cond = batch_logdet_and_cond(
        sigma,
        mf=mf0,
        g=g,
        need_condensate=True
    )

    lw = (
        logq
        + Nf*ld
    )

    logZ = float(
        logsumexp(
            lw
        )
    )

    wnorm = np.exp(
        lw-logZ
    )

    sigma_mean_each = np.mean(
        sigma,
        axis=1
    )

    sigma2_each = np.mean(
        sigma*sigma,
        axis=1
    )

    bar2_each = (
        sigma_mean_each
        * sigma_mean_each
    )

    mean_sigma = float(
        np.dot(
            wnorm,
            sigma_mean_each
        )
    )

    sigma2 = float(
        np.dot(
            wnorm,
            sigma2_each
        )
    )

    mean_bar2 = float(
        np.dot(
            wnorm,
            bar2_each
        )
    )

    chi = float(
        V
        * (
            mean_bar2
            - mean_sigma*mean_sigma
        )
    )

    condensate = float(
        np.dot(
            wnorm,
            cond
        )
    )

    dm = float(
        RCFG[
            "condensate_fd_dm"
        ]
    )

    logZ_plus = logZ_only(
        sigma,
        logq,
        mf=mf0+dm,
        g=g
    )

    logZ_minus = logZ_only(
        sigma,
        logq,
        mf=mf0-dm,
        g=g
    )

    condensate_fd = float(
        (
            logZ_plus-logZ_minus
        )
        / (
            2.0
            * dm
            * Nf
            * V
        )
    )

    return {
        "nq": int(
            nq
        ),

        "nodes": int(
            nq**V
        ),

        "logZ": logZ,

        "sigma_mean": mean_sigma,

        "sigma2": sigma2,

        "chi_sigma": chi,

        "condensate": condensate,

        "condensate_fd": condensate_fd,

        "condensate_fd_relative": relative(
            condensate,
            condensate_fd
        )
    }


# =====================================================================
# Free determinant convention gate
# =====================================================================

free_rows = []
free_pass = True

for Ltest in [
    2,
    4,
    6
]:
    D = wilson_dense(
        Ltest,
        mf0,
        r
    )

    dense = fermion_logdet(
        D
    )

    analytic = free_logdet_analytic(
        Ltest,
        mf0,
        r
    )

    rel = relative(
        dense,
        analytic
    )

    passed = (
        rel
        <= float(
            RCFG[
                "free_logdet_relative_max"
            ]
        )
    )

    free_pass &= passed

    free_rows.append({
        "L":Ltest,

        "dense_logabsdet":dense,

        "analytic_logabsdet":analytic,

        "relative_error":rel,

        "pass":bool(
            passed
        )
    })

    print(
        f"free L={Ltest}: "
        f"dense={dense:.14e} "
        f"analytic={analytic:.14e} "
        f"rel={rel:.3e} "
        f"{'PASS' if passed else 'FAIL'}",
        flush=True
    )


# =====================================================================
# Force finite-difference gate
# =====================================================================

rng = np.random.default_rng(
    20260909
)

Lforce = 4
Vforce = Lforce*Lforce

D0f = wilson_dense(
    Lforce,
    mf0,
    r
)

gforce = 0.375

force_rows = []
force_pass = True

for rep in range(
    12
):
    sigma = (
        0.35
        * rng.standard_normal(
            Vforce
        )
    )

    direction = rng.standard_normal(
        Vforce
    )

    direction /= np.linalg.norm(
        direction
    )

    D = wilson_background(
        D0f,
        sigma,
        gforce
    )

    sign,logabs = np.linalg.slogdet(
        D
    )

    inv = np.linalg.inv(
        D
    )

    force = np.zeros(
        Vforce,
        dtype=float
    )

    for s in range(
        Vforce
    ):
        force[
            s
        ] = (
            gforce
            * float(
                np.real(
                    np.trace(
                        inv[
                            2*s:2*s+2,
                            2*s:2*s+2
                        ]
                    )
                )
            )
        )

    eps = float(
        RCFG[
            "force_fd_eps"
        ]
    )

    Dp = wilson_background(
        D0f,
        sigma+eps*direction,
        gforce
    )

    Dm = wilson_background(
        D0f,
        sigma-eps*direction,
        gforce
    )

    lp = np.linalg.slogdet(
        Dp
    )[1]

    lm = np.linalg.slogdet(
        Dm
    )[1]

    fd = float(
        (
            lp-lm
        )/(2.0*eps)
    )

    exact = float(
        np.dot(
            force,
            direction
        )
    )

    rel = abs(
        fd-exact
    )/max(
        1.0,
        abs(fd),
        abs(exact)
    )

    passed = (
        rel
        <= float(
            RCFG[
                "force_relative_max"
            ]
        )
    )

    force_pass &= passed

    force_rows.append({
        "rep":rep,

        "fd":fd,

        "analytic":exact,

        "relative_error":rel,

        "pass":bool(
            passed
        )
    })

    print(
        f"force rep={rep:02d}: "
        f"fd={fd:+.10e} "
        f"analytic={exact:+.10e} "
        f"rel={rel:.3e} "
        f"{'PASS' if passed else 'FAIL'}",
        flush=True
    )


# =====================================================================
# Full-measure Gauss-Hermite benchmark
# =====================================================================

primary_n = int(
    RCFG[
        "gh_primary"
    ]
)

check_n = int(
    RCFG[
        "gh_check"
    ]
)

points = {}
quad_pass = True
thermo_pass = True

for g in CFG[
    "couplings"
]:
    print(
        f"\nGH g={g:g}: primary",
        flush=True
    )

    primary = quadrature(
        primary_n,
        float(g)
    )

    print(
        f"GH g={g:g}: check",
        flush=True
    )

    check = quadrature(
        check_n,
        float(g)
    )

    convergence = {}

    for key in [
        "sigma2",
        "chi_sigma",
        "condensate"
    ]:
        rel = relative(
            primary[
                key
            ],
            check[
                key
            ]
        )

        passed = (
            rel
            <= float(
                RCFG[
                    "quadrature_relative_max"
                ]
            )
        )

        quad_pass &= passed

        convergence[
            key
        ] = {
            "relative":rel,
            "pass":bool(
                passed
            )
        }

    mean_abs = abs(
        primary[
            "sigma_mean"
        ]
        - check[
            "sigma_mean"
        ]
    )

    mean_pass = (
        mean_abs
        <= float(
            RCFG[
                "quadrature_absolute_mean_max"
            ]
        )
    )

    quad_pass &= mean_pass

    convergence[
        "sigma_mean"
    ] = {
        "absolute":mean_abs,
        "pass":bool(
            mean_pass
        )
    }

    fdpass = (
        primary[
            "condensate_fd_relative"
        ]
        <= float(
            RCFG[
                "condensate_fd_relative_max"
            ]
        )
    )

    thermo_pass &= fdpass

    points[
        str(
            float(g)
        )
    ] = {
        "primary":primary,
        "check":check,
        "convergence":convergence,
        "condensate_identity_pass":bool(
            fdpass
        )
    }

    print(
        f"  sigma_mean={primary['sigma_mean']:+.10f} "
        f"sigma2={primary['sigma2']:.10f} "
        f"chi={primary['chi_sigma']:.10f} "
        f"Cpsi={primary['condensate']:.10f} "
        f"Cpsi_fd={primary['condensate_fd']:.10f} "
        f"fdrel={primary['condensate_fd_relative']:.3e}",
        flush=True
    )

    print(
        "  convergence:",
        convergence,
        flush=True
    )


# =====================================================================
# g=0 analytic Gaussian baseline
# =====================================================================

K = W.K

sigma2_exact = float(
    np.mean(
        1.0/K
    )
)

chi_exact = float(
    1.0/msigma_sq
)

Dfree = wilson_dense(
    L,
    mf0,
    r
)

cond_free = float(
    np.real(
        np.trace(
            np.linalg.inv(
                Dfree
            )
        )
    )/V
)

g0 = points[
    "0.0"
][
    "primary"
]

baseline_errors = {
    "sigma2": relative(
        g0[
            "sigma2"
        ],
        sigma2_exact
    ),

    "chi_sigma": relative(
        g0[
            "chi_sigma"
        ],
        chi_exact
    ),

    "condensate": relative(
        g0[
            "condensate"
        ],
        cond_free
    ),

    "sigma_mean_abs": abs(
        g0[
            "sigma_mean"
        ]
    )
}

baseline_pass = (
    baseline_errors[
        "sigma2"
    ] <= 1e-10
    and baseline_errors[
        "chi_sigma"
    ] <= 1e-10
    and baseline_errors[
        "condensate"
    ] <= 1e-10
    and baseline_errors[
        "sigma_mean_abs"
    ] <= 1e-10
)

print()
print(
    "g=0 analytic baseline:"
)

print(
    "  sigma2 exact =",
    sigma2_exact
)

print(
    "  chi exact    =",
    chi_exact
)

print(
    "  Cpsi exact   =",
    cond_free
)

print(
    "  errors       =",
    baseline_errors
)

print(
    "  BASELINE =",
    baseline_pass
)


# =====================================================================
# Nontrivial fermion-response gate
# =====================================================================

gmax = str(
    float(
        max(
            CFG[
                "couplings"
            ]
        )
    )
)

p0 = points[
    "0.0"
][
    "primary"
]

p1 = points[
    gmax
][
    "primary"
]

sigma_response = relative(
    p1[
        "sigma2"
    ],
    p0[
        "sigma2"
    ]
)

cond_response = relative(
    p1[
        "condensate"
    ],
    p0[
        "condensate"
    ]
)

response_pass = (
    sigma_response
    >= float(
        RCFG[
            "response_endpoint_relative_min"
        ]
    )
    and cond_response
    >= float(
        RCFG[
            "response_endpoint_relative_min"
        ]
    )
)

print()
print(
    f"response g=0 -> {gmax}:"
)

print(
    f"  sigma2 relative change = {sigma_response:.6f}"
)

print(
    f"  Cpsi relative change   = {cond_response:.6f}"
)

print(
    "  RESPONSE GATE =",
    response_pass
)


reference_pass = bool(
    free_pass
    and force_pass
    and quad_pass
    and thermo_pass
    and baseline_pass
    and response_pass
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

result = {
    "campaign":CFG[
        "campaign"
    ],

    "free_validation":{
        "rows":free_rows,
        "pass":bool(
            free_pass
        )
    },

    "force_validation":{
        "rows":force_rows,
        "pass":bool(
            force_pass
        )
    },

    "quadrature_points":points,

    "quadrature_pass":bool(
        quad_pass
    ),

    "condensate_identity_pass":bool(
        thermo_pass
    ),

    "gaussian_baseline":{
        "sigma2_exact":sigma2_exact,
        "chi_sigma_exact":chi_exact,
        "condensate_exact":cond_free,
        "errors":baseline_errors,
        "pass":bool(
            baseline_pass
        )
    },

    "response":{
        "sigma2_endpoint_relative":sigma_response,
        "condensate_endpoint_relative":cond_response,
        "pass":bool(
            response_pass
        )
    },

    "reference_pass":bool(
        reference_pass
    )
}

(
    OUT
    / "reference.json"
).write_text(
    json.dumps(
        result,
        indent=2,
        sort_keys=True
    )
    + "\n"
)

print()
print(
    "P5 FREE DETERMINANT GATE =",
    free_pass
)

print(
    "P5 FORCE GATE =",
    force_pass
)

print(
    "P5 QUADRATURE GATE =",
    quad_pass
)

print(
    "P5 CONDENSATE IDENTITY GATE =",
    thermo_pass
)

print(
    "P5 GAUSSIAN BASELINE GATE =",
    baseline_pass
)

print(
    "P5 FERMION RESPONSE GATE =",
    response_pass
)

print()
print(
    "P5 DETERMINISTIC FULL-MEASURE REFERENCE GATE =",
    reference_pass
)
