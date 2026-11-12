#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2


ROOT=Path(__file__).resolve().parents[1]

CFG=json.loads(
    (
        ROOT
        /"configs"
        /"p3_reference_kernel_invariance_v1.json"
    ).read_text()
)

BASE=(
    ROOT
    /"results"
    /"p3_reference_kernel_invariance_v1"
)

RAW=BASE/"raw"
OUT=BASE/"analysis"


def mtag(x):
    return (
        str(
            float(x)
        )
        .replace(
            ".",
            "p"
        )
    )


def aggregate(
    ds,
    key
):
    vals=np.asarray(
        [
            d[
                "estimates"
            ][key][
                "value"
            ]
            for d in ds
        ],
        float
    )

    within=np.asarray(
        [
            d[
                "estimates"
            ][key][
                "jk_se"
            ]
            for d in ds
        ],
        float
    )

    mean=float(
        np.mean(vals)
    )

    between=float(
        np.std(
            vals,
            ddof=1
        )
        / math.sqrt(
            len(vals)
        )
    )

    within_comb=float(
        np.sqrt(
            np.sum(
                within*within
            )
        )
        / len(
            within
        )
    )

    return (
        mean,
        max(
            between,
            within_comb,
            1e-14
        )
    )


# =====================================================================
# ARTIFACT GATE
# =====================================================================

expected=set()

for L in CFG[
    "lattice_sizes"
]:
    for mref in CFG[
        "m_ref_sq_values"
    ]:
        for seed in CFG[
            "seeds"
        ]:
            expected.add(
                RAW
                / (
                    f"L{L}_"
                    f"mref{mtag(mref)}_"
                    f"seed{seed}.json"
                )
            )

actual=set(
    RAW.glob(
        "*.json"
    )
)

artifact_pass=(
    expected==actual
)

print(
    "expected artifacts =",
    len(expected)
)

print(
    "actual artifacts   =",
    len(actual)
)

print(
    "ARTIFACT GATE =",
    artifact_pass
)

if not artifact_pass:
    print(
        "missing =",
        sorted(
            expected-actual
        )
    )

    print(
        "extra =",
        sorted(
            actual-expected
        )
    )

    raise SystemExit(
        2
    )


# =====================================================================
# READ GROUPS
# =====================================================================

groups={}

for L in CFG[
    "lattice_sizes"
]:
    groups[L]={}

    for mref in CFG[
        "m_ref_sq_values"
    ]:
        ds=[]

        for seed in CFG[
            "seeds"
        ]:
            p=(
                RAW
                / (
                    f"L{L}_"
                    f"mref{mtag(mref)}_"
                    f"seed{seed}.json"
                )
            )

            ds.append(
                json.loads(
                    p.read_text()
                )
            )

        groups[L][float(mref)]=ds


# =====================================================================
# SAMPLER GATE
# =====================================================================

print()
print("==============================================================")
print("P3 SAMPLER GATE")
print("==============================================================")

sg=CFG[
    "sampler_gates"
]

sampler_pass=True
sampler_rows=[]

for L in CFG[
    "lattice_sizes"
]:
    for mref in CFG[
        "m_ref_sq_values"
    ]:
        ds=groups[
            L
        ][
            float(mref)
        ]

        acc=np.asarray(
            [
                d[
                    "diagnostics"
                ][
                    "acceptance"
                ]
                for d in ds
            ],
            float
        )

        neff=np.asarray(
            [
                d[
                    "diagnostics"
                ][
                    "Neff_worst"
                ]
                for d in ds
            ],
            float
        )

        nf=np.asarray(
            [
                d[
                    "diagnostics"
                ][
                    "nonfinite_production"
                ]
                for d in ds
            ],
            int
        )

        eps=np.asarray(
            [
                d[
                    "diagnostics"
                ][
                    "mean_actual_step_size"
                ]
                for d in ds
            ],
            float
        )

        steps=np.asarray(
            [
                d[
                    "diagnostics"
                ][
                    "mean_leapfrog_steps"
                ]
                for d in ds
            ],
            float
        )

        epg=np.asarray(
            [
                d[
                    "diagnostics"
                ][
                    "ESS_per_1000_grad"
                ]
                for d in ds
            ],
            float
        )

        passed=(
            np.min(acc)
            >= float(
                sg[
                    "acceptance_min"
                ]
            )
            and np.max(acc)
            <= float(
                sg[
                    "acceptance_max"
                ]
            )
            and np.mean(neff)
            >= float(
                sg[
                    "group_mean_Neff_min"
                ]
            )
            and np.min(neff)
            >= float(
                sg[
                    "individual_Neff_min"
                ]
            )
            and np.max(nf)
            <= int(
                sg[
                    "nonfinite_max"
                ]
            )
        )

        sampler_pass &= passed

        row={
            "L":int(L),
            "m_ref_sq":float(mref),

            "acceptance_mean":float(
                np.mean(acc)
            ),

            "acceptance_min":float(
                np.min(acc)
            ),

            "Neff_mean":float(
                np.mean(neff)
            ),

            "Neff_min":float(
                np.min(neff)
            ),

            "step_size_mean":float(
                np.mean(eps)
            ),

            "steps_mean":float(
                np.mean(steps)
            ),

            "ESS_per_1000_grad_mean":float(
                np.mean(epg)
            ),

            "pass":bool(
                passed
            )
        }

        sampler_rows.append(
            row
        )

        print(
            f"L={L:2d} "
            f"mref={mref:5.2f}: "
            f"acc={np.mean(acc):.3f} "
            f"[min={np.min(acc):.3f}] "
            f"Neff={np.mean(neff):.1f} "
            f"[min={np.min(neff):.1f}] "
            f"eps={np.mean(eps):.5f} "
            f"steps={np.mean(steps):.1f} "
            f"ESS/1kgrad={np.mean(epg):.4f} "
            f"{'PASS' if passed else 'FAIL'}"
        )

print()
print(
    "GLOBAL SAMPLER GATE =",
    sampler_pass
)


# =====================================================================
# PHYSICAL INVARIANCE
# =====================================================================

print()
print("==============================================================")
print("P3-A PHYSICAL REFERENCE-KERNEL INVARIANCE")
print("==============================================================")

pg=CFG[
    "physical_invariance_gates"
]

obs=CFG[
    "primary_observables"
]

physics_pass=True
physics_rows=[]

for L in CFG[
    "lattice_sizes"
]:
    print()
    print(
        f"---------------- L={L} ----------------"
    )

    x=np.log(
        np.asarray(
            CFG[
                "m_ref_sq_values"
            ],
            float
        )
    )

    for key in obs:
        means=[]
        ses=[]

        for mref in CFG[
            "m_ref_sq_values"
        ]:
            m,e=aggregate(
                groups[
                    L
                ][
                    float(mref)
                ],
                key
            )

            means.append(m)
            ses.append(e)

        y=np.asarray(
            means,
            float
        )

        se=np.asarray(
            ses,
            float
        )

        w=1.0/(se*se)

        mu=float(
            np.sum(
                w*y
            )
            / np.sum(w)
        )

        var_mu=float(
            1.0
            / np.sum(w)
        )

        chi2_value=float(
            np.sum(
                (
                    y-mu
                )**2
                / (
                    se*se
                )
            )
        )

        dof=(
            len(y)-1
        )

        pval=float(
            chi2.sf(
                chi2_value,
                dof
            )
        )

        standardized=np.abs(
            y-mu
        )/np.sqrt(
            se*se
            + var_mu
        )

        max_resid=float(
            np.max(
                standardized
            )
        )

        # ------------------------------------------------------
        # Weighted y = a + b log(mref).
        # ------------------------------------------------------

        X=np.column_stack(
            [
                np.ones_like(x),
                x
            ]
        )

        XtW=(
            X.T*w
        )

        cov=np.linalg.inv(
            XtW@X
        )

        beta=cov@(
            XtW@y
        )

        slope=float(
            beta[1]
        )

        slope_se=float(
            np.sqrt(
                cov[
                    1,
                    1
                ]
            )
        )

        slope_z=float(
            abs(
                slope
            )
            / max(
                slope_se,
                1e-30
            )
        )

        passed=(
            pval
            >= float(
                pg[
                    "chi2_p_min"
                ]
            )
            and max_resid
            <= float(
                pg[
                    "max_standardized_residual"
                ]
            )
            and slope_z
            <= float(
                pg[
                    "log_mref_slope_z_max"
                ]
            )
        )

        physics_pass &= passed

        physics_rows.append({
            "L":int(L),
            "observable":key,

            "constant_fit":mu,

            "chi2":chi2_value,

            "dof":int(dof),

            "p_value":pval,

            "max_standardized_residual":max_resid,

            "log_mref_slope":slope,

            "log_mref_slope_se":slope_se,

            "log_mref_slope_z":slope_z,

            "pass":bool(
                passed
            ),

            "values":{
                str(m):{
                    "mean":float(v),
                    "se":float(e)
                }
                for m,v,e
                in zip(
                    CFG[
                        "m_ref_sq_values"
                    ],
                    y,
                    se
                )
            }
        })

        print(
            f"{key:12s}: "
            f"const={mu:+.7g} "
            f"chi2/dof="
            f"{chi2_value:.2f}/{dof} "
            f"p={pval:.3e} "
            f"maxres={max_resid:.2f} "
            f"slope_z={slope_z:.2f} "
            f"{'PASS' if passed else 'FAIL'}"
        )

print()
print(
    "P3-A PHYSICAL INVARIANCE GATE =",
    physics_pass
)


# =====================================================================
# P3-B ALGORITHMIC SENSITIVITY
# =====================================================================

print()
print("==============================================================")
print("P3-B HMC GEOMETRY / PERFORMANCE SENSITIVITY")
print("==============================================================")

ag=CFG[
    "algorithmic_sensitivity_gates"
]

performance_pass=False
performance_rows=[]

for L in CFG[
    "lattice_sizes"
]:
    rr=[
        r
        for r in sampler_rows
        if r[
            "L"
        ]==L
    ]

    vals=np.asarray(
        [
            r[
                "ESS_per_1000_grad_mean"
            ]
            for r in rr
        ],
        float
    )

    ratio=float(
        np.max(vals)
        / max(
            np.min(vals),
            1e-30
        )
    )

    best=rr[
        int(
            np.argmax(vals)
        )
    ][
        "m_ref_sq"
    ]

    worst=rr[
        int(
            np.argmin(vals)
        )
    ][
        "m_ref_sq"
    ]

    passed=(
        ratio
        >= float(
            ag[
                "ESS_per_grad_ratio_min"
            ]
        )
    )

    performance_pass |= passed

    performance_rows.append({
        "L":int(L),

        "ESS_per_grad_ratio":ratio,

        "best_m_ref_sq":float(
            best
        ),

        "worst_m_ref_sq":float(
            worst
        ),

        "pass":bool(
            passed
        )
    })

    print(
        f"L={L:2d}: "
        f"max/min ESS-per-grad={ratio:.3f} "
        f"best mref={best:g} "
        f"worst mref={worst:g} "
        f"{'PASS' if passed else 'WEAK'}"
    )

print()
print(
    "P3-B PERFORMANCE-SENSITIVITY GATE =",
    performance_pass
)


# =====================================================================
# FINAL
# =====================================================================

P3_A=bool(
    sampler_pass
    and physics_pass
)

P3_closed=bool(
    P3_A
)

if P3_closed:
    if performance_pass:
        classification=(
            "REFERENCE_KERNEL_INVARIANT_WITH_PRECONDITIONING_EFFECT"
        )

        interpretation=(
            "Changing the positive Gaussian reference kernel leaves "
            "the measured physical 2d phi4 observables invariant "
            "within preregistered statistical tolerances while "
            "materially changing latent-HMC efficiency."
        )

    else:
        classification=(
            "REFERENCE_KERNEL_INVARIANT_PERFORMANCE_EFFECT_WEAK"
        )

        interpretation=(
            "Changing the positive Gaussian reference kernel leaves "
            "the measured physical 2d phi4 observables invariant. "
            "The present campaign does not establish a strong "
            "reference-kernel dependence of HMC efficiency."
        )

else:
    classification=(
        "P3_NOT_CLOSED"
    )

    interpretation=(
        "At least one sampler or physical-invariance gate failed. "
        "Reference-kernel invariance is not promoted from this "
        "campaign."
    )

print()
print("==============================================================")
print("FINAL P3 VERDICT")
print("==============================================================")

print(
    "sampler_pass =",
    sampler_pass
)

print(
    "P3_A_physical_invariance =",
    physics_pass
)

print(
    "P3_B_performance_sensitivity =",
    performance_pass
)

print()
print(
    "P3_closed =",
    P3_closed
)

print(
    "P3_classification =",
    classification
)

print()
print(
    interpretation
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

verdict={
    "campaign":CFG[
        "campaign"
    ],

    "sampler_pass":bool(
        sampler_pass
    ),

    "physical_invariance_pass":bool(
        physics_pass
    ),

    "performance_sensitivity_pass":bool(
        performance_pass
    ),

    "P3_closed":bool(
        P3_closed
    ),

    "classification":classification,

    "interpretation":interpretation,

    "sampler_rows":sampler_rows,

    "physics_rows":physics_rows,

    "performance_rows":performance_rows
}

(
    OUT
    /"verdict.json"
).write_text(
    json.dumps(
        verdict,
        indent=2,
        sort_keys=True
    )
    +"\n"
)
