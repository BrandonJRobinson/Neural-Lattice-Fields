#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[1]

CFG = json.loads(
    (
        ROOT
        / "configs"
        / "p5_fermionic_full_measure_v1.json"
    ).read_text()
)

BASE = (
    ROOT
    / "results"
    / "p5_fermionic_full_measure_v1"
)

RAW = BASE/"raw"
REF = BASE/"reference"
OUT = BASE/"analysis"


def gtag(
    g
):
    return (
        f"{float(g):.3f}"
        .replace(
            ".",
            "p"
        )
    )


def aggregate(
    ds,
    key
):
    vals = np.asarray(
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

    ses = np.asarray(
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

    mean = float(
        np.mean(
            vals
        )
    )

    between = float(
        np.std(
            vals,
            ddof=1
        )
        / math.sqrt(
            len(vals)
        )
    )

    within = float(
        np.sqrt(
            np.sum(
                ses*ses
            )
        )
        / len(
            ses
        )
    )

    return (
        mean,
        max(
            between,
            within,
            1e-14
        )
    )


# =====================================================================
# ARTIFACTS
# =====================================================================

expected = set()

for g in CFG[
    "couplings"
]:
    for seed in CFG[
        "seeds"
    ]:
        expected.add(
            RAW
            / (
                f"g{gtag(g)}_"
                f"seed{seed}.json"
            )
        )

actual = set(
    RAW.glob(
        "*.json"
    )
)

artifact_pass = (
    expected==actual
)

print(
    "expected HMC artifacts =",
    len(expected)
)

print(
    "actual HMC artifacts   =",
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

refpath = (
    REF
    / "reference.json"
)

if not refpath.exists():
    raise SystemExit(
        "missing reference.json"
    )

reference = json.loads(
    refpath.read_text()
)

reference_pass = bool(
    reference[
        "reference_pass"
    ]
)

print()
print(
    "P5 FULL-MEASURE REFERENCE GATE =",
    reference_pass
)


# =====================================================================
# GROUPS
# =====================================================================

groups = {}

for g in CFG[
    "couplings"
]:
    ds = []

    for seed in CFG[
        "seeds"
    ]:
        ds.append(
            json.loads(
                (
                    RAW
                    / (
                        f"g{gtag(g)}_"
                        f"seed{seed}.json"
                    )
                ).read_text()
            )
        )

    groups[
        float(
            g
        )
    ] = ds


# =====================================================================
# SAMPLER
# =====================================================================

print()
print("==============================================================")
print("P5 LATENT-HMC SAMPLER GATE")
print("==============================================================")

sg = CFG[
    "sampler_gates"
]

sampler_pass = True
sampler_rows = []

for g in CFG[
    "couplings"
]:
    ds = groups[
        float(
            g
        )
    ]

    acc = np.asarray(
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

    shift = np.asarray(
        [
            d[
                "diagnostics"
            ][
                "global_shift_acceptance"
            ]
            for d in ds
        ],
        float
    )

    neff = np.asarray(
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

    tau = np.asarray(
        [
            d[
                "diagnostics"
            ][
                "tau_worst"
            ]
            for d in ds
        ],
        float
    )

    nf = np.asarray(
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

    epg = np.asarray(
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

    passed = (
        np.min(
            acc
        )
        >= float(
            sg[
                "acceptance_min"
            ]
        )
        and np.max(
            acc
        )
        <= float(
            sg[
                "acceptance_max"
            ]
        )
        and np.mean(
            neff
        )
        >= float(
            sg[
                "group_mean_Neff_min"
            ]
        )
        and np.min(
            neff
        )
        >= float(
            sg[
                "individual_Neff_min"
            ]
        )
        and np.max(
            nf
        )
        <= int(
            sg[
                "nonfinite_max"
            ]
        )
    )

    sampler_pass &= passed

    sampler_rows.append({
        "g":float(
            g
        ),

        "acceptance_mean":float(
            np.mean(
                acc
            )
        ),

        "shift_acceptance_mean":float(
            np.mean(
                shift
            )
        ),

        "Neff_mean":float(
            np.mean(
                neff
            )
        ),

        "Neff_min":float(
            np.min(
                neff
            )
        ),

        "tau_mean":float(
            np.mean(
                tau
            )
        ),

        "ESS_per_1000_grad":float(
            np.mean(
                epg
            )
        ),

        "pass":bool(
            passed
        )
    })

    print(
        f"g={g:5.3f}: "
        f"acc={np.mean(acc):.3f} "
        f"[{np.min(acc):.3f},{np.max(acc):.3f}] "
        f"shift={np.mean(shift):.3f} "
        f"tau={np.mean(tau):.2f} "
        f"Neff={np.mean(neff):.1f} "
        f"min={np.min(neff):.1f} "
        f"ESS/1kgrad={np.mean(epg):.4f} "
        f"{'PASS' if passed else 'FAIL'}"
    )

print()
print(
    "P5 GLOBAL SAMPLER GATE =",
    sampler_pass
)


# =====================================================================
# HMC VS FULL-MEASURE QUADRATURE
# =====================================================================

print()
print("==============================================================")
print("P5 HMC VS INDEPENDENT FULL-MEASURE QUADRATURE")
print("==============================================================")

mg = CFG[
    "finite_measure_match_gates"
]

primary = CFG[
    "primary_observables"
]

physics_pass = True

physics_rows = []

global_data = {
    key:[]
    for key in primary
}

for g in CFG[
    "couplings"
]:
    print()
    print(
        f"---------------- g={g:.3f} ----------------"
    )

    ds = groups[
        float(
            g
        )
    ]

    rr = reference[
        "quadrature_points"
    ][
        str(
            float(
                g
            )
        )
    ][
        "primary"
    ]

    for key in primary:
        mean,se = aggregate(
            ds,
            key
        )

        exact = float(
            rr[
                key
            ]
        )

        z = abs(
            mean-exact
        )/se

        rel = abs(
            mean-exact
        )/max(
            abs(
                exact
            ),
            1e-30
        )

        passed = (
            z
            <= float(
                mg[
                    "max_z"
                ]
            )
            and rel
            <= float(
                mg[
                    "relative_error_max"
                ][
                    key
                ]
            )
        )

        physics_pass &= passed

        global_data[
            key
        ].append(
            (
                mean,
                se,
                exact
            )
        )

        physics_rows.append({
            "g":float(
                g
            ),

            "observable":key,

            "hmc":mean,

            "se":se,

            "reference":exact,

            "z":float(
                z
            ),

            "relative_error":float(
                rel
            ),

            "pass":bool(
                passed
            )
        })

        print(
            f"{key:11s}: "
            f"HMC={mean:+.10f} +/- {se:.3g} "
            f"REF={exact:+.10f} "
            f"delta={mean-exact:+.3e} "
            f"z={z:.2f} "
            f"rel={rel:.3e} "
            f"{'PASS' if passed else 'FAIL'}"
        )


# =====================================================================
# GLOBAL CONSISTENCY
# =====================================================================

print()
print("==============================================================")
print("GLOBAL HMC / QUADRATURE CONSISTENCY")
print("==============================================================")

global_pass = True
global_rows = []

for key in primary:
    vals = global_data[
        key
    ]

    c2 = float(
        sum(
            (
                (m-r)/s
            )**2
            for m,s,r in vals
        )
    )

    dof = len(
        vals
    )

    p = float(
        chi2.sf(
            c2,
            dof
        )
    )

    passed = (
        p
        >= float(
            mg[
                "global_chi2_p_min"
            ]
        )
    )

    global_pass &= passed

    global_rows.append({
        "observable":key,

        "chi2":c2,

        "dof":dof,

        "p_value":p,

        "pass":bool(
            passed
        )
    })

    print(
        f"{key:11s}: "
        f"chi2/dof={c2:.3f}/{dof} "
        f"p={p:.3e} "
        f"{'PASS' if passed else 'FAIL'}"
    )

physics_pass &= global_pass

print()
print(
    "P5 FULL-MEASURE PHYSICS GATE =",
    physics_pass
)


# =====================================================================
# RESPONSE SUMMARY
# =====================================================================

print()
print("==============================================================")
print("FERMION-INDUCED RESPONSE")
print("==============================================================")

for g in CFG[
    "couplings"
]:
    rr = reference[
        "quadrature_points"
    ][
        str(
            float(
                g
            )
        )
    ][
        "primary"
    ]

    print(
        f"g={g:5.3f}: "
        f"<sigma>={rr['sigma_mean']:+.7f} "
        f"<sigma2>={rr['sigma2']:.7f} "
        f"chi={rr['chi_sigma']:.7f} "
        f"Cpsi={rr['condensate']:.7f}"
    )


# =====================================================================
# FINAL
# =====================================================================

P5_closed = bool(
    artifact_pass
    and reference_pass
    and sampler_pass
    and physics_pass
)

if P5_closed:
    classification = (
        "FERMIONIC_INTERACTING_FULL_MEASURE_BENCHMARK_PASS"
    )

    interpretation = (
        "Exact latent factorization with a nonlocal Wilson-fermion "
        "determinant interaction reproduces independent full-measure "
        "Gauss-Hermite results for auxiliary-field fluctuations and "
        "fermionic response across the complete Yukawa coupling scan."
    )

else:
    classification = (
        "P5_NOT_CLOSED"
    )

    interpretation = (
        "At least one determinant-reference, sampler, or interacting "
        "full-measure physics gate failed."
    )

print()
print("==============================================================")
print("FINAL P5 VERDICT")
print("==============================================================")

print(
    "reference_pass =",
    reference_pass
)

print(
    "sampler_pass =",
    sampler_pass
)

print(
    "full_measure_physics_pass =",
    physics_pass
)

print()
print(
    "P5_closed =",
    P5_closed
)

print(
    "P5_classification =",
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

(
    OUT
    / "verdict.json"
).write_text(
    json.dumps(
        {
            "campaign":CFG[
                "campaign"
            ],

            "reference_pass":bool(
                reference_pass
            ),

            "sampler_pass":bool(
                sampler_pass
            ),

            "full_measure_physics_pass":bool(
                physics_pass
            ),

            "P5_closed":bool(
                P5_closed
            ),

            "classification":classification,

            "interpretation":interpretation,

            "sampler_rows":sampler_rows,

            "physics_rows":physics_rows,

            "global_rows":global_rows
        },
        indent=2,
        sort_keys=True
    )
    + "\n"
)
