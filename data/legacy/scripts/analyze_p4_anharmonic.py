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
        / "p4_anharmonic_oscillator_v1.json"
    ).read_text()
)

BASE = (
    ROOT
    / "results"
    / "p4_anharmonic_oscillator_v1"
)

RAW = BASE/"raw"
REF = BASE/"reference"
OUT = BASE/"analysis"


def aggregate(
    ds,
    key
):
    vals = np.asarray(
        [
            d[
                "estimates"
            ][
                key
            ][
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
            ][
                key
            ][
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

for Nt in CFG[
    "euclidean"
][
    "Nt_values"
]:
    for seed in CFG[
        "seeds"
    ]:
        expected.add(
            RAW
            / f"Nt{Nt}_seed{seed}.json"
        )

actual = set(
    RAW.glob(
        "*.json"
    )
)

artifact_pass = (
    expected == actual
)

print(
    "expected raw artifacts =",
    len(expected)
)

print(
    "actual raw artifacts   =",
    len(actual)
)

print(
    "RAW ARTIFACT GATE =",
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
        "missing deterministic reference.json"
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
    "DETERMINISTIC REFERENCE GATE =",
    reference_pass
)


# =====================================================================
# GROUPS
# =====================================================================

groups = {}

for Nt in CFG[
    "euclidean"
][
    "Nt_values"
]:
    ds = []

    for seed in CFG[
        "seeds"
    ]:
        ds.append(
            json.loads(
                (
                    RAW
                    / f"Nt{Nt}_seed{seed}.json"
                ).read_text()
            )
        )

    groups[
        int(
            Nt
        )
    ] = ds


# =====================================================================
# SAMPLER
# =====================================================================

print()
print("==============================================================")
print("P4 SAMPLER GATE")
print("==============================================================")

sg = CFG[
    "sampler_gates"
]

sampler_pass = True
sampler_rows = []

for Nt in CFG[
    "euclidean"
][
    "Nt_values"
]:
    ds = groups[
        int(
            Nt
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
        "Nt":int(
            Nt
        ),

        "acceptance_mean":float(
            np.mean(
                acc
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
        f"Nt={Nt:3d}: "
        f"acc={np.mean(acc):.3f} "
        f"[{np.min(acc):.3f},{np.max(acc):.3f}] "
        f"tau={np.mean(tau):.2f} "
        f"Neff={np.mean(neff):.1f} "
        f"min={np.min(neff):.1f} "
        f"ESS/1kgrad={np.mean(epg):.4f} "
        f"{'PASS' if passed else 'FAIL'}"
    )

print()
print(
    "P4 GLOBAL SAMPLER GATE =",
    sampler_pass
)


# =====================================================================
# FINITE-LATTICE HMC VS TRANSFER MATRIX
# =====================================================================

print()
print("==============================================================")
print("P4 FINITE-LATTICE EXACT BENCHMARK")
print("==============================================================")

mg = CFG[
    "finite_lattice_match_gates"
]

primary = CFG[
    "primary_observables"
]

physics_pass = True
physics_rows = []

global_by_obs = {
    key: []
    for key in primary
}

for Nt in CFG[
    "euclidean"
][
    "Nt_values"
]:
    ds = groups[
        int(
            Nt
        )
    ]

    rr = reference[
        "transfer"
    ][
        str(
            Nt
        )
    ]

    print()
    print(
        f"---------------- Nt={Nt} ----------------"
    )

    for key in primary:
        mean,se = aggregate(
            ds,
            key
        )

        refkey = (
            key
        )

        ref = float(
            rr[
                refkey
            ]
        )

        z = abs(
            mean-ref
        )/se

        rel = abs(
            mean-ref
        )/max(
            abs(
                ref
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

        global_by_obs[
            key
        ].append(
            (
                mean,
                se,
                ref
            )
        )

        physics_rows.append({
            "Nt":int(
                Nt
            ),

            "observable":key,

            "hmc":mean,

            "se":se,

            "reference":ref,

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
            f"{key:8s}: "
            f"HMC={mean:.10f} +/- {se:.3g} "
            f"REF={ref:.10f} "
            f"z={z:.2f} "
            f"rel={rel:.3e} "
            f"{'PASS' if passed else 'FAIL'}"
        )


# =====================================================================
# GLOBAL CHI-SQUARE ACROSS Nt
# =====================================================================

print()
print("==============================================================")
print("GLOBAL FINITE-LATTICE CONSISTENCY")
print("==============================================================")

global_pass = True
global_rows = []

for key in primary:
    vals = global_by_obs[
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
        f"{key:8s}: "
        f"chi2/dof={c2:.3f}/{dof} "
        f"p={p:.3e} "
        f"{'PASS' if passed else 'FAIL'}"
    )

physics_pass &= global_pass

print()
print(
    "P4 FINITE-LATTICE PHYSICS GATE =",
    physics_pass
)


# =====================================================================
# FINAL
# =====================================================================

P4_closed = bool(
    artifact_pass
    and reference_pass
    and sampler_pass
    and physics_pass
)

if P4_closed:
    classification = (
        "ANHARMONIC_OSCILLATOR_EXACT_SPECTRAL_BENCHMARK_PASS"
    )

    interpretation = (
        "Exact latent factorization reproduces the finite-lattice "
        "quartic-oscillator moments and Euclidean spectral gap "
        "obtained from an independent deterministic transfer-matrix "
        "calculation, whose continuum limit independently agrees "
        "with Hamiltonian diagonalization."
    )

else:
    classification = (
        "P4_NOT_CLOSED"
    )

    interpretation = (
        "At least one deterministic-reference, sampler, or "
        "finite-lattice physics gate failed."
    )

print()
print("==============================================================")
print("FINAL P4 VERDICT")
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
    "finite_lattice_physics_pass =",
    physics_pass
)

print()
print(
    "P4_closed =",
    P4_closed
)

print(
    "P4_classification =",
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

            "finite_lattice_physics_pass":bool(
                physics_pass
            ),

            "P4_closed":bool(
                P4_closed
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
