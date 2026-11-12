#!/usr/bin/env python3

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

p = (
    ROOT
    / "scripts"
    / "run_p5_fermionic_hmc_rescue.py"
)

spec = importlib.util.spec_from_file_location(
    "p5r",
    p
)

m = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    m
)

from p5_fermionic_common import LatentFermionAuxiliary


rng = np.random.default_rng(
    20260909
)

# ================================================================
# 1. Gradient identity remains exact.
# ================================================================

for g in [
    0.0,
    0.25,
    0.5
]:
    model = LatentFermionAuxiliary(
        L=2,
        Nf=2,
        mf=0.5,
        r=1.0,
        msigma_sq=0.5,
        g=g
    )

    theta = (
        0.25
        * rng.standard_normal(
            (
                2,
                2
            )
        )
    )

    direction = rng.standard_normal(
        (
            2,
            2
        )
    )

    direction /= np.linalg.norm(
        direction
    )

    S,grad = model.action_grad(
        theta
    )

    eps = 1e-6

    fd = (
        model.action(
            theta+eps*direction
        )
        - model.action(
            theta-eps*direction
        )
    )/(2.0*eps)

    exact = float(
        np.sum(
            grad*direction
        )
    )

    rel = abs(
        fd-exact
    )/max(
        1.0,
        abs(fd),
        abs(exact)
    )

    print(
        f"gradient g={g:.3f}: "
        f"rel={rel:.3e}"
    )

    if rel > 2e-7:
        raise SystemExit(
            "gradient smoke failed"
        )


# ================================================================
# 2. Deterministic tau diagnostic.
# ================================================================

constant = np.full(
    10000,
    1.511111111111111,
    dtype=float
)

# Add machine-scale perturbations deliberately.
constant += (
    np.finfo(float).eps
    * np.sin(
        np.arange(
            len(
                constant
            )
        )
    )
)

tau = m.robust_tau(
    constant
)

print(
    "deterministic-series tau =",
    tau
)

if abs(
    tau-0.5
) > 1e-15:
    raise SystemExit(
        "constant-observable tau guard failed"
    )

print(
    "P5R GRADIENT SMOKE: PASS"
)

print(
    "P5R CONSTANT-OBSERVABLE TAU GUARD: PASS"
)
