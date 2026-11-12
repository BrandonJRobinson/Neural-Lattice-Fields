#!/usr/bin/env python3

import importlib.util
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[1]

p=ROOT/"scripts"/"run_p3_reference_kernel_invariance_rescue.py"

spec=importlib.util.spec_from_file_location(
    "p3run",
    p
)

m=importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    m
)

rng=np.random.default_rng(
    20260909
)

L=8
m0=-3.84
lam=1.0

for mref in [
    .25,
    .5,
    1.0,
    2.0,
    4.0,
    8.0
]:
    model=m.LatentPhi4(
        L,
        m0,
        lam,
        mref
    )

    phi=0.25*rng.standard_normal(
        (L,L)
    )

    theta=model.W.apply_sqrt(
        phi
    )

    Slat=model.action(
        theta
    )

    Sphys=m.physical_action(
        phi,
        m0_sq=m0,
        lam=lam
    )

    err=abs(
        Slat-Sphys
    )

    print(
        f"mref={mref:g}: "
        f"S_lat={Slat:.12e} "
        f"S_phys={Sphys:.12e} "
        f"|diff|={err:.3e}"
    )

    if err > 5e-10:
        raise SystemExit(
            "exact action identity failed"
        )

    # ----------------------------------------------------------
    # Directional finite-difference gradient.
    # ----------------------------------------------------------

    theta=0.15*rng.standard_normal(
        (L,L)
    )

    direction=rng.standard_normal(
        (L,L)
    )

    direction /= np.linalg.norm(
        direction
    )

    S,g=model.action_grad(
        theta
    )

    eps=1e-6

    fd=(
        model.action(
            theta+eps*direction
        )
        - model.action(
            theta-eps*direction
        )
    )/(2*eps)

    exact=float(
        np.sum(
            g*direction
        )
    )

    rel=abs(
        fd-exact
    )/max(
        1.0,
        abs(fd),
        abs(exact)
    )

    print(
        f"  gradient fd={fd:.12e} "
        f"exact={exact:.12e} "
        f"rel={rel:.3e}"
    )

    if rel > 1e-7:
        raise SystemExit(
            "gradient smoke failed"
        )

print(
    "P3 EXACT ACTION IDENTITY: PASS"
)

print(
    "P3 GRADIENT CHECK: PASS"
)
