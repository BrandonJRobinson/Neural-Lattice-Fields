#!/usr/bin/env python3

import numpy as np

from p4_anharmonic_common import (
    LatentOscillator,
    physical_lattice_action
)


rng = np.random.default_rng(
    20260909
)

beta = 12.0
omega = 1.0
g = 1.0

for Nt in [
    16,
    32,
    64
]:
    model = LatentOscillator(
        Nt=Nt,
        beta=beta,
        omega=omega,
        g=g,
        omega_ref=1.0
    )

    # ----------------------------------------------------------
    # Exact field-space / latent-space action identity.
    # ----------------------------------------------------------

    x = (
        0.2
        * rng.standard_normal(
            Nt
        )
    )

    theta = model.W.sqrt_apply(
        x
    )

    Slat = model.action(
        theta
    )

    Sphys = physical_lattice_action(
        x,
        beta=beta,
        omega=omega,
        g=g
    )

    err = abs(
        Slat-Sphys
    )

    print(
        f"Nt={Nt}: "
        f"S_lat={Slat:.14e} "
        f"S_phys={Sphys:.14e} "
        f"diff={err:.3e}"
    )

    if err > 1e-9:
        raise SystemExit(
            "action identity failed"
        )

    # ----------------------------------------------------------
    # Directional gradient.
    # ----------------------------------------------------------

    theta = (
        0.2
        * rng.standard_normal(
            Nt
        )
    )

    direction = rng.standard_normal(
        Nt
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
    )/(2*eps)

    exact = float(
        np.dot(
            grad,
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

    print(
        f"  gradient fd={fd:.12e} "
        f"exact={exact:.12e} "
        f"rel={rel:.3e}"
    )

    if rel > 1e-7:
        raise SystemExit(
            "gradient identity failed"
        )

print(
    "P4 EXACT LATENT ACTION IDENTITY: PASS"
)

print(
    "P4 GRADIENT CHECK: PASS"
)
