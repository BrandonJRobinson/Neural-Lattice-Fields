#!/usr/bin/env python3

import numpy as np

from p5_fermionic_common import (
    LatentFermionAuxiliary
)


rng = np.random.default_rng(
    20260909
)

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
        f"g={g:.3f}: "
        f"S={S:.12e} "
        f"fd={fd:.12e} "
        f"grad={exact:.12e} "
        f"rel={rel:.3e}"
    )

    if rel > 2e-7:
        raise SystemExit(
            "latent determinant gradient failed"
        )

print(
    "P5 LATENT DETERMINANT GRADIENT: PASS"
)
