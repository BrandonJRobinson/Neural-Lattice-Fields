"""Latent-HMC frequency analysis for the project's configured sampler (derived expectation, not a measurement).

Latent theta with phi = K^-1/2 theta, K = phat^2 + mref^2.  For a field mode of (renormalised) curvature
k(p) = phat^2 + m^2 the latent Hamiltonian frequency is omega(p) = sqrt(k(p)/(phat^2 + mref^2)).
With full momentum refresh and trajectory length T, a harmonic mode has per-trajectory correlation
cos(omega T) for theta and cos^2(omega T) for theta^2; tau_int = (1+c)/(2(1-c)).
Configured: mref^2 = 0.5, eps = 0.025, n_leapfrog uniform in [8,16] -> T in [0.2, 0.4].
"""
import json, numpy as np
out = {}
for L, z in [(12, 3.02), (16, 2.96), (24, 3.08), (32, 2.96)]:
    m2 = (z / L) ** 2
    pmin2 = 4 * np.sin(np.pi / L) ** 2
    row = {}
    for name, k, kref in [("zero_mode_collective", m2, 0.5), ("p_min_mode", pmin2 + m2, pmin2 + 0.5)]:
        om = np.sqrt(k / kref)
        for T in (0.2, 0.3, 0.4):
            c = np.cos(om * T) ** 2
            row[f"{name}_T{T}"] = {"omega": om, "tau_int_theta2": (1 + c) / (2 * (1 - c))}
    out[str(L)] = row
json.dump(out, open("data/hmc_mismatch.json", "w"), indent=1)
for L in out:
    r = out[L]
    print(L, "omega0=%.3f tau(T=.3)=%.0f | omega_pmin=%.3f tau(T=.3)=%.0f" % (
        r["zero_mode_collective_T0.3"]["omega"], r["zero_mode_collective_T0.3"]["tau_int_theta2"],
        r["p_min_mode_T0.3"]["omega"], r["p_min_mode_T0.3"]["tau_int_theta2"]))
