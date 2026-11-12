"""
Module: spectral_prior_architectures.py

Spectral prior families for CosRFF on the periodic torus.
Level A of the architecture-improvement campaign.

Investigates whether changing the spectral prior on frequency vectors
k_a can materially improve the NNGP calibration ceiling against the
free lattice propagator G(p) = 1/(m² + p̂²).

Architectures (all use the same torus embedding and cos activation):
  1. CosRFF_StudentT — isotropic Student-t (Matérn-like NNGP kernel)
  2. CosRFF_GammaMix — Gamma scale-mixture (rational NNGP kernel)
  3. CosRFF_TwoScale — two-component Gaussian mixture

Key physics:
  The Gaussian CosRFF kernel is K(Δ) = (σ_c²/2) exp(-σ_k² d²), which
  is exponential in d² and therefore too smooth at UV to match the
  algebraic free propagator G(p) ~ 1/p̂².

  Heavy-tailed priors produce algebraic/Matérn position-space kernels
  whose Fourier transforms have more UV power, potentially reducing
  the structural calibration ceiling.
"""

import numpy as np
from scipy.special import kv as bessel_kv
from scipy.special import gamma as gamma_fn
from scipy.optimize import differential_evolution

from lattice_nnft.nonlinear_architectures import (
    CosRFF, periodic_embedding, free_propagator_mom, free_propagator_pos,
)
from lattice_nnft.kernel_recalibration import compute_mismatch_suite


# ==================================================================
# Utility
# ==================================================================

def _d2_grid(L):
    """d²(Δi,Δj) = 2 - cos(2πΔi/L) - cos(2πΔj/L), shape (L,L)."""
    phase = 2 * np.pi * np.arange(L) / L
    return 2.0 - np.cos(phase)[:, None] - np.cos(phase)[None, :]


def _rescale_sigma_c(K_mom_raw, G_mom):
    """Optimal σ_c given raw kernel (σ_c=1)."""
    alpha = np.sum(K_mom_raw * G_mom) / (np.sum(K_mom_raw ** 2) + 1e-30)
    return np.sqrt(max(alpha, 1e-10))


def kinetic_mismatch(K_mom, G_mom, L):
    """Kinetic-energy-weighted relative mismatch.

    (Σ_p p̂² |K̂(p) - Ĝ(p)|) / (Σ_p p̂² Ĝ(p))

    UV modes weighted by p̂², so this is sensitive to high-momentum
    shape errors that affect derivative-sensitive observables.
    """
    n = np.arange(L)
    p = 2 * np.pi * n / L
    ph2 = 4 * np.sin(p / 2) ** 2
    phat2 = ph2[:, None] + ph2[None, :]
    delta_kin = np.sum(phat2 * np.abs(K_mom - G_mom))
    target_kin = np.sum(phat2 * G_mom)
    return float(delta_kin / (target_kin + 1e-30))


# ==================================================================
# Architecture: Student-t (isotropic) — Matérn-like kernel
# ==================================================================

class CosRFF_StudentT:
    """
    CosRFF with isotropic Student-t prior on frequency vectors.

    Sampling: k = σ₀ · z · √(ν/v), z ~ N(0,I₄), v ~ χ²_ν (shared).

    NNGP kernel (closed-form via Bessel K):
      K(Δ) = (σ_c²/2) · z^{ν/2} K_{ν/2}(z) / [2^{ν/2-1} Γ(ν/2)]
      where z = √(2 σ₀² ν d²).

    Special cases:
      ν=1 (Cauchy): K ∝ exp(-σ₀ √(2d²))     [exponential in √d²]
      ν→∞:         K → (σ_c²/2) exp(-σ₀² d²) [recovers Gaussian]
    """

    def __init__(self, L, sigma_0, nu, sigma_c):
        self.L = L
        self.V = L * L
        self.sigma_0 = sigma_0
        self.nu = nu
        self.sigma_c = sigma_c
        self.embed = periodic_embedding(L)
        self.name = f'StudentT_nu{nu}'

    def kernel_pos(self):
        d2 = _d2_grid(self.L)
        z = np.sqrt(np.maximum(2 * self.sigma_0 ** 2 * self.nu * d2, 0))
        nu_half = self.nu / 2.0

        result = np.ones_like(d2)
        mask = z > 1e-12
        zm = z[mask]
        # z^{p} K_p(z) / [2^{p-1} Γ(p)] → 1 as z → 0
        result[mask] = (
            zm ** nu_half * bessel_kv(nu_half, zm)
            / (2 ** (nu_half - 1) * gamma_fn(nu_half))
        )
        return (self.sigma_c ** 2 / 2) * result

    def kernel_mom(self):
        return np.real(np.fft.fft2(self.kernel_pos()))

    def generate(self, N, M, batch_size=200, seed=42):
        rng = np.random.default_rng(seed)
        configs = np.empty((M, self.V))
        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            bs = end - start
            z = rng.normal(0, 1.0, (bs, N, 4))
            v = rng.chisquare(self.nu, (bs, N))
            scale = self.sigma_0 * np.sqrt(self.nu / np.maximum(v, 1e-30))
            k = z * scale[:, :, None]
            b = rng.uniform(0, 2 * np.pi, (bs, N, 1))
            pre = np.einsum('bnd,vd->bnv', k, self.embed) + b
            configs[start:end] = (
                self.sigma_c * np.sum(np.cos(pre), axis=1) / np.sqrt(N)
            )
        return configs


# ==================================================================
# Architecture: Gamma scale mixture — rational kernel
# ==================================================================

class CosRFF_GammaMix:
    """
    CosRFF with Gamma-distributed variance for frequency vectors.

    Sampling: τ ~ Gamma(α, rate=λ), then k ~ N(0, τ I₄).

    NNGP kernel (closed-form, rational):
      K(Δ) = (σ_c²/2) · (λ / (λ + d²))^α

    For large d²: K ~ d^{-2α} (algebraic/power-law decay).
    Parameters: α = smoothness, λ = length scale.
    """

    def __init__(self, L, alpha, lam, sigma_c):
        self.L = L
        self.V = L * L
        self.alpha = alpha
        self.lam = lam
        self.sigma_c = sigma_c
        self.embed = periodic_embedding(L)
        self.name = f'GammaMix_a{alpha:.1f}_l{lam:.1f}'

    def kernel_pos(self):
        d2 = _d2_grid(self.L)
        return (self.sigma_c ** 2 / 2) * (self.lam / (self.lam + d2)) ** self.alpha

    def kernel_mom(self):
        return np.real(np.fft.fft2(self.kernel_pos()))

    def generate(self, N, M, batch_size=200, seed=42):
        rng = np.random.default_rng(seed)
        configs = np.empty((M, self.V))
        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            bs = end - start
            tau = rng.gamma(self.alpha, scale=1.0 / self.lam, size=(bs, N))
            z = rng.normal(0, 1.0, (bs, N, 4))
            k = z * np.sqrt(np.maximum(tau, 1e-30))[:, :, None]
            b = rng.uniform(0, 2 * np.pi, (bs, N, 1))
            pre = np.einsum('bnd,vd->bnv', k, self.embed) + b
            configs[start:end] = (
                self.sigma_c * np.sum(np.cos(pre), axis=1) / np.sqrt(N)
            )
        return configs


# ==================================================================
# Architecture: Two-scale Gaussian mixture
# ==================================================================

class CosRFF_TwoScale:
    """
    CosRFF with two-component Gaussian mixture prior on k.

    k ~ π₁ N(0, σ₁²I₄) + (1-π₁) N(0, σ₂²I₄)

    NNGP kernel (closed-form):
      K(Δ) = (σ_c²/2) · [π₁ exp(-σ₁²d²) + (1-π₁) exp(-σ₂²d²)]

    A small-σ component provides UV power; a large-σ component
    provides IR localisation.
    """

    def __init__(self, L, sigma1, sigma2, pi1, sigma_c):
        self.L = L
        self.V = L * L
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.pi1 = pi1
        self.sigma_c = sigma_c
        self.embed = periodic_embedding(L)
        self.name = 'TwoScale'

    def kernel_pos(self):
        d2 = _d2_grid(self.L)
        return (self.sigma_c ** 2 / 2) * (
            self.pi1 * np.exp(-self.sigma1 ** 2 * d2)
            + (1 - self.pi1) * np.exp(-self.sigma2 ** 2 * d2)
        )

    def kernel_mom(self):
        return np.real(np.fft.fft2(self.kernel_pos()))

    def generate(self, N, M, batch_size=200, seed=42):
        rng = np.random.default_rng(seed)
        configs = np.empty((M, self.V))
        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            bs = end - start
            choice = rng.random((bs, N)) < self.pi1
            sigma = np.where(choice, self.sigma1, self.sigma2)
            k = rng.normal(0, 1.0, (bs, N, 4)) * sigma[:, :, None]
            b = rng.uniform(0, 2 * np.pi, (bs, N, 1))
            pre = np.einsum('bnd,vd->bnv', k, self.embed) + b
            configs[start:end] = (
                self.sigma_c * np.sum(np.cos(pre), axis=1) / np.sqrt(N)
            )
        return configs


# ==================================================================
# Universal calibration
# ==================================================================

def calibrate_prior(L, m2, make_arch, param_bounds,
                    maxiter=400, seed=42):
    """
    Calibrate any CosRFF-style architecture against the free propagator.

    make_arch(params) → arch with σ_c=1, having .kernel_mom().
    Optimal σ_c found by amplitude rescaling.
    Objective: L²_rel after optimal rescale (same as strategy D).

    Returns dict with optimised params, calibrated kernels, metrics.
    """
    G_mom = free_propagator_mom(L, m2)
    G_pos = free_propagator_pos(L, m2)

    def _obj(params):
        try:
            arch = make_arch(params)
            K_mom = arch.kernel_mom()
            if np.any(np.isnan(K_mom)) or np.any(np.isinf(K_mom)):
                return 1e10
            alpha = np.sum(K_mom * G_mom) / (np.sum(K_mom ** 2) + 1e-30)
            if alpha < 1e-10:
                return 1e10
            K_s = K_mom * alpha
            rel = (K_s - G_mom) / (np.abs(G_mom) + 1e-30)
            return float(np.sqrt(np.mean(rel ** 2)))
        except Exception:
            return 1e10

    result = differential_evolution(
        _obj, param_bounds, seed=seed, maxiter=maxiter,
        tol=1e-12, polish=True,
    )

    # Build calibrated kernel
    arch_raw = make_arch(result.x)
    K_mom_raw = arch_raw.kernel_mom()
    sc = _rescale_sigma_c(K_mom_raw, G_mom)
    K_mom_cal = K_mom_raw * sc ** 2
    K_pos_cal = np.real(np.fft.ifft2(K_mom_cal))

    # Full mismatch diagnostics
    metrics = compute_mismatch_suite(K_mom_cal, K_pos_cal, G_mom, G_pos, L)
    metrics['kinetic_mismatch'] = kinetic_mismatch(K_mom_cal, G_mom, L)

    # Shape decomposition
    K_shape = K_mom_cal / (np.sqrt(np.sum(K_mom_cal ** 2)) + 1e-30)
    G_shape = G_mom / (np.sqrt(np.sum(G_mom ** 2)) + 1e-30)
    metrics['shape_mismatch'] = float(np.sum((K_shape - G_shape) ** 2))

    return {
        'opt_params': list(result.x),
        'sigma_c': float(sc),
        'optimizer_fun': float(result.fun),
        'metrics': metrics,
        'K_mom': K_mom_cal,
        'K_pos': K_pos_cal,
        'G_mom': G_mom,
        'G_pos': G_pos,
    }


def calibrate_all_priors(L, m2):
    """
    Calibrate all spectral prior families against the target.

    Tests: Gaussian (control), Student-t (ν=1,2,4,8),
           Gamma scale mixture, two-scale Gaussian mixture.

    Returns dict keyed by prior name → calibration results.
    """
    G_mom = free_propagator_mom(L, m2)
    G_pos = free_propagator_pos(L, m2)
    results = {}

    # --- Gaussian control ---
    from lattice_nnft.kernel_recalibration_rff import calibrate_rff_strategy
    gauss = calibrate_rff_strategy(L, m2, 'D')
    g_met = gauss['metrics']
    g_met['kinetic_mismatch'] = kinetic_mismatch(gauss['K_mom'], G_mom, L)
    results['Gaussian'] = {
        'family': 'Gaussian',
        'opt_params': [gauss['params']['sigma_k']],
        'sigma_c': gauss['params']['sigma_c'],
        'metrics': g_met,
        'K_mom': gauss['K_mom'],
        'K_pos': gauss['K_pos'],
        'G_mom': G_mom,
        'G_pos': G_pos,
        'arch': CosRFF(L, gauss['params']['sigma_k'],
                       gauss['params']['sigma_c']),
    }

    # --- Student-t: ν ∈ {1, 2, 4, 8} ---
    for nu in [1, 2, 4, 8]:
        name = f'StudentT_nu{nu}'

        def _make(params, _nu=nu):
            return CosRFF_StudentT(L, params[0], _nu, 1.0)

        cal = calibrate_prior(L, m2, _make, [(0.01, 25.0)],
                              maxiter=400, seed=42)
        s0 = cal['opt_params'][0]
        sc = cal['sigma_c']
        cal['family'] = name
        cal['arch'] = CosRFF_StudentT(L, s0, nu, sc)
        results[name] = cal

    # --- Gamma scale mixture ---
    def _make_gamma(params):
        return CosRFF_GammaMix(L, params[0], params[1], 1.0)

    cal = calibrate_prior(
        L, m2, _make_gamma,
        [(0.3, 15.0), (0.1, 50.0)],
        maxiter=500, seed=42,
    )
    a_opt, l_opt = cal['opt_params']
    sc = cal['sigma_c']
    cal['family'] = 'GammaMix'
    cal['arch'] = CosRFF_GammaMix(L, a_opt, l_opt, sc)
    results['GammaMix'] = cal

    # --- Two-scale Gaussian mixture ---
    def _make_twoscale(params):
        return CosRFF_TwoScale(L, params[0], params[1], params[2], 1.0)

    cal = calibrate_prior(
        L, m2, _make_twoscale,
        [(0.1, 25.0), (0.1, 25.0), (0.05, 0.95)],
        maxiter=500, seed=42,
    )
    s1, s2, pi1 = cal['opt_params']
    sc = cal['sigma_c']
    cal['family'] = 'TwoScale'
    cal['arch'] = CosRFF_TwoScale(L, s1, s2, pi1, sc)
    results['TwoScale'] = cal

    return results
