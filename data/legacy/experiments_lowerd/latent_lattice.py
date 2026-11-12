"""
Exact Gaussian latent factorization on a periodic d-dimensional lattice.

Scientific framing
-------------------
For a translation-invariant lattice operator K (diagonal in momentum
space with symbol K(p)), the free Gaussian measure

    P_free(phi)  ~  exp( -1/2 phi^T K phi )

is represented *exactly* in latent coordinates by the Fourier-space
whitening map

    phi = K^{-1/2} theta,        theta ~ N(0, I),

with  K^{-1/2} = F^{-1} diag( 1/sqrt(K(p)) ) F.

Interactions are added through the induced latent action

    S_latent(theta) = 1/2 ||theta||^2 + S_int[ phi(theta) ].

This module provides:
  * the momentum kernel K(p) for the 2d (general-d) lattice Laplacian,
  * the WhiteningOp implementing K^{-1/2} (real, self-adjoint),
  * a FreeField sampler (lambda = 0 / y = 0 ground truth),
  * a sanity check comparing the empirical momentum propagator to 1/K(p).

NumPy only, matching the existing 4d lattice_nnft code conventions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ======================================================================
# Momentum kernel
# ======================================================================

def phat2_grid(L: int, dim: int = 2) -> np.ndarray:
    """Lattice momentum-squared grid  phat^2 = 4 sum_mu sin^2(p_mu/2).

    Shape (L,)*dim.  Periodic momenta p_mu = 2 pi n_mu / L.
    """
    n = np.arange(L)
    p = 2.0 * np.pi * n / L
    phat2_1d = 4.0 * np.sin(p / 2.0) ** 2
    grids = np.meshgrid(*([phat2_1d] * dim), indexing='ij')
    out = np.zeros((L,) * dim)
    for g in grids:
        out = out + g
    return out


def kinetic_kernel(L: int, m_ref_sq: float, dim: int = 2) -> np.ndarray:
    """K(p) = phat^2 + m_ref^2.  Must be strictly positive for whitening."""
    return phat2_grid(L, dim) + float(m_ref_sq)


def free_propagator_mom(L: int, m_ref_sq: float, dim: int = 2) -> np.ndarray:
    """Analytic free momentum propagator  G(p) = 1 / K(p)."""
    return 1.0 / kinetic_kernel(L, m_ref_sq, dim)


def free_propagator_pos(L: int, m_ref_sq: float, dim: int = 2) -> np.ndarray:
    """Analytic free position-space propagator  G(x) = F^{-1}[1/K(p)]."""
    return np.real(np.fft.ifftn(free_propagator_mom(L, m_ref_sq, dim)))


# ======================================================================
# Whitening operator  phi = K^{-1/2} theta
# ======================================================================

class WhiteningOp:
    """phi = L_K theta  with  L_K = F^{-1} diag(1/sqrt(K(p))) F.

    K(p) is real and even under p -> -p, so L_K is real and symmetric in
    the standard real inner product on R^V; hence L_K^T = L_K and the same
    routine serves both  phi <- theta  and  L_K^T applied to a real field.
    """

    def __init__(self, L: int, m_ref_sq: float, dim: int = 2):
        if m_ref_sq <= 0:
            raise ValueError(
                f'WhiteningOp requires m_ref_sq > 0 (got {m_ref_sq}); '
                'the latent kinetic preconditioner 1/sqrt(K(0)) is otherwise '
                'singular/imaginary.')
        self.L = int(L)
        self.dim = int(dim)
        self.m_ref_sq = float(m_ref_sq)
        self.V = L ** dim
        self._K = kinetic_kernel(L, m_ref_sq, dim)
        self._inv_sqrt_K = 1.0 / np.sqrt(self._K)

    def apply(self, theta: np.ndarray) -> np.ndarray:
        """L_K theta  (real output)."""
        th_hat = np.fft.fftn(theta)
        ph_hat = th_hat * self._inv_sqrt_K
        return np.real(np.fft.ifftn(ph_hat))

    # L_K is self-adjoint in the real inner product:
    apply_T = apply

    def apply_inverse(self, phi: np.ndarray) -> np.ndarray:
        """theta = K^{1/2} phi  (inverse whitening).

        Maps a field-space configuration to its latent coordinates so that a
        cold field start phi_x = const can be used to seed latent-space HMC.
        """
        ph_hat = np.fft.fftn(phi)
        th_hat = ph_hat * np.sqrt(self._K)
        return np.real(np.fft.ifftn(th_hat))

    def kernel_mom(self) -> np.ndarray:
        return self._K


# ======================================================================
# Free-field sampler (exact ground truth)
# ======================================================================

@dataclass
class FreeField:
    """Exact i.i.d. free-field sampler  phi = K^{-1/2} theta,  theta ~ N(0,I).

    With kernel K(p) = phat^2 + m_ref^2 this draws from the Gaussian measure
    with covariance  G(p) = 1/K(p).
    """
    L: int
    m_ref_sq: float
    dim: int = 2

    def __post_init__(self):
        self.W = WhiteningOp(self.L, self.m_ref_sq, self.dim)
        self.V = self.L ** self.dim

    def propagator_mom(self) -> np.ndarray:
        return 1.0 / self.W.kernel_mom()

    def phi2_exact(self) -> float:
        """Exact <phi(x)^2> = (1/V) sum_p 1/K(p)."""
        return float(self.propagator_mom().mean())

    def sample_one(self, rng: np.random.Generator) -> np.ndarray:
        theta = rng.standard_normal((self.L,) * self.dim)
        return self.W.apply(theta)

    def sample(self, M: int, seed: int = 0) -> np.ndarray:
        """Draw M i.i.d. free configurations, shape (M, L, ..., L)."""
        rng = np.random.default_rng(seed)
        out = np.empty((M,) + (self.L,) * self.dim)
        for i in range(M):
            out[i] = self.sample_one(rng)
        return out


# ======================================================================
# Empirical free-propagator sanity check
# ======================================================================

def check_free_propagator(L: int, m_ref_sq: float, dim: int = 2,
                          M: int = 2000, seed: int = 0) -> dict:
    """Verify that the empirical momentum propagator matches 1/K(p).

    Estimates  Ghat_emp(p) = < |phi_hat(p)|^2 > / V  from M free draws and
    compares against the analytic  1/K(p).  Returns a dict with the
    relative L2 error, the maximum and RMS pointwise relative errors, and a
    boolean pass flag (tolerance scales like 1/sqrt(M)).
    """
    ff = FreeField(L, m_ref_sq, dim)
    V = ff.V
    G_analytic = ff.propagator_mom()
    acc = np.zeros((L,) * dim)
    rng = np.random.default_rng(seed)
    for _ in range(M):
        phi = ff.sample_one(rng)
        phi_hat = np.fft.fftn(phi)
        acc += np.abs(phi_hat) ** 2
    G_emp = acc / (M * V)
    num = np.linalg.norm(G_emp - G_analytic)
    den = np.linalg.norm(G_analytic)
    rel_l2 = float(num / den)
    rel_err = np.abs(G_emp - G_analytic) / np.abs(G_analytic)
    max_rel_err = float(rel_err.max())
    rms_rel_err = float(np.sqrt(np.mean(rel_err ** 2)))
    # statistical tolerance: propagator estimator error ~ a few / sqrt(M)
    tol = 8.0 / np.sqrt(M)
    return {
        'L': L, 'dim': dim, 'm_ref_sq': float(m_ref_sq), 'M': M, 'seed': seed,
        'rel_l2_error': rel_l2,
        'max_rel_err': max_rel_err,
        'rms_rel_err': rms_rel_err,
        'tol': float(tol),
        'passed': bool(rel_l2 < tol),
        'phi2_emp': float(G_emp.mean()),
        'phi2_exact': float(G_analytic.mean()),
    }


if __name__ == '__main__':
    print('latent_lattice smoke test')
    for L in (16, 32):
        chk = check_free_propagator(L, m_ref_sq=0.25, dim=2, M=2000, seed=0)
        print(f'  L={L:3d}: rel_l2={chk["rel_l2_error"]:.4f} '
              f'(tol {chk["tol"]:.4f}) passed={chk["passed"]}  '
              f'<phi^2> emp={chk["phi2_emp"]:.5f} exact={chk["phi2_exact"]:.5f}')
