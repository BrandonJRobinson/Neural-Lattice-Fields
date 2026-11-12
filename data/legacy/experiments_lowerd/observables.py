"""
Observables, error estimators, and autocorrelation tools for the
lower-dimensional NNFT experiments.

Includes:
  * per-config magnetization moments (M, |M|, M^2, M^4);
  * derived estimators (susceptibility chi, chi0, Binder cumulant U_4)
    with block-jackknife errors that respect autocorrelation;
  * a Sokal-style integrated autocorrelation time / ESS estimator;
  * vertex-correlator machinery for sine-Gordon (translation-averaged
    via FFT, radially binned) and the analytic Gaussian prediction;
  * replica-aware combination of independent seeds.

NumPy only.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List


# ======================================================================
# Magnetization moments (one scalar config summary)
# ======================================================================

def magnetization_summary(phi: np.ndarray) -> np.ndarray:
    """Return [M, |M|, M^2, M^4] for a single configuration.

    M = (1/V) sum_x phi_x.  These four scalars are the per-config
    primary data for the Ising-universality observables.
    """
    M = float(phi.mean())
    aM = abs(M)
    return np.array([M, aM, M * M, M ** 4], dtype=np.float64)


# Column layout produced by magnetization_summary
MAG_COLS = ('M', 'absM', 'M2', 'M4')


def field_summary(phi: np.ndarray) -> np.ndarray:
    """Return [M, |M|, M^2, M^4, phi2, phi4] for a single configuration.

    M    = (1/V) sum_x phi_x        (order parameter / magnetization)
    phi2 = (1/V) sum_x phi_x^2      (local quadratic moment)
    phi4 = (1/V) sum_x phi_x^4      (local quartic moment)

    The first four columns are the Ising magnetization moments; phi2/phi4
    are local field moments used for the single-site potential sanity check.
    """
    M = float(phi.mean())
    aM = abs(M)
    phi2 = float(np.mean(phi ** 2))
    phi4 = float(np.mean(phi ** 4))
    return np.array([M, aM, M * M, M ** 4, phi2, phi4], dtype=np.float64)


# Column layout produced by field_summary
FIELD_COLS = ('M', 'absM', 'M2', 'M4', 'phi2', 'phi4')


# ======================================================================
# Block jackknife (autocorrelation-aware) for derived estimators
# ======================================================================

def block_jackknife(arrays: Dict[str, np.ndarray],
                    estimator: Callable[[Dict[str, float]], float],
                    n_blocks: int = 20) -> dict:
    """Block-jackknife a derived estimator built from sample means.

    Parameters
    ----------
    arrays    : dict name -> (M,) per-config time series.
    estimator : maps a dict of *means* {name: mean} to a scalar.
    n_blocks  : number of contiguous blocks (groups consecutive samples to
                tame autocorrelation).

    Returns dict with value (full-sample), err (jackknife), n_blocks.
    """
    names = list(arrays.keys())
    M = len(arrays[names[0]])
    B = int(min(n_blocks, M))
    if B < 2:
        full = estimator({k: float(v.mean()) for k, v in arrays.items()})
        return {'value': float(full), 'err': float('nan'), 'n_blocks': B}
    blocks = np.array_split(np.arange(M), B)
    full = estimator({k: float(v.mean()) for k, v in arrays.items()})
    jk = np.empty(B)
    for b in range(B):
        keep = np.ones(M, dtype=bool)
        keep[blocks[b]] = False
        means = {k: float(v[keep].mean()) for k, v in arrays.items()}
        jk[b] = estimator(means)
    jk_mean = jk.mean()
    err = np.sqrt((B - 1) / B * np.sum((jk - jk_mean) ** 2))
    return {'value': float(full), 'err': float(err), 'n_blocks': B}


# ---- derived estimators ----------------------------------------------

def make_chi0_estimator(V: int):
    """chi0 = V <M^2>."""
    return lambda m: V * m['M2']


def make_chi_estimator(V: int):
    """chi = V (<M^2> - <|M|>^2).  (susceptibility from |M|, "chi_abs")"""
    return lambda m: V * (m['M2'] - m['absM'] ** 2)


def make_chi_abs_estimator(V: int):
    """chi_abs = V (<M^2> - <|M|>^2)."""
    return lambda m: V * (m['M2'] - m['absM'] ** 2)


def make_chi_conn_estimator(V: int):
    """chi_conn = V (<M^2> - <M>^2)."""
    return lambda m: V * (m['M2'] - m['M'] ** 2)


def binder_estimator(m: Dict[str, float]) -> float:
    """U_4 = 1 - <M^4> / (3 <M^2>^2)."""
    denom = 3.0 * m['M2'] ** 2
    return 1.0 - m['M4'] / denom if denom > 0 else float('nan')


# ======================================================================
# Integrated autocorrelation time / ESS  (Sokal automatic window)
# ======================================================================

def integrated_autocorr_time(x: np.ndarray, c: float = 6.0) -> dict:
    """Sokal-style tau_int with automatic windowing.

    Returns dict {tau_int, N_eff, W}.  Conservative / flagged on short or
    degenerate series.
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    x = x - x.mean()
    var = float((x * x).mean())
    if var <= 0 or N < 20:
        return {'tau_int': float('nan'), 'N_eff': float('nan'),
                'W': 0, 'flag': 'degenerate_or_short'}
    f = np.fft.fft(x, 2 * N)
    acf = np.real(np.fft.ifft(f * np.conj(f)))[:N] / (N - np.arange(N))
    rho = acf / acf[0]
    tau = 0.5
    W_chosen = N - 1
    for W in range(1, N):
        tau = 0.5 + np.sum(rho[1:W + 1])
        if W >= c * tau:
            W_chosen = W
            break
    tau = max(tau, 0.5)
    flag = 'ok' if W_chosen < N - 1 else 'window_not_converged'
    return {'tau_int': float(tau), 'N_eff': float(N / (2 * tau)),
            'W': int(W_chosen), 'flag': flag}


# ======================================================================
# Replica-aware combination across independent seeds
# ======================================================================

def combine_replicas(values: List[float], errs: List[float]) -> dict:
    """Combine independent per-seed estimates into a replica-aware result.

    Reports:
      * mean of per-seed central values,
      * between-replica SE = std(values, ddof=1)/sqrt(R),
      * within-replica SE  = sqrt(sum err_i^2)/R,
      * combined SE = max of the two (conservative).
    """
    v = np.asarray(values, dtype=float)
    e = np.asarray(errs, dtype=float)
    R = len(v)
    mean = float(v.mean())
    if R < 2:
        within = float(e[0]) if R == 1 and np.isfinite(e[0]) else float('nan')
        return {'mean': mean, 'se_between': float('nan'),
                'se_within': within, 'se_combined': within, 'R': R}
    se_between = float(v.std(ddof=1) / np.sqrt(R))
    finite = e[np.isfinite(e)]
    se_within = float(np.sqrt(np.sum(finite ** 2)) / R) if finite.size else float('nan')
    cands = [s for s in (se_between, se_within) if np.isfinite(s)]
    se_combined = float(max(cands)) if cands else float('nan')
    return {'mean': mean, 'se_between': se_between, 'se_within': se_within,
            'se_combined': se_combined, 'R': R}


# ======================================================================
# Sine-Gordon: vertex correlators and Gaussian prediction
# ======================================================================

def radial_bins(L: int, dim: int = 2):
    """Precompute integer radial bin indices and r-values for an L^dim grid.

    Uses minimum-image distances on the periodic torus.  Returns
    (bin_index_flat, r_values) where bin_index_flat maps each lattice site
    (in C order) to a radial bin and r_values gives the representative radius
    of each bin.
    """
    coords = [np.minimum(np.arange(L), L - np.arange(L)) for _ in range(dim)]
    grids = np.meshgrid(*coords, indexing='ij')
    r2 = np.zeros((L,) * dim)
    for g in grids:
        r2 = r2 + g ** 2
    r = np.sqrt(r2)
    rflat = r.ravel()
    # integer-rounded radius as bin key
    rb = np.round(rflat).astype(int)
    uniq = np.unique(rb)
    remap = {u: i for i, u in enumerate(uniq)}
    bin_index = np.array([remap[v] for v in rb], dtype=int)
    r_values = np.array([rflat[rb == u].mean() for u in uniq], dtype=float)
    return bin_index, r_values


def vertex_correlator_config(phi: np.ndarray, alpha: float) -> np.ndarray:
    """Translation-averaged vertex correlator field for one configuration.

    C_alpha(r) = (1/V) sum_x < e^{i alpha phi(x+r)} e^{-i alpha phi(x)} >.

    For a single config the translation average over x is computed exactly
    via FFT:  let f = e^{i alpha phi}; then
        sum_x f(x+r) conj(f(x)) = F^{-1}[ |F[f]|^2 ](r).
    The configuration average over the Monte-Carlo chain is taken later.
    Returns the real part as a real field of shape (L,)*dim.
    """
    f = np.exp(1j * alpha * phi)
    fhat = np.fft.fftn(f)
    corr = np.fft.ifftn(np.abs(fhat) ** 2) / phi.size
    return np.real(corr)


def two_point_field(phi: np.ndarray) -> np.ndarray:
    """Translation-averaged connected two-point field G(r) = <phi(x+r)phi(x)>
    for one configuration (FFT translation average, real part)."""
    fhat = np.fft.fftn(phi)
    g = np.fft.ifftn(np.abs(fhat) ** 2) / phi.size
    return np.real(g)


def radial_average_flat(field: np.ndarray, bin_index: np.ndarray,
                        n_bins: int) -> np.ndarray:
    """Average a real field over precomputed radial bins."""
    flat = field.ravel()
    out = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    np.add.at(out, bin_index, flat)
    np.add.at(cnt, bin_index, 1.0)
    return out / np.maximum(cnt, 1.0)


def gaussian_vertex_prediction(L: int, m_ref_sq: float, alpha: float,
                               dim: int, bin_index: np.ndarray,
                               n_bins: int) -> np.ndarray:
    """Analytic free-field vertex correlator

        C_alpha(r) = exp[ -alpha^2/2 <(phi(r)-phi(0))^2> ],
        <(phi(r)-phi(0))^2> = 2 (G(0) - G(r)),

    with G the exact free propagator F^{-1}[1/K(p)].  Returned radially
    binned to match the measured estimator.
    """
    from latent_lattice import free_propagator_pos
    G = free_propagator_pos(L, m_ref_sq, dim)
    D = 2.0 * (G.ravel()[0] - G)            # <(phi(r)-phi(0))^2>
    Cfield = np.exp(-0.5 * alpha ** 2 * D)
    return radial_average_flat(Cfield, bin_index, n_bins)


# ======================================================================
# 3d (and general-d) structure-factor observables for xi_{2nd}
# ======================================================================

FIELD_GHAT_COLS = ('M', 'absM', 'M2', 'M4', 'phi2', 'phi4', 'ghat0', 'ghatpmin')


def field_summary_with_ghat(phi: np.ndarray) -> np.ndarray:
    """Extended field summary including Fourier structure factor.

    Adds two columns beyond FIELD_COLS:
      ghat0    = V * <M>^2  (= G̃(0) per configuration)
      ghatpmin = |Σ_x φ_x exp(2πi x_0/L)|^2 / V  (= G̃(p_min) along axis 0)

    Works for any lattice dimension (phi.ndim).
    """
    V = phi.size
    L = phi.shape[0]
    dim = phi.ndim
    M = float(phi.mean())
    absM = float(abs(M))
    M2 = float(M ** 2)
    M4 = float(M ** 4)
    phi2 = float(np.mean(phi ** 2))
    phi4 = float(np.mean(phi ** 4))
    ghat0 = float(V * M2)
    # Sum over all axes except axis-0, then 1D DFT at p_min = 2π/L
    slice_sums = phi.reshape(L, -1).sum(axis=1).astype(complex)
    x0 = np.arange(L, dtype=float)
    phase = np.exp(2j * np.pi * x0 / L)
    ghatpmin = float(np.abs(np.dot(phase, slice_sums)) ** 2 / V)
    return np.array([M, absM, M2, M4, phi2, phi4, ghat0, ghatpmin])


def make_xi_2ndmom_estimator(L: int):
    """Return estimator for ξ_{2nd}(L) = sqrt[(G̃(0)/G̃(p_min)-1)/(4 sin²(π/L))].

    Estimator takes arrays dict with 'ghat0' and 'ghatpmin' columns and
    returns ξ as a float (or nan if the ratio is ill-defined).
    """
    denom = 4.0 * np.sin(np.pi / L) ** 2

    def est(m: Dict[str, np.ndarray]) -> float:
        g0 = float(np.mean(m['ghat0']))
        gp = float(np.mean(m['ghatpmin']))
        if gp <= 0 or not np.isfinite(gp):
            return float('nan')
        ratio = g0 / gp
        xi2 = (ratio - 1.0) / denom
        return float(np.sqrt(xi2)) if xi2 >= 0 else float('nan')

    return est


def make_xi_over_L_estimator(L: int):
    """Estimator for ξ_{2nd}/L (dimensionless ratio)."""
    xi_est = make_xi_2ndmom_estimator(L)
    L_f = float(L)

    def est(m: Dict[str, np.ndarray]) -> float:
        v = xi_est(m)
        return v / L_f if np.isfinite(v) else float('nan')

    return est
