"""
Module R1: Kernel recalibration with multiple objectives.

Implements four calibration objectives for TanhRF and CosRFF:
  A. Momentum-space L² fit (uniform weight)
  B. Low-momentum weighted fit (G²-weighted, emphasising IR)
  C. Hybrid momentum + position objective
  D. Shape + amplitude split (fit shape separately, then rescale)

For each architecture and objective: best-fit params, mismatch metrics,
health diagnostics, and the calibrated kernel arrays.
"""

import numpy as np
from scipy.optimize import differential_evolution
from .nonlinear_architectures import (
    TanhRF, CosRFF, free_propagator_mom, free_propagator_pos,
    periodic_embedding,
)

# ----------------------------------------------------------------
# Objective functions
# ----------------------------------------------------------------

def _obj_A_momentum_l2(K_mom, G_mom):
    """Objective A: uniform momentum-space L² relative mismatch."""
    res = K_mom - G_mom
    return float(np.sum(res ** 2) / np.sum(G_mom ** 2))


def _obj_B_low_momentum(K_mom, G_mom):
    """Objective B: G²-weighted L² — emphasises low/intermediate momenta."""
    w = G_mom ** 2
    res = K_mom - G_mom
    return float(np.sum(w * res ** 2) / np.sum(w * G_mom ** 2))


def _obj_C_hybrid(K_mom, K_pos, G_mom, G_pos, L, alpha_pos=0.3):
    """Objective C: hybrid momentum + position.

    (1-α) × momentum L² + α × position L² on selected separations.
    """
    mom_part = np.sum((K_mom - G_mom) ** 2) / np.sum(G_mom ** 2)
    # Position: use (0,0), (1,0), (0,1), (1,1), (2,0)
    seps = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0), (4, 0)]
    pos_sq = 0.0
    pos_norm = 0.0
    for di, dj in seps:
        diff = K_pos[di % L, dj % L] - G_pos[di % L, dj % L]
        pos_sq += diff ** 2
        pos_norm += G_pos[di % L, dj % L] ** 2
    pos_part = pos_sq / (pos_norm + 1e-30)
    return float((1 - alpha_pos) * mom_part + alpha_pos * pos_part)


def _obj_D_shape(K_mom, G_mom):
    """Objective D: shape mismatch after optimal rescaling.

    Minimize over α:  Σ (α K̂ - Ĝ)² / Σ Ĝ²
    α_opt = Σ K̂ Ĝ / Σ K̂²
    """
    alpha_opt = np.sum(K_mom * G_mom) / (np.sum(K_mom ** 2) + 1e-30)
    res = alpha_opt * K_mom - G_mom
    return float(np.sum(res ** 2) / np.sum(G_mom ** 2))


# ----------------------------------------------------------------
# Architecture calibrators
# ----------------------------------------------------------------

def _build_tanh(L, sw, sb, sc=1.0):
    return TanhRF(L, sw, sb, sc)


def _build_rff(L, sk, sc=1.0):
    return CosRFF(L, sk, sc)


def _rescale_sigma_c(arch, G_mom):
    """Find optimal σ_c so calibrated kernel best matches G_mom."""
    K_raw = arch.kernel_mom()
    alpha = np.sum(K_raw * G_mom) / (np.sum(K_raw ** 2) + 1e-30)
    sc_new = np.sqrt(max(alpha, 1e-10)) * arch.sigma_c
    return sc_new


def calibrate_single(arch_type, L, m2, objective='B',
                     sw_bounds=(0.01, 30.0), sb_bounds=(0.001, 30.0),
                     sk_bounds=(0.01, 30.0), maxiter=300, seed=42):
    """Calibrate one architecture with one objective.

    Parameters
    ----------
    arch_type : 'tanh' or 'rff'
    objective : 'A', 'B', 'C', or 'D'

    Returns dict with calibrated params, architecture, mismatch metrics.
    """
    G_mom = free_propagator_mom(L, m2)
    G_pos = free_propagator_pos(L, m2)

    if arch_type == 'tanh':
        bounds = [sw_bounds, sb_bounds]

        def _obj(params):
            sw, sb = params
            arch = _build_tanh(L, sw, sb)
            K_mom = arch.kernel_mom()
            K_pos = np.real(np.fft.ifft2(K_mom))
            if objective == 'A':
                return _obj_A_momentum_l2(K_mom, G_mom)
            elif objective == 'B':
                return _obj_B_low_momentum(K_mom, G_mom)
            elif objective == 'C':
                return _obj_C_hybrid(K_mom, K_pos, G_mom, G_pos, L)
            elif objective == 'D':
                return _obj_D_shape(K_mom, G_mom)
            return _obj_B_low_momentum(K_mom, G_mom)

        result = differential_evolution(_obj, bounds, seed=seed,
                                        maxiter=maxiter, tol=1e-10,
                                        polish=True)
        sw_opt, sb_opt = result.x

        # Determine σ_c by rescaling
        arch_raw = _build_tanh(L, sw_opt, sb_opt, 1.0)
        sc_opt = _rescale_sigma_c(arch_raw, G_mom)
        arch = _build_tanh(L, sw_opt, sb_opt, sc_opt)

        params = {'sigma_w': sw_opt, 'sigma_b': sb_opt, 'sigma_c': sc_opt}
        at_bound = (abs(sw_opt - sw_bounds[1]) < 0.1 or
                    abs(sb_opt - sb_bounds[1]) < 0.1)

    elif arch_type == 'rff':
        bounds = [sk_bounds]

        def _obj(params):
            sk = params[0]
            arch = _build_rff(L, sk)
            K_mom = arch.kernel_mom()
            K_pos = np.real(np.fft.ifft2(K_mom))
            if objective == 'A':
                return _obj_A_momentum_l2(K_mom, G_mom)
            elif objective == 'B':
                return _obj_B_low_momentum(K_mom, G_mom)
            elif objective == 'C':
                return _obj_C_hybrid(K_mom, K_pos, G_mom, G_pos, L)
            elif objective == 'D':
                return _obj_D_shape(K_mom, G_mom)
            return _obj_B_low_momentum(K_mom, G_mom)

        result = differential_evolution(_obj, bounds, seed=seed,
                                        maxiter=maxiter, tol=1e-10,
                                        polish=True)
        sk_opt = result.x[0]

        arch_raw = _build_rff(L, sk_opt, 1.0)
        sc_opt = _rescale_sigma_c(arch_raw, G_mom)
        arch = _build_rff(L, sk_opt, sc_opt)

        params = {'sigma_k': sk_opt, 'sigma_c': sc_opt}
        at_bound = abs(sk_opt - sk_bounds[1]) < 0.1
    else:
        raise ValueError(f"Unknown arch_type: {arch_type}")

    # Compute all mismatch metrics
    K_mom_cal = arch.kernel_mom()
    K_pos_cal = np.real(np.fft.ifft2(K_mom_cal))
    metrics = compute_mismatch_suite(K_mom_cal, K_pos_cal, G_mom, G_pos, L)
    metrics['optimizer_fun'] = float(result.fun)
    metrics['optimizer_nfev'] = int(result.nfev)
    metrics['at_bound'] = at_bound

    return {
        'arch_type': arch_type,
        'objective': objective,
        'params': params,
        'arch': arch,
        'K_mom': K_mom_cal,
        'K_pos': K_pos_cal,
        'G_mom': G_mom,
        'G_pos': G_pos,
        'metrics': metrics,
    }


def compute_mismatch_suite(K_mom, K_pos, G_mom, G_pos, L):
    """Comprehensive mismatch diagnostics."""
    V = L * L

    # Momentum-space relative error
    rel_mom = (K_mom - G_mom) / (np.abs(G_mom) + 1e-30)

    # Momentum regions
    n = np.arange(L)
    p = 2 * np.pi * n / L
    ph2 = 4 * np.sin(p / 2) ** 2
    phat2 = ph2[:, None] + ph2[None, :]
    low_mask = phat2 < 2.0
    mid_mask = (phat2 >= 2.0) & (phat2 < 6.0)
    high_mask = phat2 >= 6.0

    def _rms(arr):
        return float(np.sqrt(np.mean(arr ** 2))) if arr.size > 0 else 0.0

    # Position-space diagnostics at key separations
    seps = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0), (4, 0)]
    pos_table = {}
    for di, dj in seps:
        k_val = K_pos[di % L, dj % L]
        g_val = G_pos[di % L, dj % L]
        pos_table[f'({di},{dj})'] = {
            'K': float(k_val),
            'G': float(g_val),
            'rel_err': float((k_val - g_val) / (abs(g_val) + 1e-30)),
        }

    return {
        'obj_A': _obj_A_momentum_l2(K_mom, G_mom),
        'obj_B': _obj_B_low_momentum(K_mom, G_mom),
        'obj_D': _obj_D_shape(K_mom, G_mom),
        'l2_rel_full': _rms(rel_mom),
        'l2_rel_low': _rms(rel_mom[low_mask]),
        'l2_rel_mid': _rms(rel_mom[mid_mask]),
        'l2_rel_high': _rms(rel_mom[high_mask]),
        'max_rel': float(np.max(np.abs(rel_mom))),
        'zero_mode_match': float(K_mom[0, 0] / G_mom[0, 0]),
        'K_00': float(K_mom[0, 0]),
        'G_00': float(G_mom[0, 0]),
        'K_pos_00': float(K_pos[0, 0]),
        'G_pos_00': float(G_pos[0, 0]),
        'position_table': pos_table,
    }


def recalibrate_all(L, m2, objectives=None):
    """Run calibration across all objectives for both architectures.

    Returns nested dict: results[arch_type][objective] = calibration_result.
    """
    if objectives is None:
        objectives = ['A', 'B', 'C', 'D']

    results = {}
    for arch_type in ['rff', 'tanh']:
        results[arch_type] = {}
        for obj in objectives:
            cal = calibrate_single(arch_type, L, m2, objective=obj)
            results[arch_type][obj] = cal
    return results


def select_best_calibration(cal_results, arch_type, prefer_metric='obj_B'):
    """Select best calibration objective for a given architecture.

    Prefers the objective that minimises prefer_metric, with a penalty
    for hitting parameter bounds.
    """
    best_obj = None
    best_score = float('inf')

    for obj, cal in cal_results[arch_type].items():
        score = cal['metrics'].get(prefer_metric, 1e6)
        if cal['metrics'].get('at_bound', False):
            score *= 1.5  # penalise bound-hitting
        if score < best_score:
            best_score = score
            best_obj = obj

    return best_obj, cal_results[arch_type][best_obj]


def tanh_saturation_diagnostic(cal_results):
    """Diagnose whether tanh is stuck in saturation across all objectives."""
    diag = {}
    for obj, cal in cal_results.get('tanh', {}).items():
        p = cal['params']
        sw = p.get('sigma_w', 0)
        sb = p.get('sigma_b', 0)
        # In saturation: σ_w → ∞ makes tanh → sign, losing all gradients
        # Indicator: σ_w large and kernel very flat in high-momentum modes
        K_mom = cal['K_mom']
        dynamic_range = float(K_mom.max() / (K_mom.min() + 1e-30))
        diag[obj] = {
            'sigma_w': sw,
            'sigma_b': sb,
            'at_bound': cal['metrics']['at_bound'],
            'dynamic_range': dynamic_range,
            'obj_B': cal['metrics']['obj_B'],
            'zero_mode_match': cal['metrics']['zero_mode_match'],
        }

    # Verdict
    all_at_bound = all(d['at_bound'] for d in diag.values())
    best_obj_B = min(d['obj_B'] for d in diag.values())

    return {
        'per_objective': diag,
        'all_at_bound': all_at_bound,
        'best_obj_B': best_obj_B,
        'verdict': ('SATURATED — unsuitable as calibrated free-theory surrogate'
                     if all_at_bound and best_obj_B > 0.3
                     else 'PARTIAL — usable with caveats'
                     if best_obj_B > 0.1
                     else 'OK'),
    }
