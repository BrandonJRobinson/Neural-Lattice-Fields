"""
HMC samplers for the lower-dimensional NNFT exact-factorization experiments.

Two samplers are provided:

  * LatentHMC  -- HMC on the exact induced latent density
        S_latent(theta) = 1/2 ||theta||^2 + S_int[ phi(theta) ],
        phi = K^{-1/2} theta.
    The interaction is supplied as two callables, int_action(phi) and
    int_grad_phi(phi) = dS_int/dphi, so the same engine serves both the
    2d phi^4 and 2d sine-Gordon experiments.

  * FieldHMC   -- HMC directly in field space on the full action
        S[phi] = 1/2 sum_{x,mu} (phi_{x+mu}-phi_x)^2 + sum_x V(phi_x),
    used as an independent validation baseline at small volumes.

Both use leapfrog integration with a Metropolis accept/reject on Delta H,
adaptive step-size warmup during thermalization, and FIXED parameters
during production.  Measurement is streamed through a user-supplied
`measure(phi) -> 1d array` callback so memory stays O(M * k) in the number
of measured scalars k, never O(M * V).

NumPy only.
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

from latent_lattice import WhiteningOp, phat2_grid


def _all_finite(*vals) -> bool:
    """Return True if every scalar/array argument contains only finite values."""
    for v in vals:
        a = np.asarray(v)
        if not np.all(np.isfinite(a)):
            return False
    return True


# ======================================================================
# Latent-space HMC on the induced interacting density
# ======================================================================

@dataclass
class LatentHMC:
    """HMC on  S_latent(theta) = 0.5||theta||^2 + S_int[phi(theta)].

    Parameters
    ----------
    W            : WhiteningOp implementing phi = K^{-1/2} theta.
    int_action   : callable phi -> float, the interaction action S_int[phi].
    int_grad_phi : callable phi -> ndarray, dS_int/dphi (same shape as phi).
    step_size    : initial leapfrog step.
    n_leapfrog   : number of leapfrog steps per trajectory.
    """
    W: WhiteningOp
    int_action: Callable[[np.ndarray], float]
    int_grad_phi: Callable[[np.ndarray], np.ndarray]
    step_size: float = 0.05
    n_leapfrog: int = 20

    def __post_init__(self):
        self.L = self.W.L
        self.dim = self.W.dim
        self.V = self.W.V

    # -- action / gradient ------------------------------------------------
    def action(self, theta, phi=None) -> float:
        if phi is None:
            phi = self.W.apply(theta)
        return 0.5 * float(np.sum(theta ** 2)) + float(self.int_action(phi))

    def grad(self, theta):
        """Return (dS_latent/dtheta, phi).

        dS_latent/dtheta = theta + L_K^T (dS_int/dphi),  L_K^T = L_K.
        """
        phi = self.W.apply(theta)
        g = theta + self.W.apply_T(self.int_grad_phi(phi))
        return g, phi

    # -- one trajectory ---------------------------------------------------
    def leapfrog(self, theta, p, eps, n_steps):
        g, phi = self.grad(theta)
        if not _all_finite(g, phi, theta, p):
            return theta, p, phi, False
        p = p - 0.5 * eps * g
        for step in range(n_steps):
            theta = theta + eps * p
            g, phi = self.grad(theta)
            if not _all_finite(theta, p, g, phi):
                return theta, p, phi, False
            if step < n_steps - 1:
                p = p - eps * g
            else:
                p = p - 0.5 * eps * g
        return theta, p, phi, True

    def sample_step(self, theta, rng) -> dict:
        p = rng.standard_normal(theta.shape)
        phi0 = self.W.apply(theta)
        if not _all_finite(theta, p, phi0):
            return {'theta': theta, 'phi': phi0, 'accepted': False,
                    'dH': float('nan'), 'nonfinite_reject': True}
        H0_kin = 0.5 * np.sum(p ** 2)
        H0_act = self.action(theta, phi0)
        H0 = H0_kin + H0_act
        if not _all_finite(H0_kin, H0_act, H0):
            return {'theta': theta, 'phi': phi0, 'accepted': False,
                    'dH': float('nan'), 'nonfinite_reject': True}
        theta_new, p_new, phi_new, ok = self.leapfrog(
            theta.copy(), p.copy(), self.step_size, self.n_leapfrog)
        if (not ok) or (not _all_finite(theta_new, p_new, phi_new)):
            return {'theta': theta, 'phi': phi0, 'accepted': False,
                    'dH': float('nan'), 'nonfinite_reject': True}
        H1_kin = 0.5 * np.sum(p_new ** 2)
        H1_act = self.action(theta_new, phi_new)
        H1 = H1_kin + H1_act
        dH = float(H1 - H0)
        if not _all_finite(H1_kin, H1_act, H1, dH):
            return {'theta': theta, 'phi': phi0, 'accepted': False,
                    'dH': float('nan'), 'nonfinite_reject': True}
        if rng.random() < min(1.0, np.exp(-dH)):
            return {'theta': theta_new, 'phi': phi_new, 'accepted': True,
                    'dH': dH, 'nonfinite_reject': False}
        return {'theta': theta, 'phi': phi0, 'accepted': False,
                'dH': dH, 'nonfinite_reject': False}

    def run(self, measure: Callable[[np.ndarray], np.ndarray],
            n_therm: int, n_measure: int, measure_every: int = 1,
            seed: int = 0, theta_init: Optional[np.ndarray] = None,
            adapt_step: bool = True, target_acc: float = 0.75) -> dict:
        return _run_chain(self, measure, n_therm, n_measure, measure_every,
                          seed, theta_init, adapt_step, target_acc,
                          space='latent')


# ======================================================================
# Field-space HMC baseline
# ======================================================================

@dataclass
class FieldHMC:
    """HMC directly in field space on the full lattice action.

    S[phi] = 0.5 sum_{x,mu}(phi_{x+mu}-phi_x)^2 + sum_x site_potential(phi_x).

    The kinetic gradient is the (negative) lattice Laplacian computed by
    nearest-neighbour rolls; site_grad gives dV/dphi pointwise.
    """
    L: int
    dim: int
    site_potential: Callable[[np.ndarray], np.ndarray]   # V(phi) elementwise
    site_grad: Callable[[np.ndarray], np.ndarray]         # dV/dphi elementwise
    step_size: float = 0.01
    n_leapfrog: int = 20

    def __post_init__(self):
        self.V = self.L ** self.dim

    def _neg_laplacian(self, phi):
        # sum_mu (2 phi - phi_{+mu} - phi_{-mu})
        out = 2.0 * self.dim * phi
        for ax in range(self.dim):
            out = out - np.roll(phi, 1, axis=ax) - np.roll(phi, -1, axis=ax)
        return out

    def kinetic_action(self, phi):
        s = 0.0
        for ax in range(self.dim):
            s += 0.5 * np.sum((np.roll(phi, -1, axis=ax) - phi) ** 2)
        return float(s)

    def action(self, phi):
        return self.kinetic_action(phi) + float(np.sum(self.site_potential(phi)))

    def grad(self, phi):
        return self._neg_laplacian(phi) + self.site_grad(phi)

    def leapfrog(self, phi, p, eps, n_steps):
        g = self.grad(phi)
        if not _all_finite(phi, p, g):
            return phi, p, False
        p = p - 0.5 * eps * g
        for step in range(n_steps):
            phi = phi + eps * p
            g = self.grad(phi)
            if not _all_finite(phi, p, g):
                return phi, p, False
            if step < n_steps - 1:
                p = p - eps * g
            else:
                p = p - 0.5 * eps * g
        return phi, p, True

    def sample_step(self, phi, rng) -> dict:
        p = rng.standard_normal(phi.shape)
        if not _all_finite(phi, p):
            return {'phi': phi, 'accepted': False, 'dH': float('nan'),
                'nonfinite_reject': True}
        H0_kin = 0.5 * np.sum(p ** 2)
        H0_act = self.action(phi)
        H0 = H0_kin + H0_act
        if not _all_finite(H0_kin, H0_act, H0):
            return {'phi': phi, 'accepted': False, 'dH': float('nan'),
                'nonfinite_reject': True}
        phi_new, p_new, ok = self.leapfrog(phi.copy(), p.copy(),
                                       self.step_size, self.n_leapfrog)
        if (not ok) or (not _all_finite(phi_new, p_new)):
            return {'phi': phi, 'accepted': False, 'dH': float('nan'),
                'nonfinite_reject': True}
        H1_kin = 0.5 * np.sum(p_new ** 2)
        H1_act = self.action(phi_new)
        H1 = H1_kin + H1_act
        dH = float(H1 - H0)
        if not _all_finite(H1_kin, H1_act, H1, dH):
            return {'phi': phi, 'accepted': False, 'dH': float('nan'),
                'nonfinite_reject': True}
        if rng.random() < min(1.0, np.exp(-dH)):
            return {'phi': phi_new, 'accepted': True, 'dH': dH,
                'nonfinite_reject': False}
        return {'phi': phi, 'accepted': False, 'dH': dH,
            'nonfinite_reject': False}

    def run(self, measure, n_therm, n_measure, measure_every=1,
            seed=0, phi_init=None, adapt_step=True, target_acc=0.75) -> dict:
        return _run_chain(self, measure, n_therm, n_measure, measure_every,
                          seed, phi_init, adapt_step, target_acc,
                          space='field')


# ======================================================================
# Shared chain driver (works for both samplers via a `space` switch)
# ======================================================================

def _run_chain(sampler, measure, n_therm, n_measure, measure_every,
               seed, x_init, adapt_step, target_acc, space) -> dict:
    L, dim = sampler.L, sampler.dim
    rng = np.random.default_rng(seed)
    shape = (L,) * dim
    x = rng.standard_normal(shape) if x_init is None else x_init.copy()

    t0 = time.time()
    key = 'theta' if space == 'latent' else 'phi'

    # -- thermalization with step-size adaptation --
    recent_acc, recent_n = 0, 0
    nonfinite_rejects_therm = 0
    for t in range(n_therm):
        out = sampler.sample_step(x, rng)
        x = out[key]
        nonfinite_rejects_therm += int(out.get('nonfinite_reject', False))
        recent_acc += int(out['accepted'])
        recent_n += 1
        if adapt_step and recent_n >= 50 and (t + 1) % 50 == 0:
            mean_acc = recent_acc / recent_n
            if mean_acc > target_acc + 0.05:
                sampler.step_size *= 1.1
            elif mean_acc < target_acc - 0.1:
                sampler.step_size *= 0.9
            recent_acc, recent_n = 0, 0

    # -- production with streamed measurement (FIXED step size) --
    samples_list = []
    acc_count = acc_total = 0
    finite_dH = []
    accepted_finite_dH = []
    expmdH_finite = []
    nonfinite_rejects_prod = 0
    nonfinite_measurements = 0
    max_abs_phi = 0.0
    max_abs_theta = 0.0 if space == 'latent' else float('nan')
    action_vals = []
    for i in range(n_measure):
        for _ in range(measure_every):
            out = sampler.sample_step(x, rng)
            x = out[key]
            acc_count += int(out['accepted'])
            acc_total += 1
            nonfinite_rejects_prod += int(out.get('nonfinite_reject', False))
            dH = float(out['dH'])
            if np.isfinite(dH):
                finite_dH.append(dH)
                v = float(np.exp(-dH))
                if np.isfinite(v):
                    expmdH_finite.append(v)
                if out['accepted']:
                    accepted_finite_dH.append(dH)
        phi = out['phi'] if space == 'latent' else x
        if not _all_finite(phi):
            nonfinite_measurements += 1
            continue
        max_abs_phi = max(max_abs_phi, float(np.max(np.abs(phi))))
        if space == 'latent':
            max_abs_theta = max(max_abs_theta, float(np.max(np.abs(x))))

        try:
            action_val = sampler.action(x, phi) if space == 'latent' else sampler.action(x)
            if np.isfinite(action_val):
                action_vals.append(float(action_val))
        except Exception:
            pass

        m = np.atleast_1d(np.asarray(measure(phi), dtype=np.float64))
        if not np.all(np.isfinite(m)):
            nonfinite_measurements += 1
            continue
        samples_list.append(m)

    samples = np.asarray(samples_list, dtype=np.float64)
    if samples.size == 0:
        samples = np.empty((0, 0), dtype=np.float64)

    n = max(acc_total, 1)
    if finite_dH:
        dH_mean = float(np.mean(finite_dH))
        dH_std = float(np.std(finite_dH))
        dH_abs_mean = float(np.mean(np.abs(finite_dH)))
        dH_max = float(np.max(np.abs(finite_dH)))
    else:
        dH_mean = dH_std = dH_abs_mean = dH_max = float('nan')
    if accepted_finite_dH:
        dH_acc_mean = float(np.mean(accepted_finite_dH))
        dH_acc_std = float(np.std(accepted_finite_dH))
        dH_acc_max = float(np.max(np.abs(accepted_finite_dH)))
    else:
        dH_acc_mean = dH_acc_std = dH_acc_max = float('nan')

    return {
        'samples': samples,
        'acceptance': acc_count / n,
        'step_size_final': float(sampler.step_size),
        'n_leapfrog': sampler.n_leapfrog,
        'n_therm': n_therm,
        'n_measure': n_measure,
        'measure_every': measure_every,
        'seed': seed,
        'dH_stats': {
            'mean': dH_mean,
            'abs_mean': dH_abs_mean,
            'std': dH_std,
            'max_abs': dH_max,
            'exp_neg_mean': (float(np.mean(expmdH_finite))
                             if expmdH_finite else float('nan')),
            'accepted_mean': dH_acc_mean,
            'accepted_std': dH_acc_std,
            'accepted_max_abs': dH_acc_max,
        },
        'nonfinite_rejects': int(nonfinite_rejects_therm + nonfinite_rejects_prod),
        'nonfinite_rejects_therm': int(nonfinite_rejects_therm),
        'nonfinite_rejects_prod': int(nonfinite_rejects_prod),
        'nonfinite_measurements': int(nonfinite_measurements),
        'n_measure_recorded': int(samples.shape[0]),
        'max_abs_phi': float(max_abs_phi),
        'max_abs_theta': float(max_abs_theta),
        'action_stats': {
            'mean': float(np.mean(action_vals)) if action_vals else float('nan'),
            'max': float(np.max(action_vals)) if action_vals else float('nan'),
        },
        'time_s': time.time() - t0,
    }


# ======================================================================
# Interaction builders
# ======================================================================

def phi4_interaction(m0_sq: float, m_ref_sq: float, lam: float):
    """S_int[phi] = sum_x [ (m0^2 - m_ref^2)/2 phi^2 + lambda phi^4 ].

    Returns (int_action, int_grad_phi) for LatentHMC.  Combined with the
    Gaussian kernel K = phat^2 + m_ref^2 this reproduces the full action
    0.5 phi(phat^2)phi + m0^2/2 phi^2 + lambda phi^4.
    """
    dm = m0_sq - m_ref_sq

    def action(phi):
        return float(0.5 * dm * np.sum(phi ** 2) + lam * np.sum(phi ** 4))

    def grad_phi(phi):
        return dm * phi + 4.0 * lam * phi ** 3

    return action, grad_phi


def phi4_field_potential(m0_sq: float, lam: float):
    """Site potential V(phi)=m0^2/2 phi^2 + lambda phi^4 for FieldHMC."""
    def V(phi):
        return 0.5 * m0_sq * phi ** 2 + lam * phi ** 4

    def dV(phi):
        return m0_sq * phi + 4.0 * lam * phi ** 3

    return V, dV


def sine_gordon_interaction(y: float, beta: float):
    """S_int[phi] = -y sum_x cos(beta phi).  Returns (action, grad_phi)."""
    def action(phi):
        return float(-y * np.sum(np.cos(beta * phi)))

    def grad_phi(phi):
        return y * beta * np.sin(beta * phi)

    return action, grad_phi
