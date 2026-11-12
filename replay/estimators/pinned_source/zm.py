"""Zero-mode (uniform-mode) exact-conditioning analysis library.

Exact identities used (phi_x = u + psi_x, sum psi = 0, S2 = sum psi^2, S3 = sum psi^3):

  S[phi] = S_psi[psi] + a4 u^4 + a2 u^2 + a1 u,
      a4 = lambda V,  a2 = m0^2 V/2 + 6 lambda S2,  a1 = 4 lambda S3        (h = 0)

A uniform source -h sum phi = -eta X with X = L u and eta = h L^3 only shifts a1 -> a1 - eta L.
Therefore, for any F(u) and any eta, with Z_u(psi;eta) = int du exp(-a4u^4-a2u^2-(a1-eta L)u):

  <F>_eta = E_0[ Z_u(psi;eta)/Z_u(psi;0) * E_eta[F|psi] ] / E_0[ Z_u(psi;eta)/Z_u(psi;0) ]

where E_0 is the h=0 psi-marginal.  Everything is a function of (S2, S3) per configuration.
"""
from __future__ import annotations

import numpy as np

GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(80)


def load(path):
    a = np.fromfile(path, dtype=np.float64).reshape(-1, 4)
    return {"u": a[:, 0].copy(), "S2": a[:, 1].copy(), "S3": a[:, 2].copy(), "Gp": a[:, 3].copy()}


def _mode(a4, a2, a1):
    u0 = -a1 / (2.0 * a2)
    for _ in range(60):
        f1 = 4 * a4 * u0**3 + 2 * a2 * u0 + a1
        f2 = 12 * a4 * u0**2 + 2 * a2
        step = f1 / f2
        u0 = u0 - step
        if np.all(np.abs(step) < 1e-15 * (1 + np.abs(u0))):
            break
    return u0


def conditional(S2, S3, *, lam, m0sq, V, L, eta=0.0, kmax=4, halfwidth_sigmas=11.0, chunk=200000):
    """Return dict with log Z_u (shifted consistently), and conditional moments of X = L u.

    logZ is returned exactly (not shifted), so ratios Z(eta)/Z(0) are available.
    """
    S2 = np.asarray(S2, float)
    S3 = np.asarray(S3, float)
    n = S2.size
    out_logZ = np.empty(n)
    out_m = np.empty((kmax + 1, n))
    a4 = lam * V
    for s in range(0, n, chunk):
        e = slice(s, min(n, s + chunk))
        a2 = 0.5 * m0sq * V + 6.0 * lam * S2[e]
        if np.any(a2 <= 0):
            raise RuntimeError("non-convex conditional; extend quadrature")
        a1 = 4.0 * lam * S3[e] - eta * L
        u0 = _mode(a4, a2, a1)
        curv = 12 * a4 * u0**2 + 2 * a2
        w = halfwidth_sigmas / np.sqrt(curv)
        x = u0[:, None] + w[:, None] * GL_NODES[None, :]
        f = a4 * x**4 + a2[:, None] * x**2 + a1[:, None] * x
        f0 = a4 * u0**4 + a2 * u0**2 + a1 * u0
        dens = np.exp(-(f - f0[:, None])) * GL_WEIGHTS[None, :] * w[:, None]
        Z = dens.sum(axis=1)
        out_logZ[e] = np.log(Z) - f0
        X = L * x
        for k in range(kmax + 1):
            out_m[k, e] = (dens * X**k).sum(axis=1) / Z
    return {"logZ": out_logZ, "mom": out_m}


def blocks_mean(arr, nblock):
    n = (arr.shape[-1] // nblock) * nblock
    return arr[..., :n].reshape(arr.shape[:-1] + (nblock, n // nblock)).mean(axis=-1)


def jackknife(fun, data_blocks):
    """data_blocks: list of arrays shaped (..., nb); fun maps list of block-averaged arrays -> scalar/array."""
    nb = data_blocks[0].shape[-1]
    full = fun([d.mean(axis=-1) for d in data_blocks])
    tot = [d.sum(axis=-1) for d in data_blocks]
    jk = []
    for i in range(nb):
        jk.append(fun([(t - d[..., i]) / (nb - 1) for t, d in zip(tot, data_blocks)]))
    jk = np.array(jk)
    err = np.sqrt((nb - 1) / nb * np.sum((jk - jk.mean(axis=0)) ** 2, axis=0))
    return full, err, jk


def tau_int(x, c=6.0, maxlag=None):
    """Madras-Sokal windowed integrated autocorrelation time (in measurement units)."""
    x = np.asarray(x, float) - np.mean(x)
    n = x.size
    if maxlag is None:
        maxlag = min(n // 4, 20000)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[: maxlag + 1]
    acf /= acf[0]
    tau = 0.5
    for t in range(1, maxlag + 1):
        tau += acf[t]
        if t >= c * tau:
            return tau, t
    return tau, maxlag


def second_moment_z(G0, Gp, L):
    phat2 = 4.0 * np.sin(np.pi / L) ** 2
    mu2 = phat2 / (G0 / Gp - 1.0)
    mP = 2.0 * np.arcsinh(np.sqrt(mu2) / 2.0)
    return L * mP


GL200_NODES, GL200_WEIGHTS = np.polynomial.legendre.leggauss(240)


def conditional_cf(S2, S3, alphas, *, lam, m0sq, V, L, halfwidth_sigmas=12.0, chunk=20000):
    """Rao-Blackwellised characteristic function C(alpha) = E[cos(alpha X)] at h=0, per config.

    Returns array (n_alpha, n_config) of E[cos(alpha X) | psi]. Lee-Yang zeros of Z(eta) sit at
    eta = +- i alpha_j where C(alpha_j) = 0.
    """
    S2 = np.asarray(S2, float); S3 = np.asarray(S3, float)
    alphas = np.asarray(alphas, float)
    n = S2.size
    out = np.empty((alphas.size, n))
    a4 = lam * V
    for s in range(0, n, chunk):
        e = slice(s, min(n, s + chunk))
        a2 = 0.5 * m0sq * V + 6.0 * lam * S2[e]
        a1 = 4.0 * lam * S3[e]
        u0 = _mode(a4, a2, a1)
        curv = 12 * a4 * u0**2 + 2 * a2
        w = halfwidth_sigmas / np.sqrt(curv)
        x = u0[:, None] + w[:, None] * GL200_NODES[None, :]
        f = a4 * x**4 + a2[:, None] * x**2 + a1[:, None] * x
        f0 = a4 * u0**4 + a2 * u0**2 + a1 * u0
        dens = np.exp(-(f - f0[:, None])) * GL200_WEIGHTS[None, :] * w[:, None]
        Z = dens.sum(axis=1)
        X = L * x
        for i, al in enumerate(alphas):
            out[i, e] = (dens * np.cos(al * X)).sum(axis=1) / Z
    return out
