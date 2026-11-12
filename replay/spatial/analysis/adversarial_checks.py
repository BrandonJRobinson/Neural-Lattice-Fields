"""Independent, deterministic referee checks; no stochastic configurations."""
from pathlib import Path
from itertools import product
import json

import numpy as np
from scipy.special import roots_hermitenorm
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
prior = json.loads((ROOT/'evidence'/'ANALYTIC_CHECKS.json').read_text())
assert prior['result'] == 'PASS'

# Exact finite-frequency population and trigonometric quadrature through order four.
angles = np.array([[0, 0, 0, 0], [0, 0, 0, np.pi/2], [0, 0, np.pi/2, 0],
                   [0, np.pi/2, 0, 0], [0, np.pi, 0, np.pi/2], [0, np.pi/2, np.pi, 0]])
pk = np.array([1, 2, 3, 4, 2, 3], float)
pk /= pk.sum()
phase = 2*np.pi*np.arange(8)/8
base = np.sqrt(2)*np.cos(angles[:, None, :]+phase[None, :, None])
K = np.cos(angles[:, :, None]-angles[:, None, :])
C = np.einsum('s,sij->ij', pk, K)
assert np.linalg.eigvalsh(C).min() > .01
EKK = (np.einsum('s,sij,skl->ijkl', pk, K, K)
       + np.einsum('s,sik,sjl->ijkl', pk, K, K)
       + np.einsum('s,sil,sjk->ijkl', pk, K, K))
Wick = (np.einsum('ij,kl->ijkl', C, C)
        + np.einsum('ik,jl->ijkl', C, C)
        + np.einsum('il,jk->ijkl', C, C))
probes = np.vstack((np.eye(4), np.ones(4), [1, -1, 1, -1], [1, 2, -1, 3]))
max_tensor_error = 0.
max_projection_error = 0.
for q in [1., 1.4, 2., 3.]:
    amplitudes = np.array([0, np.sqrt(q), -np.sqrt(q)])
    pa = np.array([1-1/q, 1/(2*q), 1/(2*q)])
    state = (base[:, :, None, :]*amplitudes[None, None, :, None]).reshape(-1, 4)
    probability = (pk[:, None, None]*np.ones((1, 8, 1))/8*pa[None, None, :]).reshape(-1)
    for N in [1, 2]:
        if N == 1:
            z, p = state, probability
        else:
            z = ((state[:, None, :]+state[None, :, :])/np.sqrt(2)).reshape(-1, 4)
            p = (probability[:, None]*probability[None, :]).reshape(-1)
        cov = np.einsum('s,si,sj->ij', p, z, z)
        assert np.allclose(cov, C, rtol=1e-13, atol=1e-13)
        quadratic = (z[:, :, None]*z[:, None, :]).reshape(-1, 16)
        raw4 = ((quadratic.T*p)@quadratic).reshape(4, 4, 4, 4)
        direct = raw4-Wick
        predicted = (q*EKK/2-Wick)/N
        error = np.max(np.abs(direct-predicted))
        assert error < 2e-12
        max_tensor_error = max(max_tensor_error, float(error))
        for a in probes:
            values = z@a
            variance = np.dot(p, values**2)
            excess = (np.dot(p, values**4)-3*variance**2)/variance**2
            h = np.einsum('i,sij,j->s', a, K, a)
            rho = np.dot(pk, h*h)/(a@C@a)**2
            expected = 3*(q*rho/2-1)/N
            error = abs(excess-expected)
            assert error < 2e-12
            max_projection_error = max(max_projection_error, float(error))

# The polarization coefficient is verified independently by its multinomial identity.
x = sp.symbols('x0:4')
polarized = sum(np.prod(signs)*sum(e*a for e, a in zip(signs, x))**4
                for signs in product([-1, 1], repeat=4))/sp.Integer(384)
assert sp.expand(polarized-sp.prod(x)) == 0

# Independent Gaussian integration of the circle feature, retaining the coarse rule.
L = prior['input_parameters']['L']
theta = 2*np.pi*np.arange(L)/L
quadrature = []
for sigma_key in ['sigma1', 'sigma2']:
    sigma = float(prior['input_parameters'][sigma_key])
    for nodes in [128, 256, 384]:
        gh_x, gh_w = roots_hermitenorm(nodes)
        gh_w /= np.sqrt(2*np.pi)
        F = np.zeros((nodes, nodes), dtype=np.complex128)
        for angle in theta:
            F += (np.exp(1j*sigma*gh_x*np.cos(angle))[:, None]
                  * np.exp(1j*sigma*gh_x*np.sin(angle))[None, :])/L
        f2 = np.abs(F)**2
        A = float(gh_w@f2@gh_w)
        B = float(gh_w@(f2*f2)@gh_w)
        reference = prior['ring_moments'][0 if sigma_key == 'sigma1' else 1]
        discrepancy = max(abs(A-reference['A']), abs(B-reference['B']))
        if nodes >= 256:
            assert discrepancy < 1e-11
        quadrature.append(dict(sigma=sigma, nodes=nodes, A=A, B=B,
                               maximum_absolute_discrepancy=discrepancy,
                               converged_rule=nodes >= 256))
w = float(prior['input_parameters']['mixture_weight'])
rho_checks = []
for nodes in [256, 384]:
    a, b = [r for r in quadrature if r['nodes'] == nodes]
    rho = (w*a['B']**2+(1-w)*b['B']**2)/(w*a['A']**2+(1-w)*b['A']**2)**2
    assert float(prior['certified_decimal_enclosure']['lower']) < rho < float(prior['certified_decimal_enclosure']['upper'])
    rho_checks.append(dict(nodes=nodes, rho_M=rho))

report = dict(result='PASS', schema='nnlft-adversarial-source-review-v1',
    exact_population_description='Finite spectral law and exact phase quadrature through fourth order; weighted enumeration, no random draws',
    amplitude_fourth_moments=[1., 1.4, 2., 3.], widths=[1, 2],
    complete_fourth_tensor_components_checked_per_case=256,
    maximum_full_tensor_discrepancy=max_tensor_error,
    maximum_projected_kurtosis_discrepancy=max_projection_error,
    polarization_coefficient='1/(2^4*4!)=1/384', polarization_identity='PASS',
    independent_Gaussian_quadrature=quadrature, independent_rho_M_checks=rho_checks,
    coarse_quadrature_retained=True,
    new_simulations=False, new_random_draws=0, new_empirical_data=False,
    tex_compiled=False, pdf_rendering=False)
(ROOT/'evidence'/'ADVERSARIAL_CHECKS.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
