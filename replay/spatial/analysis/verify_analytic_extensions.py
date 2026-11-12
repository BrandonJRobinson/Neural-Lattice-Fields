"""Deterministic checks of the added analytic statements; no random draws."""
from pathlib import Path
from fractions import Fraction
from itertools import product
import hashlib
import json
import re

import mpmath as mp
import numpy as np
import scipy
from scipy.optimize import linprog
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'parameters.json'
parameters = json.loads(source.read_text())
sigma1 = parameters['sigma1']
sigma2 = parameters['sigma2']
weight = parameters['mixture_weight']
assert (sigma1, sigma2, weight) == ('5.053427193613765', '1.9002050977792333', '0.8013785646741373')

# Exact phase moments and projection coefficient.
b = sp.symbols('b', real=True)
phase2 = sp.integrate(sp.cos(b)**2, (b, 0, 2*sp.pi))/(2*sp.pi)
phase4 = sp.integrate(sp.cos(b)**4, (b, 0, 2*sp.pi))/(2*sp.pi)
assert phase2 == sp.Rational(1, 2) and phase4 == sp.Rational(3, 8)
q, rho, n, N, gamma = sp.symbols('q rho n N gamma', positive=True)
normalized = (sp.Rational(3, 2)*q*rho-3)/N
assert sp.simplify(normalized-3*(q*rho/2-1)/N) == 0
qbal = 4/(1+rho)
assert sp.simplify((1-qbal/2)-(qbal*rho/2-1)) == 0
rmin = 2*n/(n+2)
assert sp.simplify((rmin-1)/(rmin+1)-(n-2)/(3*n+2)) == 0
q_required = 2*(1+N*gamma/3)/rho
assert sp.simplify(3*(q_required*rho/2-1)/N-gamma) == 0
epsilon, vf, vg, k4 = sp.symbols('epsilon vf vg k4', positive=True)
eta = epsilon**2*vf/(vg+epsilon**2*vf)
assert sp.simplify(epsilon**4*k4/(vg+epsilon**2*vf)**2-eta**2*k4/vf**2) == 0

# Independent LP solution of the amplitude problem, including both sides of the kink.
lp_results = []
for rr in [1., 1.1, 1.2, 2., 3., 3.1, 13.789004407156714, 100.]:
    A = [[.5, -1], [-.5, -1], [rr/2, -1], [-rr/2, -1]]
    result = linprog([0, 1], A_ub=A, b_ub=[1, -1, 1, -1],
                     bounds=[(1, None), (0, None)], method='highs')
    q_expected = max(1, 4/(1+rr))
    loss_expected = (rr-1)/(rr+1) if rr <= 3 else rr/2-1
    assert result.success
    assert abs(result.x[0]-q_expected) < 1e-9 and abs(result.fun-loss_expected) < 1e-9
    lp_results.append(dict(rho_star=rr, q4=float(result.x[0]), loss_without_3_over_N=float(result.fun)))

# An exact finite-frequency model verifies the projected fourth polynomial and attainment.
# Independent signs give C=I and rho(a)=3-2 sum(a_i^4)/(sum(a_i^2))^2.
x = sp.symbols('x0:3')
moment = sum(sum(a*s for a, s in zip(x, signs))**4
             for signs in product([-1, 1], repeat=3))/8
expected = 3*sum(a*a for a in x)**2-2*sum(a**4 for a in x)
assert sp.expand(moment-expected) == 0
assert 3*(sp.Rational(6, 5)/2-1) == -sp.Rational(6, 5)
assert 3*(sp.Rational(6, 5)*sp.Rational(7, 3)/2-1) == sp.Rational(6, 5)

# Full L^4 quadruple sums and rotation-reduced L^3 sums, independently arranged.
L = 16
angles = 2*np.pi*np.arange(L)/L
u = np.c_[np.cos(angles), np.sin(angles)]
pair = (u[:, None, :]+u[None, :, :]).reshape(L*L, 2)
quad_d2 = ((pair[:, None, :]-pair[None, :, :])**2).sum(axis=-1)
pair_d2 = ((u[:, None, :]-u[None, :, :])**2).sum(axis=-1)
reduced_d2 = ((u[0]+u[:, None, None, :]-u[None, :, None, :]-u[None, None, :, :])**2).sum(axis=-1)
ring_rows = []
for sigma_text in [sigma1, sigma2]:
    sigma = float(sigma_text)
    A_full = np.exp(-sigma*sigma*pair_d2/2).mean()
    A_reduced = np.exp(-2*sigma*sigma*np.sin(np.pi*np.arange(L)/L)**2).mean()
    B_full = np.exp(-sigma*sigma*quad_d2/2).mean()
    B_reduced = np.exp(-sigma*sigma*reduced_d2/2).mean()
    assert np.isclose(A_full, A_reduced, rtol=2e-14, atol=1e-16)
    assert np.isclose(B_full, B_reduced, rtol=2e-14, atol=1e-16)
    ring_rows.append(dict(sigma=sigma_text, A=float(A_full), B=float(B_full)))
w = float(weight)
den = w*ring_rows[0]['A']**2+(1-w)*ring_rows[1]['A']**2
num = w*ring_rows[0]['B']**2+(1-w)*ring_rows[1]['B']**2
rho_float = num/den**2

# Outward-rounded interval arithmetic certifies the sign without statistical assumptions.
iv = mp.iv
iv.dps = 28
ui = [(iv.cos(2*iv.pi*i/L), iv.sin(2*iv.pi*i/L)) for i in range(L)]
interval_rows = []
for sigma_text in [sigma1, sigma2]:
    ss = iv.mpf(sigma_text)**2
    A = sum(iv.exp(-ss*sum((ui[i][k]-ui[j][k])**2 for k in range(2))/2)
            for i, j in product(range(L), repeat=2))/(L**2)
    B = sum(iv.exp(-ss*sum((ui[0][k]+ui[j][k]-ui[h][k]-ui[l][k])**2 for k in range(2))/2)
            for j, h, l in product(range(L), repeat=3))/(L**3)
    interval_rows.append((A, B))
wi = iv.mpf(weight)
den_i = wi*interval_rows[0][0]**2+(1-wi)*interval_rows[1][0]**2
num_i = wi*interval_rows[0][1]**2+(1-wi)*interval_rows[1][1]**2
rho_i = num_i/den_i**2
lo, hi = '13.78900440', '13.78900442'
assert rho_i.a > iv.mpf(lo) and rho_i.b < iv.mpf(hi)
assert rho_i.a > 3
assert float(lo) < rho_float < float(hi)
min_N = int((3*(Fraction(lo)/2-1))/Fraction('0.1'))+1
assert min_N == 177
cap = Fraction(3*3**4, 2*32)
assert cap == Fraction(243, 64) and float(cap) == 3.796875

report = dict(
    result='PASS', schema='nnlft-analytic-extension-checks-v1',
    input_source='parameters.json',
    input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    input_parameters=dict(L=L, sigma1=sigma1, sigma2=sigma2, mixture_weight=weight),
    symbolic_checks=['phase moments', 'projection coefficient', 'minimax balance',
                     'dimension floor', 'target fourth-order feasibility', 'Gaussian dilution'],
    independent_linear_programs=lp_results,
    exact_finite_sign_enumeration='PASS',
    full_and_rotation_reduced_finite_sums_agree=True,
    ring_moments=ring_rows, rho_M_float=rho_float,
    interval_precision_decimal_digits=iv.dps, rho_M_outward_interval=str(rho_i),
    certified_decimal_enclosure=dict(lower=lo, upper=hi, strict=True),
    rho_M_above_sign_threshold_2=True, rho_M_above_minimax_threshold_3=True,
    documented_kernel_optimal_amplitude_q4=1,
    minimum_N_times_magnetization_excess_kurtosis=3*(rho_float/2-1),
    necessary_width_for_uniform_excess_kurtosis_at_most_0_1=min_N,
    exact_rho_star_not_computed=True,
    coupling_cap_example=dict(z=3, N=32, exact_fraction=str(cap), decimal=float(cap)),
    dependencies=dict(numpy=np.__version__, sympy=sp.__version__, mpmath=mp.__version__, scipy=scipy.__version__),
    new_simulations=False, new_random_draws=0, new_empirical_data=False,
    tex_compiled=False, pdf_rendering=False)
(ROOT/'evidence'/'ANALYTIC_CHECKS.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
