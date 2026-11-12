import sys, json, numpy as np
sys.path.insert(0, "analysis")
import ly_bracket as lb
# Synthetic sanity test 1: X = 1D quartic exp(-x^4) ; compute its exact R(tau), U4 by quadrature, then bracket.
x = np.linspace(-8, 8, 200001); dx = x[1]-x[0]
def mgf_stats(dens, eta):
    w = dens*np.exp(eta*x); Z = w.sum()
    m1 = (w*x).sum()/Z; m2 = (w*x*x).sum()/Z
    return m1, m2 - m1*m1
for name, dens in [("pure quartic", np.exp(-x**4)), ("gauss+weak quartic", np.exp(-x**2-0.05*x**4))]:
    m2 = (dens*x**2).sum()/dens.sum(); m4 = (dens*x**4).sum()/dens.sum()
    U4 = 1 - m4/(3*m2**2)
    a1 = m2
    taus, R, D = [], [], []
    # choose eta so that t = eta sqrt(a1) = 1.5, 2 (as the frozen window at L12)
    for t in [1.5, 2.0]:
        eta = t/np.sqrt(a1)
        m1, s = mgf_stats(dens, eta)
        q = m1/eta
        tau = eta*eta*a1/2
        taus.append(tau); R.append(q/a1); D.append((q-s)/(2*tau*a1))
    br = lb.lp_bracket(taus, R, D)
    brq = lb.lp_bracket(taus, R, D, use_D=False)
    print(name, "U4 true", round(U4,6), " LP(q+s)", round(br["U4_min"],6), round(br["U4_max"],6), " LP(q only)", round(brq["U4_min"],6), round(brq["U4_max"],6))
