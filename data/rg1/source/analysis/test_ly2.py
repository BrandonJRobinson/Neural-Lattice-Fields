import sys, numpy as np
sys.path.insert(0, "analysis")
import ly_bracket as lb
x = np.linspace(-8, 8, 200001)
dens = np.exp(-x**2-0.05*x**4)
m2 = (dens*x**2).sum()/dens.sum(); m4 = (dens*x**4).sum()/dens.sum(); U4 = 1-m4/(3*m2**2); a1=m2
taus, R, D = [], [], []
for t in [1.5, 2.0]:
    eta = t/np.sqrt(a1); w = dens*np.exp(eta*x); Z = w.sum(); m1=(w*x).sum()/Z; s=(w*x*x).sum()/Z-m1*m1
    q = m1/eta; tau = eta*eta*a1/2; taus.append(tau); R.append(q/a1); D.append((q-s)/(2*tau*a1))
print("true U4", U4)
for sig in [0, 1e-5, 1e-4, 1e-3]:
    e = lb.ly_estimate(taus, R, D, [sig]*2, [sig]*2)
    print(sig, {k: round(v, 6) for k, v in e.items()})
