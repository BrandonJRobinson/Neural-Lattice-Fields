import sys, numpy as np
sys.path.insert(0, "analysis")
import ly_bracket as lb
rng = np.random.default_rng(5)
x = np.linspace(-8, 8, 200001)
dens = np.exp(-x**2-0.05*x**4)
m2 = (dens*x**2).sum()/dens.sum(); m4 = (dens*x**4).sum()/dens.sum(); U4 = 1-m4/(3*m2**2); a1=m2
taus, R, D = [], [], []
for t in [1.5, 2.0]:
    eta = t/np.sqrt(a1); w = dens*np.exp(eta*x); Z = w.sum(); m1=(w*x).sum()/Z; s=(w*x*x).sum()/Z-m1*m1
    q = m1/eta; tau = eta*eta*a1/2; taus.append(tau); R.append(q/a1); D.append((q-s)/(2*tau*a1))
taus=np.array(taus); R=np.array(R); D=np.array(D)
# cubic-equivalent (exact anchor): least squares of Gq,Gs combos -> U_cubic; here use the mean of secants & tangents
Ucub = np.mean(np.concatenate([(1-R)/taus, D]))
print("true U4", U4, " cubic-type (mean of secant/tangent) U", Ucub)
for sig in [3e-4, 1e-3]:
    ests, cubs, tst = [], [], []
    for _ in range(150):
        Rn = R + sig*rng.standard_normal(2); Dn = D + sig*rng.standard_normal(2)
        e = lb.ly_estimate(taus, Rn, Dn, [sig]*2, [sig]*2, ngrid=1500)
        ests.append(e["U_mid"]) if np.isfinite(e["U_mid"]) else None; tst.append(e["t_star"])
        cubs.append(np.mean(np.concatenate([(1-Rn)/taus, Dn])))
    ests=np.array(ests); cubs=np.array(cubs)
    print(f"sigma {sig:.0e}: LY mean {ests.mean():.5f} sd {ests.std():.5f} | cubic mean {cubs.mean():.5f} sd {cubs.std():.5f} | median t* {np.median(tst):.2f}")
