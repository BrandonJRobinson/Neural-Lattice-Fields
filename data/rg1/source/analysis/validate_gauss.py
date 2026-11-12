import sys, numpy as np
sys.path.insert(0, "analysis")
import zm
L, m0sq = 6, 0.3
V = L**4
d = zm.load("data/val_gauss_L6.bin")
u, Gp = d["u"], d["Gp"]
nb = 100
G0b = zm.blocks_mean(V*u**2, nb); Gpb = zm.blocks_mean(Gp, nb)
G0, eG0, _ = zm.jackknife(lambda a: a[0], [G0b])
GP, eGP, _ = zm.jackknife(lambda a: a[0], [Gpb])
phat2 = 4*np.sin(np.pi/L)**2
print("G(0) =", G0, "+-", eG0, " exact", 1/m0sq)
print("G(pmin) =", GP, "+-", eGP, " exact", 1/(phat2+m0sq))
k = np.mean(u**4)/np.mean(u**2)**2
print("kurtosis", k, "(3 exact)")
# psi second moment: <S2> = sum_{p != 0} 1/(phat^2+m^2)
ps = np.array(np.meshgrid(*[2*np.pi*np.arange(L)/L]*4, indexing='ij'))
ph2 = (4*np.sin(ps/2)**2).sum(0)
exS2 = (1/(ph2+m0sq)).sum() - 1/m0sq
print("<S2> =", d["S2"].mean(), " exact", exS2)
print("<S3> =", d["S3"].mean(), " exact 0")
