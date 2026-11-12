"""Compare measured 1/g_P(L) with exact finite-lattice one-loop PT at each run's measured z.

Two conventions (both from the frozen/Oct-27 record):
  MPT_frozen : y(L) - y(12) = 1.5 [I2(L, z_L/L) - I2(12, z_12/12)]           (literal frozen kernel)
  MPT_scheme : g_P = q_L g_SM, q_L = (m_P/mu_L)^4, mu_L = 2 sinh(z_L/(2L));
               1/g_SM(L) = 1/g_SM(12) + 1.5 [I2(L, mu_L) - I2(12, mu_12)]   (Oct-27 scheme-matched)
I2(L,m) = L^-4 sum_p (phat^2 + m^2)^-2 over ALL momenta (incl. p=0).
Anchor: measured g at L=12.  Errors: measured statistical errors only (PT remainder NOT bounded).
"""
import json, sys
import numpy as np
b1 = 3/(16*np.pi**2)

def I2(L, m):
    p = 2*np.pi*np.arange(L)/L; ph2 = 4*np.sin(p/2)**2
    t = ph2[:,None,None,None]+ph2[None,:,None,None]+ph2[None,None,:,None]+ph2[None,None,None,:]
    return np.sum(1/(t+m*m)**2)/L**4

rows = json.load(open(sys.argv[1]))
rows = sorted(rows, key=lambda r: r["L"])
ref = [r for r in rows if r["L"] == 12][0]
g12, e12 = ref["g_RB"]; z12 = ref["z_RB"][0]
mu12 = 2*np.sinh(z12/24); q12 = ((z12/12)/mu12)**4
out = []
for r in rows:
    L = r["L"]; g, e = r["g_RB"]; z = r["z_RB"][0]
    y = 1/g; ey = e/g**2
    dy_meas = y - 1/g12
    edy = np.sqrt(ey**2 + (e12/g12**2)**2) if L != 12 else 0.0
    dy_frozen = 1.5*(I2(L, z/L) - I2(12, z12/12))
    mu = 2*np.sinh(z/(2*L)); qL = ((z/L)/mu)**4
    ysm12 = q12/g12
    y_pred = (ysm12 + 1.5*(I2(L, mu) - I2(12, mu12)))/qL
    dy_scheme = y_pred - 1/g12
    out.append({"L": L, "z": z, "g": [g, e], "dy_meas": [dy_meas, edy], "dy_MPT_frozen": dy_frozen,
                "dy_MPT_scheme": dy_scheme, "dy_continuum_b1log": b1*np.log(L/12)})
    print(f"L={L:3d} z={z:.4f} g={g:.4f}+-{e:.4f}  dy_meas={dy_meas:+.5f}+-{edy:.5f}  MPT_frozen={dy_frozen:+.5f}  MPT_scheme={dy_scheme:+.5f}  b1*log={b1*np.log(L/12):+.5f}")
json.dump(out, open(sys.argv[2], "w"), indent=1)
