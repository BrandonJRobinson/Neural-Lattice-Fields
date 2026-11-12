"""Running-design information calculations (independent reimplementation).

1. One-loop finite-lattice bubble I2(L,z) = L^-4 sum_p (phat^2 + (z/L)^2)^-2 (all momenta),
   Delta(1/g) = 1.5 [I2(L,z) - I2(12,z)] -- reproduces the frozen MPT kernel convention.
2. Local Fisher sigma_B for y_L = 1/g_L = A + B log(L/12) [+ C (L^-2 - 12^-2)] with given per-point
   relative precision sigma_g/g, at a reference trajectory g(12)=g12, B=b1.
"""
import json, numpy as np
b1 = 3/(16*np.pi**2)

def I2(L, z):
    p = 2*np.pi*np.arange(L)/L
    ph2 = 4*np.sin(p/2)**2
    tot = ph2[:,None,None,None]+ph2[None,:,None,None]+ph2[None,None,:,None]+ph2[None,None,None,:]
    return np.sum(1.0/(tot+(z/L)**2)**2)/L**4

res = {}
k = {L: I2(L, 3.0) for L in [12, 16, 24, 32, 48]}
res["one_loop_kernel_z3"] = {str(L): 1.5*(k[L]-k[12]) for L in k}
res["continuum_one_loop_12_to_32"] = b1*np.log(32/12)

def sigmaB(Ls, rel, g12=5.0, cutoff=True, extra_l4=False):
    Ls = np.array(Ls, float)
    y = 1/g12 + b1*np.log(Ls/12)
    g = 1/y
    sig_y = rel*g*y*y  # sigma(1/g) = sigma_g/g^2 = rel/g
    cols = [np.ones_like(Ls), np.log(Ls/12)]
    if cutoff: cols.append(Ls**-2 - 12.0**-2)
    if extra_l4: cols.append(Ls**-4 - 12.0**-4)
    Dm = np.array(cols).T
    F = Dm.T @ np.diag(1/sig_y**2) @ Dm
    C = np.linalg.inv(F)
    return float(np.sqrt(C[1,1]))

designs = {"4vol_12-32": [12,16,24,32], "5vol_+48": [12,16,24,32,48], "6vol_+48,64": [12,16,24,32,48,64],
           "wide_8-64": [8,12,16,24,32,48,64]}
tab = {}
for name, Ls in designs.items():
    for rel in [0.01, 0.015, 0.02, 0.05]:
        tab[f"{name}|rel={rel}"] = {"no_cutoff": sigmaB(Ls, rel, cutoff=False), "free_L-2": sigmaB(Ls, rel, cutoff=True)}
res["sigma_B"] = tab
# reproduce the Oct-5 review's equal sigma(1/g)=0.02 numbers
def sigmaB_abs(Ls, sy, cutoff):
    Ls = np.array(Ls, float); cols=[np.ones_like(Ls), np.log(Ls/12)]
    if cutoff: cols.append(Ls**-2-12.0**-2)
    Dm=np.array(cols).T; F=Dm.T@Dm/sy**2; return float(np.sqrt(np.linalg.inv(F)[1,1]))
res["oct5_check_sigma_y_0.02"] = {name: [sigmaB_abs(Ls,0.02,False), sigmaB_abs(Ls,0.02,True)] for name, Ls in
                                   {"none":[12,16,24,32],"+40":[12,16,24,32,40],"+48":[12,16,24,32,48],"+64":[12,16,24,32,64]}.items()}
# rel precision needed for 80% power two-sided (sigma_B = b1/2.80)
target = b1/2.80
need = {}
for name, Ls in designs.items():
    lo, hi = 1e-4, 0.2
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if sigmaB(Ls, mid, cutoff=True) > target: hi = mid
        else: lo = mid
    need[name] = lo
res["target_sigma_B_80pct"] = target
res["rel_precision_needed_free_cutoff_80pct"] = need
json.dump(res, open("data/design_info.json","w"), indent=1)
print(json.dumps(res, indent=1))
