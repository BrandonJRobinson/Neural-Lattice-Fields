"""Locate Lee-Yang zeros alpha_j as sign changes of the RB characteristic function C(alpha)."""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import zm
out = sys.argv[1]; L = int(sys.argv[2]); m0sq = float(sys.argv[3]); lam = float(sys.argv[4]); nb = int(sys.argv[5])
files = sys.argv[6:]
V = L**4
parts = [zm.load(f) for f in files]
d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
c0 = zm.conditional(d["S2"], d["S3"], lam=lam, m0sq=m0sq, V=V, L=L, kmax=2)
a1 = c0["mom"][2].mean()
alphas = np.linspace(0.5, 30, 119)
C = zm.conditional_cf(d["S2"], d["S3"], alphas, lam=lam, m0sq=m0sq, V=V, L=L)
Cb = zm.blocks_mean(C, nb)
mean = Cb.mean(axis=1); err = Cb.std(axis=1, ddof=1)/np.sqrt(nb)
gauss = np.exp(-alphas**2*a1/2)
zeros = []
for i in range(len(alphas)-1):
    if mean[i] > 0 and mean[i+1] <= 0 or mean[i] < 0 and mean[i+1] >= 0:
        a0 = alphas[i] - mean[i]*(alphas[i+1]-alphas[i])/(mean[i+1]-mean[i])
        # significance of the bracketing values
        zeros.append({"alpha": float(a0), "w": float(2/(a1*a0*a0)),
                      "C_left": [float(mean[i]), float(err[i])], "C_right": [float(mean[i+1]), float(err[i+1])]})
res = {"L": L, "a1": float(a1), "alphas": alphas.tolist(), "C": mean.tolist(), "C_err": err.tolist(),
       "gaussian_envelope": gauss.tolist(), "zeros": zeros}
json.dump(res, open(out, "w"), indent=1)
for a, m, e, g in zip(alphas[::6], mean[::6], err[::6], gauss[::6]):
    print(f"alpha {a:6.2f}  C {m: .3e} +- {e:.1e}   gaussian {g:.3e}")
print("zeros:", json.dumps(zeros, indent=1))
