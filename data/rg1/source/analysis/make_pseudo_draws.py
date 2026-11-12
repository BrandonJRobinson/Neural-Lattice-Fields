"""Build archive-format bootstrap draws from independent local ensembles (project design emulation).

anchor/z: h=0 ensemble (subset of size n0); q,s at each radius: independent direct finite-source runs.
Block bootstrap within each ensemble; the same draw index pairs (anchor, radii) as in the frozen policy.
Variants: raw moments or exact uniform-mode conditional (RB) moments.
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import zm  # noqa: E402

rng = np.random.default_rng(20261028)
L, m0sq, lam = 12, -0.6223, 0.4
V = L ** 4
out = sys.argv[1]
n0 = int(sys.argv[2])            # number of h=0 configurations used for the anchor
nsrc = int(sys.argv[3])          # number of configurations per source run used
mode = sys.argv[4]               # raw | rb
ndraw = int(sys.argv[5]) if len(sys.argv) > 5 else 400
nb = 100

h0 = zm.load("data/L12_h0_s1201.bin")
h0 = {k: v[:n0] for k, v in h0.items()}
c0 = zm.conditional(h0["S2"], h0["S3"], lam=lam, m0sq=m0sq, V=V, L=L, kmax=2)
X0 = L * h0["u"]
x2 = c0["mom"][2] if mode == "rb" else X0 ** 2
B0 = np.stack([zm.blocks_mean(x2, nb), zm.blocks_mean(h0["Gp"], nb)])

src = {}
for eta, f in [("4.5", "data/L12_eta4p5_s1245.bin"), ("6.0", "data/L12_eta6_s1260.bin")]:
    d = zm.load(f)
    d = {k: v[:nsrc] for k, v in d.items()}
    e = float(eta)
    if mode == "rb":
        c = zm.conditional(d["S2"], d["S3"], lam=lam, m0sq=m0sq, V=V, L=L, eta=e, kmax=2)
        m1, m2 = c["mom"][1], c["mom"][2]
    else:
        X = L * d["u"]; m1, m2 = X, X * X
    src[eta] = np.stack([zm.blocks_mean(m1, nb), zm.blocks_mean(m2, nb)])

draws = []
for b in range(ndraw):
    i0 = rng.integers(0, nb, nb)
    a1 = B0[0, i0].mean(); gp = B0[1, i0].mean()
    z = float(zm.second_moment_z(a1 * L * L, gp, L))
    q, s = {}, {}
    for eta, arr in src.items():
        ii = rng.integers(0, nb, nb)
        mu = arr[0, ii].mean(); mm = arr[1, ii].mean()
        q[eta] = float(mu / float(eta)); s[eta] = float(mm - mu * mu)
    draws.append({"z": z, "a1": float(a1), "q": q, "s": s})
json.dump({"volumes": [{"L": L, "label": f"local emulation {mode} n0={n0} nsrc={nsrc}", "draws": draws}]}, open(out, "w"))
print("wrote", out)
