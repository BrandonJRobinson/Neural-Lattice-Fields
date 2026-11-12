import sys, numpy as np
sys.path.insert(0, "analysis")
import zm
L, m0sq, lam = 4, -0.30, 0.4
V = L**4
for tag in ["zm", "nozm"]:
    d = zm.load(f"data/val_L4_{tag}.bin")
    X = L*d["u"]
    c = zm.conditional(d["S2"], d["S3"], lam=lam, m0sq=m0sq, V=V, L=L)
    r2, r4 = c["mom"][2], c["mom"][4]
    nb = 200
    B = [zm.blocks_mean(X**2, nb), zm.blocks_mean(X**4, nb), zm.blocks_mean(r2, nb), zm.blocks_mean(r4, nb)]
    raw = lambda a: 1 - a[1]/(3*a[0]**2)
    rb = lambda a: 1 - a[3]/(3*a[2]**2)
    v1, e1, _ = zm.jackknife(lambda a: a[0], B)
    v2, e2, _ = zm.jackknife(lambda a: a[2], B)
    u1, eu1, _ = zm.jackknife(raw, B)
    u2, eu2, _ = zm.jackknife(rb, B)
    print(tag, f"<X^2> raw {v1:.5f}+-{e1:.5f}  RB {v2:.5f}+-{e2:.5f} | U4 raw {u1:.5f}+-{eu1:.5f} RB {u2:.5f}+-{eu2:.5f}")
