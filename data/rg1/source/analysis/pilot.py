import sys, numpy as np
sys.path.insert(0, "analysis")
import zm
L, m0sq, lam = int(sys.argv[2]), float(sys.argv[3]), 0.4
V = L**4
d = zm.load(sys.argv[1])
u, S2, S3, Gp = d["u"], d["S2"], d["S3"], d["Gp"]
X = L*u
c = zm.conditional(S2, S3, lam=lam, m0sq=m0sq, V=V, L=L)
r1, r2, r3, r4 = c["mom"][1:5]
v = r2 - r1**2
print("N", u.size)
print("Var X raw", X.var(), " E[v]", v.mean(), " Var(r1)", r1.var(), " f=E[v]/VarX", v.mean()/X.var())
print("mu_c^2 = (m0^2+12 lam <S2>/V) =", m0sq + 12*lam*S2.mean()/V)
for name, s in [("X^2", X**2), ("r2", r2), ("S2", S2), ("S3", S3), ("X^4-6aX^2", X**4-6*X.var()*X**2), ("r4-6a r2", r4-6*X.var()*r2), ("Gp", Gp)]:
    t, w = zm.tau_int(s)
    print(f"tau_int[{name}] = {t:.2f} (window {w})")
G0 = V*np.mean(r2)/L**2
z = zm.second_moment_z(G0, Gp.mean(), L)
U4raw = 1-np.mean(X**4)/(3*np.mean(X**2)**2); U4rb = 1-np.mean(r4)/(3*np.mean(r2)**2)
print("z =", z, " U4 raw", U4raw, " U4 RB", U4rb, " g raw", 3*z**4*U4raw, " g RB", 3*z**4*U4rb)
infl_raw = X**4-6*np.mean(X**2)*X**2; infl_rb = r4-6*np.mean(r2)*r2
print("per-config variance ratio RB/raw (kappa4 influence):", infl_rb.var()/infl_raw.var())
