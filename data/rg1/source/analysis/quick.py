import sys, numpy as np
sys.path.insert(0, "analysis")
import zm
L=int(sys.argv[1]); lam=float(sys.argv[2])
for f in sys.argv[3:]:
    m0sq=float(f.split("_m")[-1][:-4]); V=L**4
    d=zm.load(f); X=L*d["u"]
    c=zm.conditional(d["S2"],d["S3"],lam=lam,m0sq=m0sq,V=V,L=L)
    r2,r4=c["mom"][2],c["mom"][4]
    z=zm.second_moment_z(L*L*r2.mean(), d["Gp"].mean(), L)
    U4=1-r4.mean()/(3*r2.mean()**2); U4r=1-np.mean(X**4)/(3*np.mean(X**2)**2)
    f_=(r2-c["mom"][1]**2).mean()/np.mean(X**2)
    print(f"m0sq {m0sq}: z {z:.3f} g_RB {3*z**4*U4:.2f} g_raw {3*z**4*U4r:.2f} U4 {U4:.4f} f {f_:.3f} min a2/V {(0.5*m0sq+6*lam*d['S2']/V).min():.4f}")
