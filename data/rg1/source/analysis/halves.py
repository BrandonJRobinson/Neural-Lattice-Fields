import sys, numpy as np
sys.path.insert(0,"analysis"); import zm
L=int(sys.argv[1]); m0=float(sys.argv[2]); lam=0.4; f=sys.argv[3]; V=L**4
d=zm.load(f); n=d["u"].size
c=zm.conditional(d["S2"],d["S3"],lam=lam,m0sq=m0,V=V,L=L)
r2,r4=c["mom"][2],c["mom"][4]
for k in range(4):
    s=slice(k*n//4,(k+1)*n//4)
    z=zm.second_moment_z(L*L*r2[s].mean(), d["Gp"][s].mean(), L)
    U=1-r4[s].mean()/(3*r2[s].mean()**2)
    print(f"quarter {k}: z {z:.4f}  g_RB {3*z**4*U:.3f}  <S2>/V {d['S2'][s].mean()/V:.6f}")
