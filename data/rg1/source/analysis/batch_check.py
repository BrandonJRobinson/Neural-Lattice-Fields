"""Robustness of large-L errors: compare block-jackknife error of g with the scatter of K independent batch estimates."""
import json, sys
import numpy as np
sys.path.insert(0, "analysis"); import zm
out = {}
for L, m0, f in [(16,-0.657,"data/L16_h0_s1601.bin"),(24,-0.676,"data/L24_h0_s2401.bin"),(24,-0.6775,"data/L24b_h0_s2402.bin"),(32,-0.686,"data/L32_h0_s3201.bin")]:
    V=L**4; d=zm.load(f); n=d["u"].size
    c=zm.conditional(d["S2"],d["S3"],lam=0.4,m0sq=m0,V=V,L=L); r2,r4=c["mom"][2],c["mom"][4]
    def gfun(sl):
        z=zm.second_moment_z(L*L*r2[sl].mean(), d["Gp"][sl].mean(), L); return 3*z**4*(1-r4[sl].mean()/(3*r2[sl].mean()**2))
    g_all=gfun(slice(0,n))
    res={}
    for K in (4,8,10):
        gb=np.array([gfun(slice(k*n//K,(k+1)*n//K)) for k in range(K)])
        res[K]={"batch_err": float(gb.std(ddof=1)/np.sqrt(K)), "batch_mean": float(gb.mean())}
    out[f"{L}_{m0}"]={"g": float(g_all), "batches": res}
    print(L, m0, round(g_all,3), {K:(round(v["batch_mean"],2), round(v["batch_err"],2)) for K,v in res.items()})
json.dump(out, open("data/batch_check.json","w"), indent=1)
