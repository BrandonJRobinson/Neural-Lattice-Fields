"""Independent raw-to-primary replay using the pinned routines, no RNG/sampler.

Computes only the zero-source observables and the eta=4.5 reweighting control,
instead of regenerating the archival eleven-radius diagnostic scan. All 228
original per-chain QC checks are replayed. Original files remain untouched.
"""
from __future__ import annotations
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[_k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,math,time,hashlib,importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.integrate import quad_vec

B=Path(__file__).resolve().parent
ROOT=B/'extracted/RG1_RAW_REPLAY_20261111/rg1'
OUT=B/'primary_replay';OUT.mkdir(exist_ok=True)
CACHE=OUT/'chains';CACHE.mkdir(exist_ok=True)
os.environ['RG1_ROOT']=str(ROOT)
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'stage2_release_v1'))
from rg1.frozen import zm,require
from rg1.qc import chain_qc
from rg1.analyze_stage1 import interpolate,classify,REF_L12,per_chain_g
from rg1.analyze_stage2 import fit_g

def save(p,x):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2,default=float)+'\n');os.replace(tmp,p)
def read(p):return json.loads(p.read_text())
def compare_qc(q,old):
    deltas={}
    def close(a,b,path=''):
        if isinstance(a,dict):return set(a)==set(b) and all(close(a[k],b[k],path+'/'+k) for k in a)
        if isinstance(a,list):return a==b
        if isinstance(a,(int,float)):
            deltas[path]=float(a-b)
            return math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-9)
        return a==b
    keys=['n','acc_hb','acc_or','zm_skipped_meas','tau_int','block_length_over_50tau','split_chain_z','means','status','reasons']
    ok=all(close(q[k],old[k],k) for k in keys)
    return ok,deltas
def obs(a,L):
    x2,x4,gp=a;z=zm.second_moment_z(x2*L*L,gp,L)
    U4=1-x4/(3*x2*x2)
    return np.array([z,U4,3*z**4*U4,x2])
def qref(s2,s3,L,m,lam):
    a4=lam*L**4;a2=.5*m*L**4+6*lam*s2;a1=4*lam*s3
    u0=float(zm._mode(a4,np.array([a2]),np.array([a1]))[0])
    curv=12*a4*u0*u0+2*a2;scale=math.sqrt(curv)
    f1=4*a4*u0**3+2*a2*u0+a1
    def integrand(s):
        d=s/scale;pot=f1*d+.5*s*s+4*a4*u0*d**3+a4*d**4
        if not np.isfinite(pot) or pot>700:return np.zeros(5)
        X=L*(u0+d);return np.array([1,X,X*X,X**3,X**4])*math.exp(-pot)
    total,error=quad_vec(integrand,-np.inf,np.inf,epsabs=1e-12,epsrel=1e-11)
    return total/total[0],float(error/total[0])

def process(task):
    stage,t=task;key=f"s{stage}_{t['ensemble']}_{t['chain']:02d}";path=CACHE/(key+'.json')
    if path.exists():
        cached=read(path);d=ROOT/f'stage{stage}_data'/t['ensemble']/f"chain_{t['chain']:02d}"
        cached['archived_qc_match'],cached['qc_numeric_differences']=compare_qc(cached['qc'],read(d.parent/(d.name+'.qc.json')))
        save(path,cached);return cached
    start=time.time();d=ROOT/f'stage{stage}_data'/t['ensemble']/f"chain_{t['chain']:02d}"
    meta=read(d/'chain.json');L=meta['L'];m=meta['m0sq'];lam=meta['lambda'];h=meta['h']
    q=chain_qc(d/'chain.bin',t,meta);old=read(d.parent/(d.name+'.qc.json'))
    qc_match,qc_differences=compare_qc(q,old)
    a=zm.load(d/'chain.bin');X=L*a['u'];block=lambda x:zm.blocks_mean(x,8)
    result={'key':key,'stage':stage,'ensemble':t['ensemble'],'chain':t['chain'],'L':L,'m0sq':m,'lambda':lam,'h':h,
            'N':len(X),'qc':q,'archived_qc_match':qc_match,'qc_numeric_differences':qc_differences,'blocks_raw':[block(X*X).tolist(),block(X**4).tolist(),block(a['Gp']).tolist()]}
    if h==0:
        c=zm.conditional(a['S2'],a['S3'],lam=lam,m0sq=m,V=L**4,L=L)
        blocks=[block(c['mom'][2]),block(c['mom'][4]),block(a['Gp'])]
        v,e,jk=zm.jackknife(lambda b:obs(b,L),blocks)
        result.update(blocks_rb=[x.tolist() for x in blocks],estimate=v.tolist(),error=e.tolist())
        # Independent adaptive full-line integration on actual extrema and time samples.
        a2=.5*m*L**4+6*lam*a['S2']
        indices=sorted(set([0,len(X)//2,len(X)-1,int(np.argmin(a2)),int(np.argmax(np.abs(a['S3'])))]))
        probes=[]
        for i in indices:
            ref,qe=qref(a['S2'][i],a['S3'][i],L,m,lam)
            got=c['mom'][:,i];err=np.abs(got-ref)/np.maximum(np.abs(ref),1e-14)
            probes.append({'row':i,'relative_error_r2':float(err[2]),'relative_error_r4':float(err[4]),'adaptive_error_estimate':qe})
        result['quadrature_checks']=probes
        result['min_conditional_a2']=float(a2.min())
        if stage==1 and t['ensemble'] in ['P24_hi','P32_hi']:
            s2=np.r_[a['S2'],a['S2']];s3=np.r_[a['S3'],-a['S3']]
            ce=zm.conditional(s2,s3,lam=lam,m0sq=m,V=L**4,L=L,eta=4.5,kmax=2)
            c0=zm.conditional(s2,s3,lam=lam,m0sq=m,V=L**4,L=L,kmax=0)
            lw=ce['logZ']-c0['logZ'];shift=float(lw.max());w=np.exp(lw-shift);n=len(X)
            result['eta4p5']={'log_scale':shift,'blocks':[block(w[:n]+w[n:]).tolist(),
                block(w[:n]*ce['mom'][1][:n]+w[n:]*ce['mom'][1][n:]).tolist(),
                block(w[:n]*ce['mom'][2][:n]+w[n:]*ce['mom'][2][n:]).tolist()]}
    else:
        result['direct_blocks']=[block(X).tolist(),block(X*X).tolist()]
    result['elapsed_s']=time.time()-start
    save(path,result);return result

def ensemble(rows,L):
    blk=[np.concatenate([np.array(r['blocks_rb'][k]) for r in rows]) for k in range(3)]
    val,err,jk=zm.jackknife(lambda b:obs(b,L),blk)
    braw=[np.concatenate([np.array(r['blocks_raw'][k]) for r in rows]) for k in range(3)]
    rv,re,rjk=zm.jackknife(lambda b:obs(b,L),braw)
    gs=np.array([r['estimate'][2] for r in rows]);zs=np.array([r['estimate'][0] for r in rows]);es=np.array([r['error'][2] for r in rows])
    scatter=float(gs.std(ddof=1)/np.sqrt(len(rows)));w=1/np.maximum(es,1e-100)**2;mean=np.sum(w*gs)/w.sum()
    cov=(len(jk)-1)/len(jk)*(jk[:,[0,2]]-jk[:,[0,2]].mean(axis=0)).T@(jk[:,[0,2]]-jk[:,[0,2]].mean(axis=0))
    return {'N':sum(r['N'] for r in rows),'n_chains':len(rows),'n_blocks':len(jk),'z_RB':[val[0],err[0]],'g_RB':[val[2],err[2]],
            'g_raw':[rv[2],re[2]],'a1_raw':[rv[3],re[3]],'U4_RB':[val[1],err[1]],
            'g_err_used':max(float(err[2]),scatter),'z_err_used':max(float(err[0]),float(zs.std(ddof=1)/np.sqrt(len(rows)))),
            'replica_scatter_g':scatter,'replica_chi2':float(np.sum(w*(gs-mean)**2)),
            'reconstructed_joint_covariance_order':['z','g'],'reconstructed_joint_jackknife_covariance':cov.tolist(),
            'joint_jackknife_z_g':jk[:,[0,2]].tolist()}

def main():
    require();assert read(B/'RAW_INTEGRITY_VERIFICATION.json')['status']=='PASS'
    p1=read(ROOT/'stage1_release_v1/PLAN.json');p2=read(ROOT/'stage2_release_v1/PLAN.json')
    tasks=[(1,t) for t in p1['tasks']]+[(2,t) for t in p2['tasks']]
    start=time.time();records=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(process,t):t for t in tasks}
        for f in as_completed(futures):
            r=f.result();records.append(r)
            print(f"{len(records)}/228 {r['key']} QC={r['qc']['status']} archived_match={r['archived_qc_match']} {r['elapsed_s']:.1f}s",flush=True)
    records.sort(key=lambda r:(r['stage'],r['ensemble'],r['chain']))
    ens={}
    for stage,key in sorted({(r['stage'],r['ensemble']) for r in records if r['h']==0}):
        rows=[r for r in records if r['stage']==stage and r['ensemble']==key]
        ens[key]=ensemble(rows,rows[0]['L'])
    for L in [12,16,24,32]:
        for side in ['lo','hi']:
            keys=[f'Q{L}_{side}']+([f'P{L}_{side}'] if L>=24 else [])
            rows=[r for k in keys for r in records if r['ensemble']==k]
            # Original order: Stage 1 P then Stage 2 Q, each in ascending chain index.
            rows.sort(key=lambda r:(r['stage'],r['chain']))
            ens[f'L{L}_{side}']=ensemble(rows,L)
    adj=read(ROOT/'stage2_data/adjudication_20261031T025902Z/STAGE2_RESULT.json')
    z3={};checks={}
    for L in [12,16,24,32]:
        lo,hi=[ens[f'L{L}_{side}'] for side in ['lo','hi']]
        z3[str(L)]=interpolate(*[{'z':[s['z_RB'][0],s['z_err_used']],'g':[s['g_RB'][0],s['g_err_used']]} for s in [lo,hi]])
        want=adj['z3'][str(L)]['g_at_z3'];got=z3[str(L)]['g_at_z3']
        checks[f'g_z3_L{L}']={'replayed':got,'archived':want,'match':all(math.isclose(x,y,rel_tol=1e-9,abs_tol=1e-11) for x,y in zip(got,want))}
    Ls=np.array([12.,16.,24.,32.]);g=np.array([z3[str(int(L))]['g_at_z3'][0] for L in Ls]);eg=np.array([z3[str(int(L))]['g_at_z3'][1] for L in Ls])
    fits={}
    for model in ['M1','MCUT']:
        p,e,c,dof,_=fit_g(Ls,g,eg,np.zeros(4),np.zeros(4),model)
        want=adj['fits']['running_fits_g']['L12-32'][model]['B'];got=[float(p[1]),float(e[1])]
        fits[model]={'B':got,'chi2':c,'dof':dof}
        checks[model]={'replayed':got,'archived':want,'match':all(math.isclose(x,y,rel_tol=1e-5,abs_tol=1e-8) for x,y in zip(got,want))}
    # Original Stage 1 control formulas, using recomputed ensemble moments.
    s1=read(ROOT/'stage1_data/adjudication_local_20261029T100334Z/STAGE1_ADJUDICATION.json')
    ctl={}
    zg=lambda s:{'z':[s['z_RB'][0],s['z_err_used']],'g':[s['g_RB'][0],s['g_err_used']]}
    for name,a,b in [('L12_reproduction',zg(ens['C12_repro']),REF_L12),
                     ('cross_kernel_C12_local',zg(ens['C12_local']),zg(ens['C12_repro'])),
                     ('cross_kernel_C16_local',zg(ens['C16_local']),zg(ens['C16_gibbs']))]:
        ctl[name]={q:{'sigma':(a[q][0]-b[q][0])/math.hypot(a[q][1],b[q][1])} for q in ['z','g']}
        for q in ['z','g']:ctl[name][q]['status']=classify(ctl[name][q]['sigma'])
    s=ens['C08_gauss'];exact=1/(8*8*.5)
    dev=max(abs(s['g_raw'][0]/s['g_raw'][1]),abs((s['a1_raw'][0]-exact)/s['a1_raw'][1]))
    ctl['lambda0']={'status':classify(dev),'max_abs_sigma':dev}
    for sk,pk in [('S24_eta4p5','P24_hi'),('S32_eta4p5','P32_hi')]:
        source=[r for r in records if r['ensemble']==sk];source.sort(key=lambda r:r['chain'])
        direct=[np.concatenate([r['direct_blocks'][k] for r in source]) for k in range(2)]
        dv,de,_=zm.jackknife(lambda a:np.array([a[0]/4.5,a[1]-a[0]**2]),direct)
        ref=[r for r in records if r['ensemble']==pk];ref.sort(key=lambda r:r['chain']);scale=max(r['eta4p5']['log_scale'] for r in ref)
        wb=[np.concatenate([np.array(r['eta4p5']['blocks'][k])*np.exp(r['eta4p5']['log_scale']-scale) for r in ref]) for k in range(3)]
        rv,re,_=zm.jackknife(lambda a:np.array([a[1]/a[0]/4.5,a[2]/a[0]-(a[1]/a[0])**2]),wb)
        pulls=(dv-rv)/np.hypot(de,re)
        ctl['reweighting_identity_'+sk]={'direct':[dv.tolist(),de.tolist()],'reweighted':[rv.tolist(),re.tolist()],
                                      'pulls':pulls.tolist(),'status':max([classify(x) for x in pulls],key=['PASS','WARN','FAIL'].index)}
    def statuses(x,prefix=''):
        out={}
        if isinstance(x,dict):
            for k,v in x.items():
                if k=='status':out[prefix]=v
                else:out.update(statuses(v,prefix+'/'+k))
        return out
    checks['stage1_controls']={'replayed':statuses(ctl),'archived':statuses(s1['controls']),'match':statuses(ctl)==statuses(s1['controls'])}
    checks['all_228_qc']={'match':all(r['qc']['status']=='PASS' and r['archived_qc_match'] for r in records)}
    probes=[p for r in records for p in r.get('quadrature_checks',[])]
    qcheck={'actual_rows_checked':len(probes),'rule':'first, middle, last, minimum a2 and maximum |S3| per zero-source chain; duplicates removed',
            'reference':'independent adaptive full-real-line scipy quad_vec, same conditional polynomial',
            'max_relative_r2_error':max(p['relative_error_r2'] for p in probes),'max_relative_r4_error':max(p['relative_error_r4'] for p in probes),
            'scope':'deterministic selected-row accuracy check; not a rigorous all-row quadrature bound'}
    save(OUT/'REPLAY_RESULT.json',{'status':'PASS' if all(v['match'] for v in checks.values()) else 'DIFFERENCES_REQUIRE_REVIEW',
        'scope':'raw-to-primary zero-source coupling, all per-chain QC and Stage 1 controls; full diagnostic radius scan not regenerated',
        'new_simulations':False,'new_synthetic_draws':False,'elapsed_s':time.time()-start,'z3':z3,'fits':fits,'controls':ctl,'quadrature':qcheck,'checks':checks,
        'calibration':'original archived calibration summary retained; not regenerated','lee_yang_grid':'historical diagnostic-grid checks retained; not regenerated',
        'ensembles':ens})
    print('FINAL',json.dumps({'checks':checks,'quadrature':qcheck,'fits':fits},default=float),flush=True)

if __name__=='__main__':main()
