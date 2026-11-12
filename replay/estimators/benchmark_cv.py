"""Existing-row SD control-variate benchmark; no sampler, RNG or TeX invocation."""
from __future__ import annotations
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse, csv, hashlib, importlib.util, json, math, platform, sys, time
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import least_squares

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
OLD=ROOT/'raw_replay_20261111'
RG=Path(os.environ.get('NNLFT_RG1_ROOT',str(OLD/'extracted/RG1_RAW_REPLAY_20261111/rg1')))
REFERENCE=Path(os.environ.get('NNLFT_COMPARISON_ROOT',str(ROOT/'conditional_estimator_audit_20261112')))
SOURCE=RG/'source/analysis/zm.py'
if not SOURCE.exists():SOURCE=OUT/'pinned_source/zm.py'
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('pinned_cv_zm',SOURCE)
zm=importlib.util.module_from_spec(spec);spec.loader.exec_module(zm)
METHODS=['raw','conditional','SD4','SD8','SD12','SD24','SD24_r1e-4','SD24_r1e-2']
MODELS=[(4,1e-6),(8,1e-6),(12,1e-6),(24,1e-6),(24,1e-4),(24,1e-2)]
def read(p):return json.loads(p.read_text())
def save(p,x):
    p.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else float(a))+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for s in iter(lambda:f.read(8*1024*1024),b''):h.update(s)
    return h.hexdigest()
def timed(fun):
    a=time.perf_counter();b=time.process_time();value=fun()
    return value,dict(wall_s=time.perf_counter()-a,cpu_s=time.process_time()-b)
def load_rows(path):
    v=np.fromfile(path,dtype=np.float64).reshape(-1,4)
    return [v[:,i].copy() for i in range(4)]
def raw_blocks(a,L):
    u,s2,s3,q=a;X=L*u;f=X*X
    return np.array([zm.blocks_mean(f,8),zm.blocks_mean(f*f,8),zm.blocks_mean(q,8)]).T
def conditional_blocks(a,L,m):
    u,s2,s3,q=a
    r=zm.conditional(s2,s3,lam=.4,m0sq=m,V=L**4,L=L)
    return np.array([zm.blocks_mean(r['mom'][2],8),zm.blocks_mean(r['mom'][4],8),zm.blocks_mean(q,8)]).T
def controls(a,L,m):
    u,s2,s3,q=a;t=3*L*u
    b=(.5*m*L**4+2.4*s2)/(9*L*L);c=1.6*s3/(3*L)
    force=4*(.4/81)*t**3+2*b*t+c
    h=[np.ones_like(t),t]
    for j in range(1,7):h.append((t*h[j]-math.sqrt(j)*h[j-1])/math.sqrt(j+1))
    A=[(-force if j==0 else math.sqrt(j)*h[j-1]-force*h[j]) for j in range(8)]
    odd=[A[j] for j in [1,3,5,7]];even=[A[j] for j in [0,2,4,6]];d=b-.5
    return np.column_stack(odd+[c*x for x in even]+[d*x for x in odd]+[c*c*x for x in odd]+[c*d*x for x in even]+[c**3*x for x in even])
def sd_statistics(a,L,m):
    n=len(a[0]);assert n%8==0
    B=n//8;result=[]
    for k in range(8):
        aa=[v[k*B:(k+1)*B] for v in a]
        C=controls(aa,L,m)
        X=L*aa[0];v=X*X;Y=np.column_stack([v,v*v,aa[3]])
        result.append(dict(n=B,sum_y=Y.sum(axis=0),sum_c=C.sum(axis=0),sum_cc=C.T@C,sum_cy=C.T@Y))
    return result

def make_record(ref,index):
    output=OUT/'chains'/f"{ref['key']}.json"
    if output.exists():return read(output)
    path=RG/ref['input_path'];digest=sha(path)
    assert digest==ref['input_sha256'] and path.stat().st_size==32*ref['N']
    a=load_rows(path);assert all(np.isfinite(v).all() for v in a)
    # Warm functions outside timings on the same fixed small prefix.
    sub=[v[:1024] for v in a]
    raw_blocks(sub,ref['L']);conditional_blocks(sub,ref['L'],ref['m0sq']);sd_statistics(sub,ref['L'],ref['m0sq'])
    results={};times={}
    order=['raw','conditional','sd']
    order=order[index%3:]+order[:index%3]
    for name in order:
        fun={'raw':lambda:raw_blocks(a,ref['L']),
             'conditional':lambda:conditional_blocks(a,ref['L'],ref['m0sq']),
             'sd':lambda:sd_statistics(a,ref['L'],ref['m0sq'])}[name]
        results[name],times[name]=timed(fun)
    assert np.allclose(results['raw'],np.array(ref['blocks']['raw']).T,rtol=2e-12,atol=1e-14)
    assert np.allclose(results['conditional'],np.array(ref['blocks']['conditional']).T,rtol=2e-12,atol=1e-14)
    result={k:ref[k] for k in ['key','L','side','stage','chain','m0sq','N','input_path','metadata']}
    result.update(input_sha256=digest,fold=ref['chain']%2,raw=results['raw'],conditional=results['conditional'],
                  sd=results['sd'],timings=times,timing_order=order)
    save(output,result);return read(output)

def stack_endpoint(records):
    data={k:[] for k in ['raw','conditional','fold','n','sum_y','sum_c','sum_cc','sum_cy']}
    chain_slices=[]
    for row in records:
        start=len(data['fold']);data['raw'].extend(row['raw']);data['conditional'].extend(row['conditional'])
        for block in row['sd']:
            data['fold'].append(row['fold'])
            for k in ['n','sum_y','sum_c','sum_cc','sum_cy']:data[k].append(block[k])
        chain_slices.append((start,start+8))
    data={k:np.asarray(v) for k,v in data.items()}
    data['slices']=chain_slices
    assert len(set(data['n']))==1
    assert sum(data['fold']==0)==sum(data['fold']==1)
    return data

def fit_beta(data,train,k,ridge):
    n=data['n'][train].sum();sC=data['sum_c'][train,:k].sum(axis=0);sY=data['sum_y'][train].sum(axis=0)
    CC=(data['sum_cc'][train,:k,:k].sum(axis=0)-np.outer(sC,sC)/n)/(n-1)
    CY=(data['sum_cy'][train,:k].sum(axis=0)-np.outer(sC,sY)/n)/(n-1)
    std=np.sqrt(np.maximum(CC.diagonal(),1e-100));R=CC/np.outer(std,std)
    beta=np.linalg.solve(R+ridge*np.eye(k),CY/std[:,None])/std[:,None]
    assert np.isfinite(beta).all()
    return beta
def moments(data,keep=None,details=False):
    if keep is None:keep=np.ones(len(data['fold']),dtype=bool)
    n=data['n'][keep].sum();assert len(set(data['n'][keep]))==1
    v=[data['raw'][keep].mean(axis=0),data['conditional'][keep].mean(axis=0)]
    all_beta=[]
    for k,ridge in MODELS:
        total=np.zeros(3);betas=[]
        for fold in [0,1]:
            train=keep&(data['fold']!=fold);test=keep&(data['fold']==fold)
            assert train.any() and test.any() and not (train&test).any()
            B=fit_beta(data,train,k,ridge);betas.append(B)
            # Evaluate UNcentered controls: their exact population mean is zero.
            total+=data['sum_y'][test].sum(axis=0)-data['sum_c'][test,:k].sum(axis=0)@B
        v.append(total/n);all_beta.append(betas)
    return (np.asarray(v),all_beta) if details else np.asarray(v)
def observable(v,L):
    m2,m4,q=v.T
    z=zm.second_moment_z(m2*L*L,q,L)
    g=3*z**4*(1-m4/(3*m2*m2))
    result=np.column_stack([z,g])
    assert np.isfinite(result).all()
    return result
def endpoint(data,L,nblocks):
    full=observable(moments(data),L)
    reps=[];groups=[]
    width=8//nblocks
    for start,end in data['slices']:
        for i in range(start,end,width):groups.append(np.arange(i,i+width))
    for group in groups:
        keep=np.ones(len(data['fold']),dtype=bool);keep[group]=False
        reps.append(observable(moments(data,keep),L))
    return dict(full=full,reps=np.asarray(reps))
def interpolate(lo,hi):
    t=(3-lo[:,0])/(hi[:,0]-lo[:,0]);g=(1-t)*lo[:,1]+t*hi[:,1]
    assert np.isfinite(g).all()
    return g,t
def match(lo,hi):
    value,t=interpolate(lo['full'],hi['full']);C=np.zeros((len(METHODS),len(METHODS)))
    for side in ['lo','hi']:
        reps=lo['reps'] if side=='lo' else hi['reps']
        r=np.asarray([interpolate(v,hi['full'])[0] if side=='lo' else interpolate(lo['full'],v)[0] for v in reps])
        n=len(r);d=r-r.mean(axis=0);C+=(n-1)/n*(d.T@d)
    se=np.sqrt(C.diagonal());details={}
    for i,name in enumerate(METHODS):
        diff=value[i]-value[1];ds=math.sqrt(max(C[i,i]+C[1,1]-2*C[i,1],0))
        details[name]=dict(g=float(value[i]),se=float(se[i]),variance_over_conditional=float(C[i,i]/C[1,1]),
                          raw_variance_gain=float(C[0,0]/C[i,i]),difference_from_conditional=float(diff),
                          paired_difference_se=ds,paired_pull=float(diff/ds) if ds>0 else 0,
                          interpolation_fraction=float(t[i]))
    return dict(methods=details,covariance=C,method_order=METHODS,all_inside_brackets=bool(((t>0)&(t<1)).all()))

def fit_slopes(result):
    L=np.array([12.,16.,24.,32.]);D=np.column_stack([np.ones(4),np.log(L/12),L**-2-12.**-2])
    out={}
    for method in METHODS:
        g=np.array([result[str(int(x))]['blocking']['8']['methods'][method]['g'] for x in L])
        se=np.array([result[str(int(x))]['blocking']['8']['methods'][method]['se'] for x in L])
        f=least_squares(lambda p:(1/(D@p)-g)/se,[1/g[0],.021,.59],xtol=1e-13,gtol=1e-13,ftol=1e-13)
        err=np.sqrt(np.linalg.inv(f.jac.T@f.jac)[1,1])
        out[method]=dict(B=float(f.x[1]),se=float(err),chi2=float(sum(f.fun**2)),dof=1,
                         scope='secondary common model using benchmark joint-jackknife errors, not adopted primary envelopes')
    return out
def subsets(records):
    out=[]
    for L in [12,16,24,32]:
        for side in ['lo','hi']:
            d=next(r for r in records if r['L']==L and r['side']==side)
            a=np.memmap(RG/d['input_path'],mode='r',dtype=np.float64,shape=(d['N'],4))
            inds=np.linspace(0,len(a)-1,65536,dtype=int);chunk=np.array(a[inds]);del a
            aa=[chunk[:,j].copy() for j in range(4)]
            for rep in range(3):
                order=['raw','conditional','sd'];order=order[rep:]+order[:rep]
                for name in order:
                    fn={'raw':lambda:raw_blocks(aa,L),'conditional':lambda:conditional_blocks(aa,L,d['m0sq']),'sd':lambda:sd_statistics(aa,L,d['m0sq'])}[name]
                    val,t=timed(fn);out.append(dict(L=L,side=side,repeat=rep,method=name,measurements=len(aa[0]),**t))
            print('subset',L,side,flush=True)
    return out
def main():
    start=time.perf_counter();reference=read(REFERENCE/'COMPARISON_RESULT.json')
    assert sha(SOURCE)==reference['source_sha256']
    refs=[read(p) for p in sorted((REFERENCE/'chains').glob('*.json'))]
    refs.sort(key=lambda d:(d['L'],d['side'],d['stage'],d['chain']))
    assert len(refs)==200 and sum(d['N'] for d in refs)==123988000
    records=[]
    for i,d in enumerate(refs):
        r=make_record(d,i);records.append(r)
        print(f"{i+1}/200 {d['key']} raw={r['timings']['raw']['cpu_s']:.3f} cond={r['timings']['conditional']['cpu_s']:.3f} SD={r['timings']['sd']['cpu_s']:.3f}",flush=True)
    results={};fits={};endpoint_archive={}
    for L in [12,16,24,32]:
        groups={s:stack_endpoint([r for r in records if r['L']==L and r['side']==s]) for s in ['lo','hi']}
        t=time.process_time();endpoint_archive[str(L)]={}
        for side,data in groups.items():
            mom,beta=moments(data,details=True)
            endpoint_archive[str(L)][side]=dict(moments=mom,coefficients=beta,model_order=METHODS[2:],
                fold_chains={str(f):sum(r['L']==L and r['side']==side and r['fold']==f for r in records) for f in [0,1]})
        point_fit_cpu=time.process_time()-t;blocks={};jk_times={}
        for nb in [8,4,2,1]:
            t=time.process_time();ep=[endpoint(groups[s],L,nb) for s in ['lo','hi']];blocks[str(nb)]=match(*ep)
            assert blocks[str(nb)]['all_inside_brackets']
            jk_times[str(nb)]=time.process_time()-t
            old=reference['results'][str(L)]['blocking'][str(nb)]
            assert np.allclose([blocks[str(nb)]['methods'][m]['g'] for m in METHODS[:2]],old['value'],rtol=1e-11,atol=1e-11)
            assert np.allclose([blocks[str(nb)]['methods'][m]['se'] for m in METHODS[:2]],old['se'],rtol=1e-9,atol=1e-11)
            print('BLOCK',L,nb,json.dumps({m:(round(v['g'],7),round(v['se'],7),round(v['variance_over_conditional'],5)) for m,v in blocks[str(nb)]['methods'].items()}),flush=True)
        ds=[r for r in records if r['L']==L]
        results[str(L)]=dict(blocking=blocks,
            local_cpu_s={m:sum(r['timings'][m]['cpu_s'] for r in ds) for m in ['raw','conditional','sd']},
            coefficient_fit_cpu_s=point_fit_cpu,jackknife_refit_cpu_s=jk_times,
            recorded_generation_hours=sum(r['metadata']['meas_seconds']+r['metadata']['therm_seconds'] for r in ds)/3600)
    save(OUT/'ENDPOINT_COEFFICIENTS.json',endpoint_archive)
    subfile=OUT/'REPEATED_TIMING.json'
    if not subfile.exists():save(subfile,subsets(records))
    report=dict(status='PASS',primary_comparator='SD24',ridge=1e-6,methods=METHODS,results=results,
        slopes=fit_slopes(results),scope='new deterministic analysis of recovered primary measurements; two-fold independent-chain cross-fitting; coefficient refit in every deletion',
        chains=200,rows=123988000,fold_rule='within each stage/endpoint, chain index parity; evaluate each chain with opposite-fold coefficients',
        optimization='row-level covariance minimization, standardized controls, fixed ridge; evaluation uses uncentered zero-mean controls',
        source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.txt'),reference_result_sha256=sha(REFERENCE/'COMPARISON_RESULT.json'),
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),numerical_thread_limit=1),
        wall_s=time.perf_counter()-start,new_simulations=False,new_random_draws=False)
    save(OUT/'CV_BENCHMARK_RESULT.json',report)
    with (OUT/'MATCHED_COMPARISON.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['L','method','g','se','variance_over_conditional','raw_variance_gain','paired_difference','paired_difference_se','paired_pull'])
        for L,x in results.items():
            for name,v in x['blocking']['8']['methods'].items():w.writerow([L,name,v['g'],v['se'],v['variance_over_conditional'],v['raw_variance_gain'],v['difference_from_conditional'],v['paired_difference_se'],v['paired_pull']])
    print('COMPLETE',OUT/'CV_BENCHMARK_RESULT.json',flush=True)

if __name__=='__main__':main()
