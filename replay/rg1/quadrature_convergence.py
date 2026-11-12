"""Recompute existing zero-source rows with a stricter numerical integral.
160 nodes / 16 curvature widths versus archived 80 / 11. No RNG or sampler.
Frozen source files are unchanged; quadrature globals change in worker memory.
"""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import time,json,math
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
import replay_primary as r

OUT=r.B/'quadrature_convergence';OUT.mkdir(exist_ok=True)
CACHE=OUT/'chains';CACHE.mkdir(exist_ok=True)

def process(old):
    path=CACHE/(old['key']+'.json')
    if path.exists():return r.read(path)
    start=time.time();d=r.ROOT/f"stage{old['stage']}_data"/old['ensemble']/f"chain_{old['chain']:02d}"
    a=r.zm.load(d/'chain.bin');L=old['L']
    r.zm.GL_NODES,r.zm.GL_WEIGHTS=np.polynomial.legendre.leggauss(160)
    c=r.zm.conditional(a['S2'],a['S3'],lam=old['lambda'],m0sq=old['m0sq'],V=L**4,L=L,halfwidth_sigmas=16.0)
    blocks=[r.zm.blocks_mean(c['mom'][2],8),r.zm.blocks_mean(c['mom'][4],8),r.zm.blocks_mean(a['Gp'],8)]
    v,e,jk=r.zm.jackknife(lambda b:r.obs(b,L),blocks)
    rel=[float(np.max(np.abs(np.array(old['blocks_rb'][k])-blocks[k])/np.maximum(np.abs(blocks[k]),1e-30))) for k in [0,1]]
    result={'key':old['key'],'estimate':v.tolist(),'error':e.tolist(),'blocks_rb':[x.tolist() for x in blocks],
            'max_relative_block_r2_change':rel[0],'max_relative_block_r4_change':rel[1],'elapsed_s':time.time()-start}
    r.save(path,result);return result

def main():
    base=r.read(r.OUT/'REPLAY_RESULT.json')
    assert base['status']=='PASS'
    records=[r.read(p) for p in sorted(r.CACHE.glob('*.json'))];assert len(records)==228
    zero=[x for x in records if x['h']==0];assert len(zero)==224
    start=time.time();high={}
    with ProcessPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(process,x) for x in zero]
        for f in as_completed(fs):
            x=f.result();high[x['key']]=x
            print(f"{len(high)}/224 {x['key']} block r4 change={x['max_relative_block_r4_change']:.3g}",flush=True)
    new=[dict(x,**{k:high[x['key']][k] for k in ['estimate','error','blocks_rb']}) if x['h']==0 else x for x in records]
    z3={};changes={}
    for L in [12,16,24,32]:
        end=[]
        for side in ['lo','hi']:
            keys=[f'Q{L}_{side}']+([f'P{L}_{side}'] if L>=24 else [])
            rows=sorted([x for x in new if x['ensemble'] in keys],key=lambda x:(x['stage'],x['chain']))
            end.append(r.ensemble(rows,L))
        z3[str(L)]=r.interpolate(*[{'z':[s['z_RB'][0],s['z_err_used']],'g':[s['g_RB'][0],s['g_err_used']]} for s in end])
        old=base['z3'][str(L)]['g_at_z3'];got=z3[str(L)]['g_at_z3']
        changes[str(L)]={'baseline':old,'stricter':got,'delta_g':got[0]-old[0],
                         'delta_g_over_baseline_se':(got[0]-old[0])/old[1],'delta_se':got[1]-old[1]}
    status='PASS' if all(abs(x['delta_g_over_baseline_se'])<1e-6 and abs(x['delta_se'])<1e-8 for x in changes.values()) else 'REVIEW'
    result={'status':status,'baseline_rule':{'nodes':80,'halfwidth_sigmas':11},'stricter_rule':{'nodes':160,'halfwidth_sigmas':16},
            'chains_recomputed':len(high),'measured_rows_recomputed':sum(x['N'] for x in zero),
            'max_relative_block_r2_change':max(x['max_relative_block_r2_change'] for x in high.values()),
            'max_relative_block_r4_change':max(x['max_relative_block_r4_change'] for x in high.values()),
            'coupling_changes':changes,'elapsed_s':time.time()-start,'new_simulations':False,
            'scope':'full existing zero-source data reprocessing with stricter quadrature; numerical convergence evidence, not a rigorous arbitrary-integrand error bound'}
    r.save(OUT/'CONVERGENCE_RESULT.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
