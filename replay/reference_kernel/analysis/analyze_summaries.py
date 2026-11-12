#!/usr/bin/env python3
"""Independent-replication inference from saved summaries only; no sampler imports."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
OBS=['phi2','abs_M','chi_abs','binder_U4','xi_over_L','S_pmin']
MASS=[.25,.5,1.,2.,4.,8.]
SEEDS=[11001,11002,11003,11004]
LS=[16,24,32]
ALPHA=.05


def write_csv(name,rows):
    with (ROOT/'evidence'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def holm(ps):
    p=np.array(ps);order=np.argsort(p);out=np.empty(len(p))
    out[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return out


def main():
    old={r['relative_path']:r for r in csv.DictReader((ROOT/'records/transfer_hash_verification.csv').open())}
    files=sorted((ROOT/'records/final').glob('*.json'));assert len(files)==72
    records=[];manifest=[]
    for p in files:
        b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
        expected=old['results/p3_reference_kernel_invariance_rescue_v1/raw/'+p.name]
        assert h==expected['sha256']
        d=json.loads(b);assert d['campaign']=='p3_reference_kernel_invariance_rescue_v1'
        assert d['m0_sq']==-3.84 and d['lambda']==1 and d['dimension']==2
        assert d['effective_seed']==d['seed']+1000000*d['L']+round(10000*d['m_ref_sq'])
        records.append(d);manifest.append(dict(file=p.name,sha256=h,transfer_hash_match=True))
    idx={(d['L'],d['m_ref_sq'],d['seed']):d for d in records}
    assert set(idx)=={(L,m,s) for L in LS for m in MASS for s in SEEDS}
    assert len({d['effective_seed'] for d in records})==72
    x=np.log(MASS);c=(x-x.mean())/np.sum((x-x.mean())**2)
    assert abs(c.sum())<1e-14 and np.isclose(c@x,1)
    q=float(t.ppf(1-ALPHA/2,3));q18=float(t.ppf(1-ALPHA/(2*18),3));q108=float(t.ppf(1-ALPHA/(2*108),3))
    frozen=json.loads((ROOT/'records/archived_verdict.json').read_text())
    old_fits={(r['L'],r['observable']):r for r in frozen['physics_rows']}
    trend=[];replicate=[];means=[];ranges=[];covrows=[];checks=[];projections=[]
    for L in LS:
        contrast_vectors=[]
        for ob in OBS:
            y=np.array([[idx[L,m,s]['estimates'][ob]['value'] for s in SEEDS] for m in MASS])
            assert np.isfinite(y).all()
            center=y.mean(axis=1);between=np.std(y,axis=1,ddof=1)/2
            assert (between>0).all()
            ds=c@y;b=float(ds.mean());se=float(np.std(ds,ddof=1)/2)
            assert np.isclose(b,c@center,rtol=1e-12,atol=1e-15)
            assert np.isclose(se**2,np.sum((ds-b)**2)/12)
            statistic=b/se;pref=float(2*t.sf(abs(statistic),3));psafe=pref if pref<=.05 else 1.
            terms=c[:,None]*(y-center[:,None])
            aligned=np.sort(terms,axis=1).sum(axis=0)
            env=float(np.sqrt(np.sum(aligned**2)/12))
            assert env>=se-1e-14
            # Check rearrangement deterministically on actual three-reference
            # subsets, enumerating all allocations (no random permutations).
            subset=terms[[0,2,5]];largest=-np.inf
            for p1 in itertools.permutations(range(4)):
                for p2 in itertools.permutations(range(4)):
                    v=subset[0]+subset[1,list(p1)]+subset[2,list(p2)]
                    largest=max(largest,float(v@v/12))
            expected=float(np.sum(np.sort(subset,axis=1).sum(axis=0)**2)/12)
            assert np.isclose(largest,expected,rtol=1e-11,atol=1e-20)
            checks.append(dict(L=L,observable=ob,mean_identity=True,
                               envelope_contains_fixed_grouping=True,
                               exhaustive_three_reference_rearrangement=True))
            row=dict(L=L,observable=ob,beta=b,replicate_se=se,T3=statistic,
                     p_t3_reference=pref,p_safe=psafe,
                     individual95_lower=b-q*se,individual95_upper=b+q*se,
                     simultaneous18_lower=b-q18*se,simultaneous18_upper=b+q18*se,
                     grouping_envelope_se=env,
                     envelope_individual95_lower=b-q*env,envelope_individual95_upper=b+q*env,
                     envelope_simultaneous18_lower=b-q18*env,
                     envelope_simultaneous18_upper=b+q18*env)
            trend.append(row);contrast_vectors.append(ds)
            for s,dv in zip(SEEDS,ds):
                replicate.append(dict(L=L,observable=ob,seed_label=s,contrast=float(dv),
                                      distinct_chains=6,common_random_numbers=False))
            lo=center-q108*between;hi=center+q108*between
            # Project the simultaneous run-mean rectangle using the ORIGINAL
            # data-dependent fitting weights. Coverage of the whole rectangle
            # protects any selected zero-sum contrast under constant means.
            old=old_fits[L,ob]
            old_se=np.array([old['values'][str(float(m))]['se'] for m in MASS])
            w=old_se**-2;wx=x-np.dot(w,x)/sum(w)
            aw=w*wx/np.dot(w,wx**2)
            bw=float(aw@center)
            assert np.isclose(bw,old['log_mref_slope'],rtol=1e-9,atol=1e-12)
            assert abs(aw.sum())<1e-12
            lower=float(np.sum(np.where(aw>=0,aw*lo,aw*hi)))
            upper=float(np.sum(np.where(aw>=0,aw*hi,aw*lo)))
            projections.append(dict(L=L,observable=ob,archived_weighted_slope=bw,
                                    projected_lower=lower,projected_upper=upper,
                                    zero_in_projection=lower<=0<=upper,
                                    common_mean_intersection_lower=float(max(lo)),
                                    common_mean_intersection_upper=float(min(hi)),
                                    constant_reference_model_intersects=bool(max(lo)<=min(hi))))
            for mi,m in enumerate(MASS):
                means.append(dict(L=L,m_ref_sq=m,observable=ob,run_mean=float(center[mi]),
                                  between_chain_se=float(between[mi]),
                                  individual95_lower=float(center[mi]-q*between[mi]),
                                  individual95_upper=float(center[mi]+q*between[mi]),
                                  simultaneous108_lower=float(lo[mi]),simultaneous108_upper=float(hi[mi])))
            ub=max(hi[a]-lo[b1] for a in range(6) for b1 in range(6) if a!=b1)
            lb=max(0,max(lo)-min(hi))
            ranges.append(dict(L=L,observable=ob,observed_grid_mean=float(center.mean()),
                               observed_range=float(np.ptp(center)),
                               simultaneous108_range_lower=float(lb),simultaneous108_range_upper=float(ub),
                               equivalence_margin_prespecified=False))
        cv=np.cov(np.array(contrast_vectors),ddof=1)/4
        assert np.linalg.matrix_rank(cv,tol=np.max(abs(cv))*1e-10)<=3
        for j,ob in enumerate(OBS):
            for k,ob2 in enumerate(OBS):
                covrows.append(dict(L=L,observable=ob,observable_2=ob2,covariance=float(cv[j,k])))
    adjusted=holm([r['p_safe'] for r in trend])
    for r,a in zip(trend,adjusted):r['p_safe_holm18']=float(a)
    write_csv('input_summary_hashes.csv',manifest)
    write_csv('replicate_trends.csv',trend)
    write_csv('replicate_contrasts.csv',replicate)
    write_csv('run_mean_intervals.csv',means)
    write_csv('reference_spread_bounds.csv',ranges)
    write_csv('contrast_covariance_descriptive.csv',covrows)
    write_csv('original_weight_confidence_projections.csv',projections)
    write_csv('deterministic_checks.csv',checks)
    low=min(trend,key=lambda r:r['p_t3_reference'])
    result=dict(schema='nnlft-p3-summary-inference-v1',summary_files_verified=72,
                independent_effective_seeds=72,reference_points=18,observables=6,
                replicate_groups_per_contrast=4,scalar_trend_tests=18,
                alpha=ALPHA,t3_critical_individual95=q,t3_critical_simultaneous18=q18,
                t3_critical_simultaneous108=q108,
                min_reference_tail_area=dict(L=low['L'],observable=low['observable'],p=low['p_t3_reference']),
                min_safe_holm18=float(min(adjusted)),familywise_05_rejections=int(sum(adjusted<=.05)),
                L32_reference_tail_area_range=[min(r['p_t3_reference'] for r in trend if r['L']==32),max(r['p_t3_reference'] for r in trend if r['L']==32)],
                simultaneous108_range_lower_bounds_all_zero=all(r['simultaneous108_range_lower']==0 for r in ranges),
                original_weight_projections_all_contain_zero=all(r['zero_in_projection'] for r in projections),
                all_18_reference_constant_subspaces_intersect_rectangle=all(r['constant_reference_model_intersects'] for r in projections),
                normal_centered_independent_replica_model_assumed=True,
                equivalence_established=False,equilibration_established=False,
                archived_weighted_fit_recalibrated=False,new_random_draws=0,new_simulations=False,
                latex_compiled=False,pdf_rendering=False)
    (ROOT/'evidence/SUMMARY_INFERENCE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    print('L32 spread bounds (observed, simultaneous-108 upper):')
    for r in ranges:
        if r['L']==32:print(r['observable'],r['observed_range'],r['simultaneous108_range_upper'])


if __name__=='__main__':main()
