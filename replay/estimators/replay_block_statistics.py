"""Portable deterministic inference replay from included block sufficient statistics."""
from pathlib import Path
import json
import benchmark_cv as b
import numpy as np

HERE=Path(__file__).resolve().parent

def main():
    records=[b.read(p) for p in sorted((HERE/'chains').glob('*.json'))]
    report=b.read(HERE/'CV_BENCHMARK_RESULT.json')
    assert len(records)==200 and sum(r['N'] for r in records)==123988000
    assert b.sha(b.SOURCE)==report['source_sha256']
    maximum=0.;replayed={}
    for L in [12,16,24,32]:
        data={s:b.stack_endpoint([r for r in records if r['L']==L and r['side']==s]) for s in ['lo','hi']}
        replayed[str(L)]={}
        for blocks in [8,4,2,1]:
            actual=b.match(*[b.endpoint(data[s],L,blocks) for s in ['lo','hi']])
            expected=report['results'][str(L)]['blocking'][str(blocks)]
            difference=np.max(np.abs(actual['covariance']-np.asarray(expected['covariance'])))
            maximum=max(maximum,float(difference))
            assert np.allclose(actual['covariance'],expected['covariance'],rtol=2e-7,atol=1e-10)
            for method in b.METHODS:
                for key in ['g','se','interpolation_fraction','paired_difference_se','variance_over_conditional']:
                    x=actual['methods'][method][key];y=expected['methods'][method][key]
                    assert np.isclose(x,y,rtol=2e-7,atol=1e-10),(L,blocks,method,key,x,y)
            replayed[str(L)][str(blocks)]=actual
    output=dict(status='PASS',chains=200,rows=123988000,methods=b.METHODS,blockings=[8,4,2,1],
                maximum_covariance_absolute_discrepancy=maximum,results=replayed,
                scope='inference replay from included sufficient statistics; does not reread or certify unavailable raw binaries',
                simulations=False,random_draws=False,tex_execution=False)
    b.save(HERE/'BLOCK_REPLAY_VERIFICATION.json',output)
    print(json.dumps({k:v for k,v in output.items() if k!='results'},indent=2))

if __name__=='__main__':main()
