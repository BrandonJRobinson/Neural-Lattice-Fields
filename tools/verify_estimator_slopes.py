"""Replay the archived secondary slope fits from freshly replayed estimator results."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    blocks = json.loads(args.blocks.read_text())
    reference = json.loads(args.reference.read_text())
    assert blocks['status'] == reference['status'] == 'PASS'
    assert blocks['chains'] == 200 and blocks['rows'] == 123988000
    assert not args.output.exists(), 'Choose a new output file'
    # Import the preserved function alongside the replayed block results.
    sys.path.insert(0, str(args.blocks.resolve().parent))
    os.environ.setdefault('NNLFT_RG1_ROOT',str(Path(__file__).resolve().parents[1]/'data/rg1'))
    import benchmark_cv as benchmark
    assert benchmark.sha(benchmark.SOURCE) == reference['source_sha256']
    result = {L:dict(blocking=values) for L,values in blocks['results'].items()}
    slopes = benchmark.fit_slopes(result)
    maximum = 0.
    for method, actual in slopes.items():
        expected = reference['slopes'][method]
        assert actual['dof'] == expected['dof'] == 1
        assert actual['scope'] == expected['scope']
        for key in ['B','se','chi2']:
            assert np.isclose(actual[key],expected[key],rtol=2e-5,atol=1e-9),(method,key)
            maximum = max(maximum,abs(actual[key]-expected[key]))
    output = dict(status='PASS',methods=list(slopes),slopes=slopes,
                  maximum_absolute_discrepancy=maximum,
                  scope='Secondary common-model fits using benchmark joint-jackknife errors; distinct from the adopted primary error envelopes.',
                  new_simulations=False,new_random_draws=0,tex_rendering=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='slopes'},indent=2))


if __name__ == '__main__':
    main()
