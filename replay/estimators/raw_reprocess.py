"""Read-only reprocessing of the existing binaries into a new output directory."""
import argparse, json
from pathlib import Path
import benchmark_cv as b
import numpy as np

BUNDLE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rg1-root',type=Path,required=True,help='Extracted rg1 directory containing stage1_data/stage2_data')
    parser.add_argument('--output',type=Path,required=True,help='New output directory; must not already exist')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'chains').mkdir()
    b.OUT=args.output.resolve();b.RG=args.rg1_root.resolve()
    refs=b.read(BUNDLE/'RAW_INPUT_INDEX.json')
    assert len(refs)==200 and sum(r['N'] for r in refs)==123988000
    for i,reference in enumerate(refs):
        actual=b.make_record(reference,i)
        expected=b.read(BUNDLE/'chains'/f"{actual['key']}.json")
        for j,block in enumerate(actual['sd']):
            for key in ['sum_y','sum_c','sum_cc','sum_cy']:
                assert np.allclose(block[key],expected['sd'][j][key],rtol=2e-9,atol=1e-8),(actual['key'],j,key)
        print(f"{i+1}/200 verified and reprocessed {actual['key']}",flush=True)
    result=dict(status='PASS',raw_binary_hashes=200,rows=123988000,
                scope='raw hashes, primitive block means and full SD24 block sufficient statistics reproduced',
                simulations=False,random_draws=False,tex_execution=False)
    b.save(args.output/'RAW_REPLAY_VALIDATION.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
