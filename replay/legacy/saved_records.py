"""Deterministic recombination of saved legacy run summaries, not chain histories."""
import argparse
import ast
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm


def load(path):
    return json.loads(path.read_text())


def original_function(root, script, name):
    path = root/'scripts'/script
    parsed = ast.parse(path.read_text())
    node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {'np':np, 'math':math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.data_root
    results = {}
    cases = [('p4_anharmonic_oscillator_v1', 'analyze_p4_anharmonic.py', 'Nt'),
             ('p5_fermionic_full_measure_v1', 'analyze_p5_fermionic.py', 'g')]
    for campaign, script, axis in cases:
        verdict = load(root/'results'/campaign/'analysis/verdict.json')
        records = [load(p) for p in sorted((root/'results'/campaign/'raw').glob('*.json'))]
        aggregate = original_function(root, script, 'aggregate')
        checked = []
        for row in verdict['physics_rows']:
            group = [r for r in records if r[axis] == row[axis]]
            mean, se = aggregate(group, row['observable'])
            assert np.allclose([mean,se], [row['hmc'],row['se']], rtol=1e-11, atol=1e-15)
            checked.append(dict(row))
        tests = []
        for observable in sorted({r['observable'] for r in checked}):
            used = [r for r in checked if r['observable'] == observable]
            excluded = []
            if campaign.startswith('p5') and observable == 'condensate':
                excluded = [r for r in used if r['g'] == 0]
                assert len(excluded) == 1 and abs(excluded[0]['hmc']-68/45) < 1e-13
                used = [r for r in used if r['g'] != 0]
            statistic = sum(((r['hmc']-r['reference'])/r['se'])**2 for r in used)
            tests.append(dict(observable=observable, chi2=statistic, dof=len(used),
                              nominal_p=float(chi2.sf(statistic,len(used))), deterministic_rows_excluded=len(excluded)))
        results[campaign] = dict(run_summaries=len(records), physics_rows=checked,
                                 tests=tests, archived_sampler_pass=verdict['sampler_pass'])
    combine = original_function(root, 'analyze_moment_engineered_cumulants.py', 'combine')
    records = [load(p) for p in sorted((root/'results/moment_engineering_v1/raw').glob('*.json'))]
    rows = list(csv.DictReader((root/'results/moment_engineering_v1/analysis/summary.csv').open()))
    for row in rows:
        group = [r for r in records if r['kind']==row['kind'] and r['width']==int(row['width'])]
        for key in ['variance','N_kappa4','N2_kappa6']:
            mean, _, _, error, _ = combine([r[key] for r in group], [r[key+'_se'] for r in group])
            assert np.allclose([mean,error], [float(row[key]),float(row[key+'_err'])], rtol=1e-10)
    results['coincident_cumulants'] = dict(run_summaries=len(records), width_law_rows=len(rows))

    p3 = load(root/'results/p3_reference_kernel_invariance_rescue_v1/analysis/verdict.json')
    trends = []
    for row in p3['physics_rows']:
        masses = np.array(sorted(float(x) for x in row['values']))
        cells = [row['values'][str(float(m))] for m in masses]
        y = np.array([r['mean'] for r in cells]); se = np.array([r['se'] for r in cells])
        weights = se**-2
        center = weights@y/weights.sum()
        residual_variance = se**2-1/weights.sum()
        maximum = float(np.max(abs(y-center)/np.sqrt(residual_variance)))
        x = np.log(masses); dx = x-(weights@x)/weights.sum()
        slope = float(weights@(dx*y)/(weights@(dx*dx)))
        slope_se = float(1/np.sqrt(weights@(dx*dx)))
        trends.append(dict(L=row['L'], observable=row['observable'], slope=slope, slope_se=slope_se,
                           nominal_p=float(2*norm.sf(abs(slope/slope_se))), maximum_corrected_residual=maximum))
    ps = np.array([r['nominal_p'] for r in trends]); order = np.argsort(ps)
    adjusted = np.empty(len(ps)); adjusted[order] = np.minimum(1,np.maximum.accumulate((len(ps)-np.arange(len(ps)))*ps[order]))
    for row,p in zip(trends,adjusted):row['nominal_holm18_p'] = float(p)
    results['reference_kernel_original_diagnostics'] = dict(trends=trends,
        maximum_corrected_residual=max(r['maximum_corrected_residual'] for r in trends),
        minimum_nominal_holm18_p=float(min(adjusted)))
    output = dict(status='PASS', results=results,
                  scope='Saved-run-summary recombination and nominal diagnostics; no time-history reconstruction or resampling',
                  new_simulations=False, new_random_draws=0)
    (args.output/'LEGACY_SUMMARY_REPLAY.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(dict(status='PASS', campaigns=list(results), new_simulations=False), indent=2))


if __name__ == '__main__':
    main()
