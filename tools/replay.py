"""Read-only raw/summary replay of the release; no sampler or random draws."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from verify_release import ROOT, verify_raw, verify_source


def read(path):
    return json.loads(path.read_text())


def run(script, *args, cwd=None, env=None):
    settings = dict(os.environ)
    settings.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                    MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                    NNLFT_RG1_ROOT=str(ROOT/'data/rg1'), NNLFT_LEGACY_ROOT=str(ROOT/'data/legacy'))
    if env:settings.update(env)
    subprocess.run([sys.executable, '-B', str(script), *map(str,args)], cwd=cwd,
                   env=settings, check=True)


def make_work(path):
    path = path.resolve()
    for protected in ['data','expected','replay','tools','provenance','manifests']:
        assert not path.is_relative_to(ROOT/protected), f'Output is within preserved {protected}'
    path.mkdir(parents=True, exist_ok=False)
    return path


def summaries(output):
    output = make_work(output)
    estimator = output/'estimators'
    shutil.copytree(ROOT/'replay/estimators', estimator)
    run(estimator/'replay_block_statistics.py')
    p3 = output/'reference_kernel'
    shutil.copytree(ROOT/'replay/reference_kernel', p3)
    (p3/'evidence').mkdir()
    run(p3/'analysis/analyze_summaries.py')
    spatial = output/'spatial'
    shutil.copytree(ROOT/'replay/spatial', spatial)
    (spatial/'evidence').mkdir()
    run(spatial/'analysis/verify_analytic_extensions.py')
    run(spatial/'analysis/adversarial_checks.py')
    legacy = output/'legacy'
    legacy.mkdir()
    for n in ['saved_records.py','calibration_and_transitions.py']:
        shutil.copyfile(ROOT/'replay/legacy'/n, legacy/n)
    run(legacy/'saved_records.py', '--data-root', ROOT/'data/legacy', '--output', legacy/'summaries')
    run(legacy/'calibration_and_transitions.py')
    report = dict(status='PASS', kind='summary-and-analytic-replay',
                  estimator=read(estimator/'BLOCK_REPLAY_VERIFICATION.json')['status'],
                  reference_kernel=read(p3/'evidence/SUMMARY_INFERENCE.json'),
                  analytic_checks=read(spatial/'evidence/ANALYTIC_CHECKS.json')['result'],
                  independent_analytic_checks=read(spatial/'evidence/ADVERSARIAL_CHECKS.json')['result'],
                  legacy_summary=read(legacy/'summaries/LEGACY_SUMMARY_REPLAY.json')['status'],
                  calibration_and_transitions=read(legacy/'VERIFIED_LEGACY_RESULTS.json'),
                  new_simulations=False, new_random_draws=0, tex_rendering=False)
    (output/'SUMMARY_REPLAY_VALIDATION.json').write_text(json.dumps(report, indent=2)+'\n')
    return dict(status='PASS', path=str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output))


def rg1(output, quadrature):
    integrity = verify_raw()
    output = make_work(output)
    for n in ['replay_primary.py','quadrature_convergence.py']:
        shutil.copyfile(ROOT/'replay/rg1'/n, output/n)
    # Byte-identical scientific replay expects this historical relative layout.
    holder = output/'extracted/RG1_RAW_REPLAY_20261111'
    holder.mkdir(parents=True)
    (holder/'rg1').symlink_to(ROOT/'data/rg1', target_is_directory=True)
    (output/'RAW_INTEGRITY_VERIFICATION.json').write_text(json.dumps(integrity, indent=2)+'\n')
    run(output/'replay_primary.py')
    actual = read(output/'primary_replay/REPLAY_RESULT.json')
    assert actual['status'] == 'PASS'
    expected = read(ROOT/'expected/rg1/primary_replay/REPLAY_RESULT.json')
    import numpy as np
    for L in ['12','16','24','32']:
        assert np.allclose(actual['z3'][L]['g_at_z3'], expected['z3'][L]['g_at_z3'], rtol=1e-9, atol=1e-11)
    if quadrature:
        run(output/'quadrature_convergence.py')
        assert read(output/'quadrature_convergence/CONVERGENCE_RESULT.json')['status'] == 'PASS'
    result = dict(status='PASS', kind='raw-to-primary', raw_files=228,
                  principal_chains=200, primary_rows=123988000, controls=28,
                  quadrature_replayed=bool(quadrature), new_simulations=False,
                  synthetic_calibration='archived summary retained; no new draws',
                  historical_diagnostic_grid='carried records; full grid not rerun', tex_rendering=False)
    (output/'RAW_TO_RESULT_VALIDATION.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def estimator_raw(output):
    verify_raw()
    output = make_work(output)
    source = output/'source'
    shutil.copytree(ROOT/'replay/estimators', source)
    generated = output/'recomputed'
    run(source/'raw_reprocess.py', '--rg1-root', ROOT/'data/rg1', '--output', generated)
    for name in ['benchmark_cv.py','replay_block_statistics.py','CV_BENCHMARK_RESULT.json']:
        shutil.copyfile(source/name, generated/name)
    shutil.copytree(source/'pinned_source', generated/'pinned_source')
    run(generated/'replay_block_statistics.py')
    assert read(generated/'RAW_REPLAY_VALIDATION.json')['status'] == 'PASS'
    assert read(generated/'BLOCK_REPLAY_VERIFICATION.json')['status'] == 'PASS'
    return dict(status='PASS', kind='raw-to-control-variate-results', raw_files=200,
                rows=123988000, new_simulations=False, new_random_draws=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['summaries','rg1','estimator-raw','all'])
    parser.add_argument('--output', type=Path, required=True, help='New output directory; no previous outputs are reused')
    parser.add_argument('--quadrature', action='store_true', help='Also replay the 160-node full-data convergence check')
    args = parser.parse_args()
    if sys.byteorder != 'little':
        raise RuntimeError('The preserved raw reader is validated for little-endian IEEE754 hosts.')
    count = verify_source()
    start = time.monotonic()
    if args.mode == 'all':
        output = make_work(args.output)
        reports = dict(summaries=summaries(output/'summaries'),
                       rg1=rg1(output/'rg1',args.quadrature),
                       estimator_raw=estimator_raw(output/'estimator_raw'))
    else:
        function = {'summaries':summaries,'rg1':lambda p:rg1(p,args.quadrature),'estimator-raw':estimator_raw}[args.mode]
        reports = {args.mode:function(args.output)}
        output = args.output.resolve()
    result = dict(status='PASS', mode=args.mode, source_files_verified=count,
                  elapsed_seconds=time.monotonic()-start, reports=reports,
                  new_simulations=False, new_random_draws=0, tex_rendering=False)
    (output/'RELEASE_REPLAY_REPORT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
