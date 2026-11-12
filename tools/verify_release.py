"""Verify public source/record manifests and, optionally, all measurement bytes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_rows(rows):
    for i, row in enumerate(rows, 1):
        path = ROOT/row['path']
        if not path.is_file():
            raise FileNotFoundError(f'Missing {row["path"]}; obtain the matching release assets first.')
        assert path.stat().st_size == row['bytes'], row['path']
        assert sha(path) == row['sha256'], row['path']
        if row['path'].endswith('/chain.bin') and i % 20 == 0:
            print(f'Verified {i}/{len(rows)} measurement files', flush=True)


def verify_source():
    manifest = json.loads((ROOT/'manifests/files.json').read_text())
    verify_rows(manifest['files'])
    return len(manifest['files'])


def verify_raw():
    manifest = json.loads((ROOT/'manifests/raw-files.json').read_text())
    rows = manifest['files']
    assert len(rows) == 228 and sum(r['role']=='primary' for r in rows) == 200
    assert sum(r['bytes'] for r in rows) == 4069376000
    verify_rows(rows)
    for row in rows:
        meta = json.loads((ROOT/row['path']).with_name('chain.json').read_text())
        assert meta['n_meas'] == row['rows'] and row['bytes'] == 32*row['rows']
    return dict(status='PASS', files=228, primary_files=200, control_files=28,
                bytes=sum(r['bytes'] for r in rows), rows=sum(r['rows'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = dict(status='PASS', source_and_record_files=verify_source(),
                  new_simulations=False, new_random_draws=0, tex_rendering=False)
    if args.raw:report['raw'] = verify_raw()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
