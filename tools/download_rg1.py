"""Download and checksum existing public RG-1 measurements; never run a sampler."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import urllib.request

from verify_release import ROOT, sha


def download(url, path):
    # System curl uses the operating system's trusted certificate store on macOS.
    # Python.org builds may lack configured CA roots; never disable verification.
    curl = shutil.which('curl')
    if curl:
        subprocess.run([curl, '--fail', '--location', '--silent', '--show-error',
                        '--proto', '=https', '--proto-redir', '=https',
                        '--retry', '3', '--connect-timeout', '30',
                        '--output', str(path), url], check=True)
    else:
        request = urllib.request.Request(url, headers={'User-Agent':'NNLFT-replay/0.1.0'})
        with urllib.request.urlopen(request, timeout=120) as response, path.open('wb') as output:
            for block in iter(lambda:response.read(8*1024*1024), b''):output.write(block)


def unpack(path, asset, all_rows, extra_rows, verify_only):
    expected = {r['path']:r for r in all_rows if r['asset'] == asset['name']}
    extras = {r['name']:r for r in extra_rows}
    seen = set()
    with tarfile.open(path, 'r|gz') as tar:
        for member in tar:
            name = member.name
            pure = PurePosixPath(name)
            assert member.isfile() and not pure.is_absolute() and '..' not in pure.parts, name
            assert name in expected or name in extras, f'Unexpected archive member: {name}'
            assert name not in seen, f'Duplicate archive member: {name}'
            seen.add(name)
            row = expected.get(name, extras.get(name))
            assert member.size == row['bytes']
            target = ROOT/name
            write = name in expected and not verify_only
            if write and target.exists():
                assert target.is_file() and sha(target) == row['sha256'], f'Refusing to replace differing {name}'
                write = False
            temp = target.with_name(target.name+'.download-partial')
            output = None
            if write:
                assert ROOT in target.resolve().parents
                target.parent.mkdir(parents=True, exist_ok=True)
                output = temp.open('xb')
            digest = hashlib.sha256()
            count = 0
            try:
                with tar.extractfile(member) as source:
                    for block in iter(lambda:source.read(8*1024*1024), b''):
                        digest.update(block); count += len(block)
                        if output:output.write(block)
                if output:output.close()
                assert count == row['bytes'] and digest.hexdigest() == row['sha256'], name
                if write:temp.replace(target)
            except BaseException:
                if output:output.close()
                if write and temp.exists():temp.unlink()
                raise
    assert seen == set(expected)|set(extras)
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir', type=Path, help='Use already downloaded matching assets; no network access')
    parser.add_argument('--cache', type=Path, default=ROOT/'work/downloads')
    parser.add_argument('--verify-only', action='store_true', help='Verify archive and all member hashes without extracting')
    args = parser.parse_args()
    bundle = json.loads((ROOT/'manifests/assets.json').read_text())
    raw = json.loads((ROOT/'manifests/raw-files.json').read_text())
    args.cache.mkdir(parents=True, exist_ok=True)
    count = 0
    for asset in bundle['assets']:
        path = (args.asset_dir or args.cache)/asset['name']
        if not path.exists():
            if args.asset_dir:raise FileNotFoundError(path)
            prefix = 'https://github.com/BrandonJRobinson/Neural-Lattice-Fields/releases/download/v0.1.0/'
            assert asset['url'] == prefix+asset['name']
            part = path.with_suffix(path.suffix+'.partial')
            print(f'Downloading {asset["name"]} ({asset["bytes"]} bytes)', flush=True)
            download(asset['url'], part)
            assert part.stat().st_size == asset['bytes'] and sha(part) == asset['sha256']
            part.replace(path)
        assert path.stat().st_size == asset['bytes'] and sha(path) == asset['sha256']
        print(f'Verifying archive and members: {asset["name"]}', flush=True)
        count += unpack(path, asset, raw['files'], bundle['extra_members'], args.verify_only)
    assert count == 228
    print(json.dumps(dict(status='PASS', verified_measurement_members=count,
                         extraction=not args.verify_only, new_simulations=False), indent=2))


if __name__ == '__main__':
    main()
