"""Package existing RG-1 measurement bytes; never run a sampler."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


class HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        self.count += len(block)
        return block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'manifests/raw-files.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    assets = []
    notice = ('RG-1 existing measurement records\n'
              'Copyright (c) 2026 Vishnu Jejjala and Brandon Robinson.\n'
              'Data license: Creative Commons Attribution 4.0 International.\n'
              'https://creativecommons.org/licenses/by/4.0/\n'
              'Release: https://github.com/BrandonJRobinson/Neural-Lattice-Fields/releases/tag/v0.1.0\n'
              'See the repository manifests for per-file SHA256 values and the data contract.\n').encode()
    extras = [('README_DATA.txt', notice), ('CC-BY-4.0.txt', (root/'LICENSES/CC-BY-4.0.txt').read_bytes())]
    for name in sorted({r['asset'] for r in manifest['files']}):
        rows = sorted((r for r in manifest['files'] if r['asset'] == name), key=lambda r:r['path'])
        path = args.output/name
        temp = path.with_suffix(path.suffix+'.partial')
        assert not path.exists(), f'Refusing to overwrite {path}'
        print(f'Packaging {name}: {len(rows)} files, {sum(r["bytes"] for r in rows)} measurement bytes', flush=True)
        with temp.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=1, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w|', format=tarfile.USTAR_FORMAT) as tar:
                for filename, data in extras:
                    info = tarfile.TarInfo(filename)
                    info.size = len(data); info.mode = 0o644; info.mtime = 0
                    tar.addfile(info, io.BytesIO(data))
                for row in rows:
                    info = tarfile.TarInfo(row['path'])
                    info.size = row['bytes']; info.mode = 0o644; info.mtime = 0
                    with (root/row['path']).open('rb') as stream:
                        checked = HashedReader(stream)
                        tar.addfile(info, checked)
                        assert checked.count == row['bytes']
                        assert checked.digest.hexdigest() == row['sha256'], row['path']
        temp.replace(path)
        assert path.stat().st_size < 2*1024**3
        item = dict(name=name, bytes=path.stat().st_size, sha256=sha(path),
                    url='https://github.com/BrandonJRobinson/Neural-Lattice-Fields/releases/download/v0.1.0/'+name,
                    measurement_files=len(rows), measurement_bytes=sum(r['bytes'] for r in rows))
        assets.append(item)
        print(json.dumps(item), flush=True)
    report = dict(schema='nnlft-release-assets-v1', version=manifest['version'],
                  primary_files=200, control_files=28, measurement_files=228,
                  measurement_bytes=manifest['raw_bytes'],
                  extra_members=[dict(name=n, bytes=len(b), sha256=hashlib.sha256(b).hexdigest()) for n,b in extras],
                  assets=assets, new_simulations=False)
    (root/'manifests/assets.json').write_text(json.dumps(report, indent=2)+'\n')
    sums = ''.join(f'{r["sha256"]}  {r["name"]}\n' for r in assets)
    (args.output/'SHA256SUMS_ASSETS.txt').write_text(sums)
    (root/'manifests/SHA256SUMS_ASSETS.txt').write_text(sums)
    print('All 228 input hashes verified during packaging.', flush=True)


if __name__ == '__main__':
    main()
