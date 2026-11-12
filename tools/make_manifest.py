"""Build the source/record checksum manifest for a prepared release tree."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(p in rel.parts for p in ['.git','.venv','work','__pycache__']):continue
        if path.name in ['chain.bin','.DS_Store'] or str(rel) in ['manifests/files.json','SHA256SUMS.txt']:continue
        raw = path.read_bytes()
        rows.append(dict(path=str(rel), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    output = dict(schema='nnlft-release-files-v1', version='v0.1.0', files=rows,
                  excludes=['manifests/files.json','SHA256SUMS.txt','data/rg1/**/chain.bin','work/','.git/','.venv/','__pycache__/'])
    (ROOT/'manifests/files.json').write_text(json.dumps(output, indent=2)+'\n')
    (ROOT/'SHA256SUMS.txt').write_text(''.join(f'{r["sha256"]}  {r["path"]}\n' for r in rows))
    print(json.dumps(dict(files=len(rows), bytes=sum(r['bytes'] for r in rows)), indent=2))


if __name__ == '__main__':
    main()
