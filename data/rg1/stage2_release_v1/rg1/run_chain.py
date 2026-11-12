"""Run ONE planned chain with the pinned exact sampler; capture to persistent storage BEFORE any QC.

Sequence (each step's failure is recorded; nothing is ever deleted from persistent storage):
  1. verify the pinned binary hash; if this chain already has a verified capture, exit 0 (idempotent restart);
  2. run the unmodified binary into node-local scratch, with a wall-clock guard below the SLURM limit;
  3. structural completeness: exit status 0, exactly one stdout JSON whose parameters echo the plan,
     output size == 32 * n_meas bytes (the binary does not check fwrite/fclose itself);
  4. durable capture: copy (fsync) into `<data>/<ensemble>/.chain_<c>.partial.<nonce>/`, verify SHA-256 of
     source == copy, write MANIFEST.sha256 + PROVENANCE.json, atomic rename to `chain_<c>/`;
  5. only then QC (rg1.qc.chain_qc) on the captured copy, written beside it as `chain_<c>.qc.json`.
A chain that fails 1-4 leaves no `chain_<c>/` and is never analysed; its diagnostics go to `<data>/failures/`.
Exit: 0 captured (QC verdict recorded separately), 3 refused, 1 failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fsync_path(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verify_capture(d: Path) -> list[str]:
    man = d / "MANIFEST.sha256"
    if not man.is_file():
        return ["MANIFEST.sha256 missing"]
    bad = []
    for line in man.read_text().splitlines():
        h, name = line.split("  ", 1)
        p = d / name
        if not p.is_file():
            bad.append(f"missing {name}")
        elif sha256(p) != h:
            bad.append(f"digest mismatch {name}")
    return bad


def capture(files: dict[str, Path], dest: Path, provenance: dict) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.parent / f".{dest.name}.partial.{os.getpid()}.{time.time_ns()}"
    partial.mkdir()
    lines = []
    for name, src in files.items():
        before = sha256(src)
        shutil.copyfile(src, partial / name)
        fsync_path(partial / name)
        after, copied = sha256(src), sha256(partial / name)
        if not (before == after == copied):
            raise RuntimeError(f"hash mismatch capturing {name}; partial kept at {partial}")
        lines.append(f"{copied}  {name}\n")
    (partial / "MANIFEST.sha256").write_text("".join(lines))
    (partial / "PROVENANCE.json").write_text(json.dumps(provenance, indent=1, sort_keys=True) + "\n")
    for n in ("MANIFEST.sha256", "PROVENANCE.json"):
        fsync_path(partial / n)
    fsync_path(partial)
    bad = verify_capture(partial)
    if bad:
        raise RuntimeError(f"post-copy verification failed {bad}; partial kept at {partial}")
    if dest.exists():
        raise RuntimeError(f"{dest} appeared concurrently; partial kept at {partial}")
    os.rename(partial, dest)
    fsync_path(dest.parent)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True); ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--binary", required=True); ap.add_argument("--data-root", required=True)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--guard-seconds", type=float, default=600.0, help="stop this long before the time limit")
    a = ap.parse_args(argv)
    plan = json.loads(Path(a.plan).read_text())
    t = plan["tasks"][a.task]
    assert t["task"] == a.task
    data = Path(a.data_root); dest = data / t["ensemble"] / f"chain_{t['chain']:02d}"
    fail_dir = data / "failures"; fail_dir.mkdir(parents=True, exist_ok=True)
    job = f"{os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID', 'local'))}_{os.environ.get('SLURM_ARRAY_TASK_ID', a.task)}"
    rec = {"task": a.task, "ensemble": t["ensemble"], "chain": t["chain"], "seed": t["seed"], "job": job,
           "host": socket.gethostname(), "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def fail(reason, rc=1, extra=None):
        rec.update(status="FAILED" if rc == 1 else "REFUSED", reason=reason, **(extra or {}))
        rec["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (fail_dir / f"task{a.task:03d}_{job}.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps(rec), file=sys.stderr)
        return rc

    if sha256(Path(a.binary)) != plan["sampler"]["binary_sha256"]:
        return fail("binary hash differs from plan", 3)
    if dest.is_dir():
        bad = verify_capture(dest)
        if not bad:
            print(f"already captured and verified: {dest}")
            return 0
        return fail(f"existing capture fails verification {bad}; not overwritten", 3)

    work = Path(a.scratch) / f"rg1_{job}_t{a.task}"
    work.mkdir(parents=True, exist_ok=True)
    out_bin, out_json, out_err = work / "chain.bin", work / "chain.json", work / "chain.stderr"
    budget = t["time_limit_hours"] * 3600 - a.guard_seconds
    cmd = [a.binary, *map(str, t["args"]), str(out_bin), str(t["zero_mode_gibbs"])]
    rec["cmd"] = cmd
    t0 = time.time()
    try:
        with open(out_json, "w") as so, open(out_err, "w") as se:
            p = subprocess.run(cmd, stdout=so, stderr=se, stdin=subprocess.DEVNULL, timeout=budget)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        return fail(f"wall-clock guard {budget:.0f}s reached; chain incomplete", 1, {"elapsed_s": time.time() - t0})
    rec["elapsed_s"] = time.time() - t0; rec["returncode"] = rc
    if rc != 0:
        return fail(f"sampler exit status {rc}", 1, {"stderr": out_err.read_text()[-2000:]})
    try:
        lines = [ln for ln in out_json.read_text().splitlines() if ln.strip()]
        meta = json.loads(lines[-1]); assert len(lines) == 1
    except Exception as exc:  # noqa: BLE001
        return fail(f"stdout JSON unreadable: {exc}")
    L, m0, lam, h, nth, nme, nor, mev, seed = t["args"]
    echo = {"L": L, "n_therm": nth, "n_meas": nme, "n_or": nor, "meas_every": mev, "seed": seed,
            "zero_mode_gibbs": t["zero_mode_gibbs"]}
    mism = {k: (meta.get(k), v) for k, v in echo.items() if meta.get(k) != v}
    for k, v in (("m0sq", m0), ("lambda", lam), ("h", h)):
        if abs(float(meta.get(k, "nan")) - float(v)) > 1e-12 * max(1.0, abs(float(v))):
            mism[k] = (meta.get(k), v)
    if mism:
        return fail(f"stdout parameters differ from plan {mism}")
    size = out_bin.stat().st_size if out_bin.is_file() else -1
    if size != t["expected_bytes"]:
        return fail(f"output size {size} != expected {t['expected_bytes']}")
    rec["sampler_meta"] = meta
    try:
        d = capture({"chain.bin": out_bin, "chain.json": out_json, "chain.stderr": out_err}, dest,
                    {**rec, "plan_sha256": sha256(Path(a.plan)), "binary_sha256": plan["sampler"]["binary_sha256"],
                     "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    except Exception as exc:  # noqa: BLE001
        return fail(f"capture failed: {exc}")
    print(f"captured {d}")
    # QC strictly after capture, on the persistent copy; a QC exception never touches the capture.
    try:
        from rg1.qc import chain_qc
        q = chain_qc(d / "chain.bin", t, meta)
    except Exception as exc:  # noqa: BLE001
        q = {"status": "QC_ERROR", "error": f"{type(exc).__name__}: {exc}"}
    q["capture_integrity"] = verify_capture(d) or "OK"
    (d.parent / f"{d.name}.qc.json").write_text(json.dumps(q, indent=1, default=float) + "\n")
    shutil.rmtree(work, ignore_errors=True)       # node-local scratch only
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
