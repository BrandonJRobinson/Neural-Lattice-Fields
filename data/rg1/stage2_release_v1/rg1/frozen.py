"""Pinned inputs. The staged package's own analysis code is imported only after its hash is verified."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

RG1_ROOT = Path(os.environ.get("RG1_ROOT", "/home/brobins/RG1_SCIENTIFIC_RESCUE_20261028"))
SOURCE = RG1_ROOT / "source"
ANALYSIS = SOURCE / "analysis"


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_sums() -> dict:
    out = {}
    for line in (SOURCE / "SHA256SUMS.txt").read_text().splitlines():
        if line.strip():
            h, p = line.split(None, 1)
            out[p.strip().lstrip("./")] = h
    return out


PINNED = ("src/phi4_local2.cpp", "analysis/zm.py", "analysis/analyze_ensemble.py", "analysis/pt_compare.py",
          "analysis/running_fits_g.py", "analysis/running_fits.py", "results/scaling_final.json",
          "results/forecast.json", "results/batch_check.json")
BINARY_SHA = "fe5601a3a624572c60bd5c72c81f6cd55a164be64d817507879f6e1b477311c9"


def verify() -> dict:
    sums = package_sums(); res = {}
    for rel in PINNED:
        got = sha256(SOURCE / rel) if (SOURCE / rel).is_file() else None
        res[rel] = {"expected": sums.get(rel), "actual": got, "ok": got is not None and got == sums.get(rel)}
    b = RG1_ROOT / "bin" / "phi4_local2"
    res["bin/phi4_local2"] = {"expected": BINARY_SHA, "actual": sha256(b), "ok": sha256(b) == BINARY_SHA}
    return res


def require():
    bad = [k for k, v in verify().items() if not v["ok"]]
    if bad:
        raise RuntimeError(f"pinned RG-1 inputs differ: {bad}")


def _load_zm():
    require()
    spec = importlib.util.spec_from_file_location("rg1_frozen_zm", ANALYSIS / "zm.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


zm = _load_zm()
