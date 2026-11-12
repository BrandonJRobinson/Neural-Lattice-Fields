"""Shared reproducibility and statistics helpers for the precision upgrade."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results_precision_upgrade"


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=False, allow_nan=True)
        handle.write("\n")
    os.replace(tmp, path)


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    rows = [jsonable(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(tmp, path)


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_metadata(source_files: Iterable[Path] = (), command: list[str] | None = None) -> dict:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for module in ("scipy", "matplotlib"):
        try:
            mod = __import__(module)
            versions[module] = mod.__version__
        except Exception:
            versions[module] = None
    sources = []
    for path in source_files:
        p = Path(path).resolve()
        sources.append({
            "path": str(p.relative_to(ROOT)) if ROOT in p.parents else str(p),
            "sha256": sha256(p),
        })
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timezone_requested": "Europe/Rome",
        "cwd": str(Path.cwd()),
        "command": list(sys.argv if command is None else command),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "versions": versions,
        "source_files": sources,
        "git_commit": None,
        "git_note": "Repository directory is not a Git worktree in this execution environment.",
    }


def mean_se(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "se": float("nan"), "n": 0}
    se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
    return {"mean": float(arr.mean()), "se": se, "n": int(arr.size)}


def conservative_seed_combine(points: list[dict], key: str, err_key: str | None = None) -> dict:
    vals = np.asarray([p[key] for p in points], dtype=float)
    good = np.isfinite(vals)
    vals = vals[good]
    if vals.size == 0:
        return {"mean": float("nan"), "se_between": float("nan"),
                "se_within": float("nan"), "se": float("nan"), "n_seeds": 0}
    between = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else float("nan")
    within = float("nan")
    if err_key:
        errs = np.asarray([p.get(err_key, float("nan")) for p in points], dtype=float)[good]
        errs = errs[np.isfinite(errs)]
        if errs.size:
            within = float(np.sqrt(np.sum(errs ** 2)) / vals.size)
    finite = [x for x in (between, within) if np.isfinite(x)]
    return {
        "mean": float(vals.mean()), "se_between": between, "se_within": within,
        "se": float(max(finite)) if finite else float("nan"), "n_seeds": int(vals.size),
    }


def fixed_block_bootstrap(
    arrays: dict[str, np.ndarray],
    estimator: Callable[[dict[str, np.ndarray]], float],
    n_boot: int,
    rng: np.random.Generator,
    n_blocks: int = 20,
) -> dict:
    """Bootstrap contiguous blocks; report invalid nonlinear replicates."""
    n = len(next(iter(arrays.values())))
    if n < 4:
        val = float(estimator(arrays))
        return {"value": val, "se": float("nan"), "ci16": float("nan"),
                "ci84": float("nan"), "n_boot": 0, "n_invalid": n_boot,
                "n_blocks": 0, "block_size": 0}
    n_blocks = max(2, min(int(n_blocks), n // 2))
    blocks = [idx for idx in np.array_split(np.arange(n), n_blocks) if idx.size]
    value = float(estimator(arrays))
    boots = []
    for _ in range(int(n_boot)):
        draw = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[i] for i in draw])[:n]
        val = float(estimator({key: arr[idx] for key, arr in arrays.items()}))
        if np.isfinite(val):
            boots.append(val)
    b = np.asarray(boots, dtype=float)
    return {
        "value": value,
        "se": float(b.std(ddof=1)) if b.size > 1 else float("nan"),
        "ci16": float(np.percentile(b, 16)) if b.size else float("nan"),
        "ci84": float(np.percentile(b, 84)) if b.size else float("nan"),
        "n_boot": int(b.size), "n_invalid": int(n_boot - b.size),
        "n_blocks": int(len(blocks)), "block_size": int(np.median([len(x) for x in blocks])),
    }


def timed_call(fn: Callable, *args, **kwargs):
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    return value, time.perf_counter() - start

