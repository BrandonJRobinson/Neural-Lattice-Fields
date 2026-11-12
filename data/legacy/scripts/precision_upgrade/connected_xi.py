"""Raw and connected/sector-aware second-moment correlation lengths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from common import fixed_block_bootstrap  # noqa: E402


def xi_from_structure_factors(g0: float, gp: float, L: int) -> float:
    if not (np.isfinite(g0) and np.isfinite(gp)) or gp <= 0:
        return float("nan")
    xi2 = (g0 / gp - 1.0) / (4.0 * np.sin(np.pi / L) ** 2)
    return float(np.sqrt(xi2)) if xi2 >= 0 else float("nan")


def estimator_values(arrays: dict[str, np.ndarray], L: int) -> dict:
    """Return raw, ordinary-connected, and ordered-sector xi/L estimates.

    A global sector alignment phi -> sign(M) phi leaves the nonzero-momentum
    structure factor unchanged. Its connected zero mode is therefore
    V*(<M^2>-<|M|>^2), denoted ``sector`` below.
    """
    V = L ** int(arrays.get("dim", np.asarray([2]))[0]) if "dim" in arrays else None
    if V is None:
        # ghat0 = V*M^2 fixes V without requiring the dimension.
        m2 = float(np.mean(arrays["M2"]))
        gh0 = float(np.mean(arrays["ghat0"]))
        V = int(round(gh0 / m2)) if m2 > 0 else L * L
    mean_m = float(np.mean(arrays["M"]))
    mean_abs = float(np.mean(arrays["absM"]))
    mean_m2 = float(np.mean(arrays["M2"]))
    gp = float(np.mean(arrays["ghatpmin"]))
    g0_raw = float(np.mean(arrays["ghat0"]))
    g0_conn = float(V * (mean_m2 - mean_m ** 2))
    g0_sector = float(V * (mean_m2 - mean_abs ** 2))
    raw = xi_from_structure_factors(g0_raw, gp, L) / L
    conn = xi_from_structure_factors(g0_conn, gp, L) / L
    sector = xi_from_structure_factors(g0_sector, gp, L) / L
    m4 = float(np.mean(arrays["M4"]))
    u4 = 1.0 - m4 / (3.0 * mean_m2 ** 2) if mean_m2 > 0 else float("nan")
    # Sector subtraction is appropriate once the magnetization distribution is
    # visibly two-lobed; ordinary connected subtraction is retained otherwise.
    selected_name = "sector_abs" if np.isfinite(u4) and u4 > 0.30 else "connected_M"
    selected = sector if selected_name == "sector_abs" else conn
    return {
        "xi_raw_over_L": raw,
        "xi_connected_M_over_L": conn,
        "xi_sector_abs_over_L": sector,
        "xi_connected_over_L": selected,
        "connected_policy": selected_name,
        "g0_raw": g0_raw, "g0_connected_M": g0_conn,
        "g0_sector_abs": g0_sector, "ghatpmin": gp, "U4": u4,
        "chi0": g0_raw, "chi_abs": g0_sector,
    }


def bootstrap_estimators(arrays: dict[str, np.ndarray], L: int, seed: int,
                         n_boot: int = 500, n_blocks: int = 20) -> dict:
    base = estimator_values(arrays, L)
    rng = np.random.default_rng(seed)
    result = dict(base)
    keys = ("xi_raw_over_L", "xi_connected_M_over_L",
            "xi_sector_abs_over_L", "xi_connected_over_L", "U4", "chi0", "chi_abs")
    n = len(next(iter(arrays.values())))
    blocks = [idx for idx in np.array_split(np.arange(n), max(2, min(n_blocks, n // 2))) if idx.size]
    draws = {key: [] for key in keys}
    for _ in range(n_boot):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[i] for i in selected])[:n]
        values = estimator_values({name: values[idx] for name, values in arrays.items()}, L)
        for key in keys:
            if np.isfinite(values[key]):
                draws[key].append(values[key])
    for key in keys:
        b = np.asarray(draws[key], dtype=float)
        result[key + "_bootstrap"] = {
            "value": base[key], "se": float(b.std(ddof=1)) if b.size > 1 else float("nan"),
            "ci16": float(np.percentile(b, 16)) if b.size else float("nan"),
            "ci84": float(np.percentile(b, 84)) if b.size else float("nan"),
            "n_boot": int(b.size), "n_invalid": int(n_boot - b.size),
            "n_blocks": len(blocks), "block_size": int(np.median([len(x) for x in blocks])),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, help="Scalar chain observable archive")
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()
    data = np.load(args.npz)
    arrays = {key: data[key] for key in data.files}
    import json
    print(json.dumps(bootstrap_estimators(arrays, args.L, args.seed, args.n_boot), indent=2))


if __name__ == "__main__":
    main()
