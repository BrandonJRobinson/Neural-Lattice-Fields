"""Failure injection and negative controls for the RG-1 Stage 1 runner, capture, QC and adjudication."""
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REL))
BIN = REL.parent / "bin" / "phi4_local2"

from rg1 import run_chain  # noqa: E402
from rg1.qc import chain_qc  # noqa: E402


def mini_task(tmp_path, *, L=4, n=4000, therm=200, hours=0.25, expected=None, sha=None):
    plan = json.loads((REL / "PLAN.json").read_text())
    t = copy.deepcopy(plan["tasks"][0])
    t["args"] = [L, "-0.3", "0.4", "0.0", therm, n, 2, 1, t["seed"]]
    t.update(n_meas=n, expected_bytes=expected if expected is not None else 32 * n, time_limit_hours=hours)
    plan["tasks"] = [t]
    if sha:
        plan["sampler"]["binary_sha256"] = sha
    p = tmp_path / "plan.json"; p.write_text(json.dumps(plan))
    return p


def run(plan, tmp_path, guard=0.0):
    return run_chain.main(["--plan", str(plan), "--task", "0", "--binary", str(BIN), "--data-root", str(tmp_path / "data"),
                           "--scratch", str(tmp_path / "scr"), "--guard-seconds", str(guard)])


def dest(tmp_path):
    return tmp_path / "data" / "P24_lo" / "chain_00"


def failures(tmp_path):
    return [json.loads(p.read_text()) for p in (tmp_path / "data" / "failures").glob("*.json")]


def test_complete_chain_captured_then_qc(tmp_path):
    assert run(mini_task(tmp_path), tmp_path) == 0
    d = dest(tmp_path)
    assert run_chain.verify_capture(d) == []
    assert (d / "chain.bin").stat().st_size == 32 * 4000
    q = json.loads((d.parent / "chain_00.qc.json").read_text())
    assert q["status"] == "PASS" and q["capture_integrity"] == "OK"


def test_rerun_is_idempotent_and_bit_identical(tmp_path):
    p = mini_task(tmp_path); assert run(p, tmp_path) == 0
    h1 = run_chain.sha256(dest(tmp_path) / "chain.bin")
    assert run(p, tmp_path) == 0                                  # verified capture -> no rerun
    other = tmp_path / "again"; other.mkdir()
    assert run(p, other) == 0                                     # same seed elsewhere -> identical bytes
    assert run_chain.sha256(dest(other) / "chain.bin") == h1


def test_tampered_capture_is_never_overwritten(tmp_path):
    p = mini_task(tmp_path); assert run(p, tmp_path) == 0
    f = dest(tmp_path) / "chain.bin"; f.chmod(0o644)
    with open(f, "r+b") as fh:
        fh.seek(64); fh.write(b"\x00" * 8)
    before = f.read_bytes()
    assert run(p, tmp_path) == 3
    assert f.read_bytes() == before and any("verification" in r["reason"] for r in failures(tmp_path))


def test_wrong_binary_hash_refused(tmp_path):
    assert run(mini_task(tmp_path, sha="0" * 64), tmp_path) == 3
    assert not dest(tmp_path).exists()


def test_size_mismatch_never_captured(tmp_path):
    assert run(mini_task(tmp_path, expected=32 * 4000 + 32), tmp_path) == 1
    assert not dest(tmp_path).exists() and "output size" in failures(tmp_path)[0]["reason"]


def test_wall_clock_guard_kills_incomplete_chain(tmp_path):
    # 0.25 h limit with a 899 s guard -> 1 s budget; L=16 with 10^6 sweeps cannot finish.
    p = mini_task(tmp_path, L=16, n=1_000_000, therm=10)
    assert run(p, tmp_path, guard=899.0) == 1
    assert not dest(tmp_path).exists() and "wall-clock guard" in failures(tmp_path)[0]["reason"]


def test_copy_corruption_detected_partial_kept(tmp_path, monkeypatch):
    real = shutil.copyfile

    def corrupt(src, dst, *a, **k):
        real(src, dst)
        if Path(dst).name == "chain.bin":
            with open(dst, "r+b") as fh:
                fh.seek(0); fh.write(b"\xff")
    monkeypatch.setattr(run_chain.shutil, "copyfile", corrupt)
    assert run(mini_task(tmp_path), tmp_path) == 1
    assert not dest(tmp_path).exists()
    assert list((tmp_path / "data" / "P24_lo").glob(".chain_00.partial.*"))
    assert "capture failed" in failures(tmp_path)[0]["reason"]


def test_qc_exception_cannot_touch_capture(tmp_path, monkeypatch):
    import rg1.qc
    monkeypatch.setattr(rg1.qc, "chain_qc", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert run(mini_task(tmp_path), tmp_path) == 0
    q = json.loads((dest(tmp_path).parent / "chain_00.qc.json").read_text())
    assert q["status"] == "QC_ERROR" and q["capture_integrity"] == "OK"


def test_qc_rejects_nonfinite_and_short_chains(tmp_path):
    t = json.loads((REL / "PLAN.json").read_text())["tasks"][0]
    meta = {"acc_hb": 0.97, "acc_or": 0.98}
    a = np.random.default_rng(0).standard_normal((4000, 4)); a[100, 2] = np.nan
    f = tmp_path / "x.bin"; a.tofile(f)
    assert chain_qc(f, t, meta)["status"] == "FAIL"
    b = np.random.default_rng(1).standard_normal((100, 4)); g = tmp_path / "y.bin"; b.tofile(g)
    q = chain_qc(g, t, meta)
    assert q["status"] == "FAIL" and any("50 * max tau" in r for r in q["reasons"])
    q = chain_qc(f.with_name("z.bin") if False else g, t, {"acc_hb": 0.5, "acc_or": 0.98})
    assert any("acc_hb" in r for r in q["reasons"])


def test_qc_detects_nonstationary_chain(tmp_path):
    t = json.loads((REL / "PLAN.json").read_text())["tasks"][0]
    rng = np.random.default_rng(2); n = 40000
    a = rng.standard_normal((n, 4)); a[:, 1] += np.linspace(0, 3, n)           # drifting S2
    f = tmp_path / "d.bin"; a.tofile(f)
    q = chain_qc(f, t, {"acc_hb": 0.97, "acc_or": 0.98})
    assert any("split-chain S2" in r for r in q["reasons"])


def test_plan_is_within_ceiling_and_contract():
    p = json.loads((REL / "PLAN.json").read_text())
    assert p["allocated_core_hours"] <= p["resource_ceiling_core_hours"] == 600.0
    seeds = [t["seed"] for t in p["tasks"]]
    assert len(seeds) == len(set(seeds))
    for e in p["ensembles"]:
        assert e["n_therm"] >= 5000
        if e["role"] == "production":
            assert e["chains"] >= 8 and e["lambda"] == 0.4 and e["h"] == 0.0
    names = {e["ensemble"]: e for e in p["ensembles"]}
    assert names["P24_lo"]["m0sq"] < -0.6775 < names["P24_hi"]["m0sq"]
    assert names["P32_lo"]["m0sq"] < -0.686 < names["P32_hi"]["m0sq"]


def test_adjudicator_end_to_end_and_fail_closed(tmp_path):
    """Full mini campaign; then break production below 8 usable chains and require STOP."""
    py = sys.executable
    subprocess.run([py, str(REL / "tests/make_mini_plan.py"), str(tmp_path / "plan.json")], check=True)
    plan = json.loads((tmp_path / "plan.json").read_text())
    for t in plan["tasks"]:
        assert run_chain.main(["--plan", str(tmp_path / "plan.json"), "--task", str(t["task"]), "--binary", str(BIN),
                               "--data-root", str(tmp_path / "data"), "--scratch", str(tmp_path / "scr"),
                               "--guard-seconds", "0"]) == 0
    cmd = [py, "-m", "rg1.analyze_stage1", "--plan", str(tmp_path / "plan.json"), "--data", str(tmp_path / "data"),
           "--out", str(tmp_path / "adj")]
    assert subprocess.run(cmd, cwd=REL, capture_output=True).returncode == 0
    r = json.loads((tmp_path / "adj/STAGE1_ADJUDICATION.json").read_text())
    assert r["stage1_gate"]["brackets"] == {"24": "PASS", "32": "PASS"}
    assert r["controls"]["lambda0"]["status"] in ("PASS", "WARN")
    assert all(v["status"] == "PASS" for k, v in r["controls"].items() if k.startswith("reweighting"))
    assert r["fits"] and "running_fits_g" in r["fits"] and isinstance(r["fits"]["running_fits_g"], dict)
    # negative control: mark one production chain's QC as FAIL -> 7 usable -> invalid -> STOP
    q = tmp_path / "data/P32_lo/chain_03.qc.json"; d = json.loads(q.read_text()); d["status"] = "FAIL"; q.write_text(json.dumps(d))
    assert subprocess.run(cmd[:-1] + [str(tmp_path / "adj2")], cwd=REL, capture_output=True).returncode == 0
    r2 = json.loads((tmp_path / "adj2/STAGE1_ADJUDICATION.json").read_text())
    assert r2["ensemble_status"]["P32_lo"]["valid"] is False and r2["stage1_gate"]["stage2_gate"] == "STOP"
    assert r2["z3"]["32"]["bracket"] == "FAIL"
