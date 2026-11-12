"""Stage 2 plan contract, end-to-end adjudication on a mini campaign (pooled with a mini Stage 1), and fail-closed checks.
The runner, capture and QC modules are byte-identical to Stage 1 and covered by its failure-injection suite."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REL = Path(__file__).resolve().parents[1]
S1 = REL.parent / "stage1_release_v1"
BIN = REL.parent / "bin" / "phi4_local2"
sys.path.insert(0, str(REL))
from rg1 import run_chain  # noqa: E402

PY = sys.executable


def test_plan_contract():
    p = json.loads((REL / "PLAN.json").read_text())
    assert p["allocated_core_hours"] <= p["resource_ceiling_core_hours"] == 1400.0
    assert p["design"]["targets_rel"] == {"12": 0.002, "16": 0.003, "24": 0.0135, "32": 0.0135}
    s1 = json.loads((S1 / "PLAN.json").read_text())
    seeds = [t["seed"] for t in p["tasks"]]
    assert len(seeds) == len(set(seeds)) and not set(seeds) & {t["seed"] for t in s1["tasks"]}
    assert p["sampler"] == s1["sampler"]
    e1 = {e["ensemble"]: e for e in s1["ensembles"]}
    for e in p["ensembles"]:
        assert e["chains"] >= 8 and e["n_therm"] >= 5000 and e["lambda"] == 0.4 and e["h"] == 0.0 and e["zero_mode_gibbs"] == 1
        if e["ensemble"] in p["stage1"]["pooled_with"]:
            q = e1[p["stage1"]["pooled_with"][e["ensemble"]]]
            assert all(e[k] == q[k] for k in ("L", "m0sq", "lambda", "h", "n_therm", "n_meas", "n_or", "meas_every", "zero_mode_gibbs"))
        assert e["time_limit_hours"] >= 1.4 * e["expected_hours_per_chain"]


def _run_all(plan, data, scr):
    for t in json.loads(Path(plan).read_text())["tasks"]:
        assert run_chain.main(["--plan", str(plan), "--task", str(t["task"]), "--binary", str(BIN),
                               "--data-root", str(data), "--scratch", str(scr), "--guard-seconds", "0"]) == 0


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    t = tmp_path_factory.mktemp("s2")
    subprocess.run([PY, str(S1 / "tests/make_mini_plan.py"), str(t / "p1.json")], check=True)
    _run_all(t / "p1.json", t / "d1", t / "scr")
    subprocess.run([PY, "-m", "rg1.analyze_stage1", "--plan", str(t / "p1.json"), "--data", str(t / "d1"), "--out", str(t / "a1")],
                   cwd=S1, check=True, capture_output=True)
    adj1 = t / "a1/STAGE1_ADJUDICATION.json"
    subprocess.run([PY, str(REL / "tests/make_mini_plan.py"), str(t / "p2.json"), str(t / "p1.json"), str(adj1)], check=True)
    _run_all(t / "p2.json", t / "d2", t / "scr")
    cmd = [PY, "-m", "rg1.analyze_stage2", "--plan", str(t / "p2.json"), "--data", str(t / "d2"), "--stage1-plan", str(t / "p1.json"),
           "--stage1-data", str(t / "d1"), "--stage1-adj", str(adj1), "--ncal", "40", "--workers", "4", "--out"]
    return t, cmd


def test_end_to_end(mini):
    t, cmd = mini
    p = subprocess.run(cmd + [str(t / "o")], cwd=REL, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    r = json.loads((t / "o/STAGE2_RESULT.json").read_text())
    assert r["prerequisites"]["brackets"] == {"12": "PASS", "16": "PASS", "24": "PASS", "32": "PASS"}
    assert r["ensembles"]["L24_lo"]["replica"]["n_chains"] == 16 and r["ensembles"]["Q24_lo_only"]["replica"]["n_chains"] == 8
    assert all(c["g"]["status"] != "FAIL" for k, c in r["controls"].items() if k.startswith("stage2_vs_stage1"))
    f = r["fits"]
    assert isinstance(f["running_fits_g"], dict) and "L12-32" in f["running_fits_g"] and isinstance(f["pt_compare"], list)
    assert set(f["L16-32_sensitivity"]["L"]) == {16, 24, 32}
    cal = f["calibration_L12-32"]
    assert cal["n"] == 40 and 0 <= cal["alpha0_false_positive_rate"]["both"] <= 1
    assert r["claim"]["verdict"] in ("POSITIVE_RUNNING_RESOLVED", "NEGATIVE_SLOPE_RESOLVED", "NOT_RESOLVED_BOUNDED",
                                     "INCONCLUSIVE_PREREQUISITE_FAIL")


def test_fail_closed_on_lost_chain(mini):
    t, cmd = mini
    q = t / "d2/Q32_hi/chain_05.qc.json"; d = json.loads(q.read_text()); keep = dict(d)
    d["status"] = "FAIL"; q.write_text(json.dumps(d))
    try:
        assert subprocess.run(cmd + [str(t / "o2")], cwd=REL, capture_output=True).returncode == 0
        r = json.loads((t / "o2/STAGE2_RESULT.json").read_text())
        assert r["ensemble_status"]["Q32_hi"]["valid"] is False
        assert r["z3"]["32"]["bracket"] == "FAIL" and r["claim"]["verdict"] == "INCONCLUSIVE_PREREQUISITE_FAIL"
    finally:
        q.write_text(json.dumps(keep))


def test_refuses_unpinned_stage1(mini, tmp_path):
    t, cmd = mini
    bad = tmp_path / "adj.json"; bad.write_text((t / "a1/STAGE1_ADJUDICATION.json").read_text() + " ")
    c = list(cmd); c[c.index("--stage1-adj") + 1] = str(bad)
    p = subprocess.run(c + [str(tmp_path / "o")], cwd=REL, capture_output=True, text=True)
    assert p.returncode != 0 and "differ from the hashes pinned" in p.stderr
