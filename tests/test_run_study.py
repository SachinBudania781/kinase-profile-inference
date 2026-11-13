"""Runner, lockbox gate and evaluator CLI on synthetic panels (fast methods only)."""
import json
from pathlib import Path

import numpy as np
import pytest

import data_pipeline as dp
import evaluator as ev
import integrity as ig
import run_study as rs
from conftest import write_ser_panel

FAST = "XGBoost,SER,QueryMedian,RidgeSupports,CovarianceLocalMedian"


@pytest.fixture
def world(tmp_path):
    archive, lock_b = tmp_path / "archive", tmp_path / "lockB"
    write_ser_panel(archive, "DevA", n_sites=12, per_site=2, drugs=("a", "b", "c", "d", "e"), seed=1)
    write_ser_panel(archive, "DevB", n_sites=10, per_site=2, drugs=("a", "b", "c", "d"), seed=2, censor_frac=0.2)
    write_ser_panel(lock_b, "LockB1", n_sites=10, per_site=2, drugs=("a", "b", "c", "d"), seed=3)
    reg = {"seed": dp.SEED, "panels": {
        "DevA": {"role": "core_dev", "root": "archive", "format": "ser_csv", "multi_gene": False, "prior": "drug_median"},
        "DevB": {"role": "stress_test", "root": "archive", "format": "ser_csv", "multi_gene": False, "prior": "drug_median"},
        "LockB1": {"role": "lockbox_B", "root": "lockbox_b", "format": "ser_csv", "multi_gene": False, "prior": "drug_median"}}}
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps(reg))
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "a.py").write_text("x = 1\n")
    (repo / "configs" / "models.json").write_text(json.dumps({}))
    return {"archive": archive, "lock_b": lock_b, "registry": reg_path, "repo": repo, "tmp": tmp_path}


def args_for(w, out, roles="core_dev,stress_test", extra=()):
    return ["--archive-root", str(w["archive"]), "--lockbox-b-root", str(w["lock_b"]), "--registry", str(w["registry"]),
            "--repo", str(w["repo"]), "--config", str(w["repo"] / "configs" / "models.json"), "--roles", roles,
            "--methods", FAST, "--out", str(out), *extra]


def test_dev_run_then_evaluate(world):
    out = world["tmp"] / "runs"
    rs.main(args_for(world, out))
    man = json.loads((out / "run_manifest.json").read_text())
    assert not man["errors"] and len(man["runs"]) >= 3
    for run in man["runs"]:
        assert (out / run["file"]).exists()
    # predictions files carry no hidden responses: only ids, fold, method columns, failure reasons
    head = __import__("gzip").open(out / man["runs"][0]["file"], "rt").readline().strip().split(",")
    assert head[0] == "episode_id" and "y" not in head and "truth" not in head
    # evaluator CLI joins truth and writes results
    res_dir = world["tmp"] / "results"
    ev.main(["--runs", str(out), "--archive-root", str(world["archive"]), "--registry", str(world["registry"]),
             "--out", str(res_dir), "--draws", "100"])
    res = json.loads((res_dir / "results.json").read_text())
    assert "SER" in res["summaries"] and res["primary"]["complete"]
    assert {"QueryMedian", "RidgeSupports"} <= set(res["plain_baselines"])
    assert all(0 <= r["intervention_rate"] <= 1 for r in res["intervention"])


def test_logged_intervention_rate_matches_predictions(world):
    out = world["tmp"] / "runs"
    rs.main(args_for(world, out, extra=("--panels", "DevA", "--budgets", "2")))
    preds, fails, ids = ev.load_predictions_csv(out / "DevA_k2_predictions.csv.gz")
    logged = json.loads((out / "DevA_k2_decisions.json").read_text())["ser_intervention_rate"]
    assert logged == pytest.approx((preds["SER"] != preds["XGBoost"]).mean())
    assert np.isfinite(preds["SER"]).all()


def test_lockbox_requires_flag_record_and_runs_once(world):
    out = world["tmp"] / "lockrun"
    with pytest.raises(SystemExit):
        rs.main(args_for(world, out, roles="lockbox_B"))                          # no --allow-lockbox
    with pytest.raises(SystemExit):
        rs.main(args_for(world, out, roles="lockbox_B", extra=("--allow-lockbox",)))   # no integrity record
    record = world["tmp"] / "record.json"
    rec = ig.compute(world["repo"], world["archive"], world["lock_b"], world["registry"])
    record.write_text(json.dumps(rec))
    (world["repo"] / "a.py").write_text("x = 2\n")                                 # code changed after the freeze
    with pytest.raises(SystemExit) as e:
        rs.main(args_for(world, out, roles="lockbox_B", extra=("--allow-lockbox", "--integrity-record", str(record))))
    assert "Integrity check failed" in str(e.value)
    assert not (out / rs.LOCKBOX_MARKER).exists()
    (world["repo"] / "a.py").write_text("x = 1\n")                                 # restore
    rs.main(args_for(world, out, roles="lockbox_B", extra=("--allow-lockbox", "--integrity-record", str(record))))
    assert (out / rs.LOCKBOX_MARKER).exists() and (out / "LockB1_k2_predictions.csv.gz").exists()
    with pytest.raises(SystemExit) as e:                                           # second run refused
        rs.main(args_for(world, out, roles="lockbox_B", extra=("--allow-lockbox", "--integrity-record", str(record))))
    assert "already been run" in str(e.value)


def test_evaluator_refuses_lockbox_without_flag(world):
    with pytest.raises(SystemExit):
        ev.main(["--runs", str(world["tmp"]), "--archive-root", str(world["archive"]), "--registry", str(world["registry"]),
                 "--roles", "lockbox_B"])


def test_runner_records_errors_instead_of_hiding_them(world, monkeypatch):
    out = world["tmp"] / "runs_err"
    import core_models as cm
    real = cm.run_panel

    def boom(panel, k, *a, **kw):
        if panel.name == "DevB":
            raise RuntimeError("synthetic failure")
        return real(panel, k, *a, **kw)
    monkeypatch.setattr(cm, "run_panel", boom)
    with pytest.raises(SystemExit):
        rs.main(args_for(world, out))
    man = json.loads((out / "run_manifest.json").read_text())
    assert man["errors"] and all(e["panel"] == "DevB" for e in man["errors"])
    assert any(r["panel"] == "DevA" for r in man["runs"])
