"""Smoke tests for the development-only deep-learning baselines (tiny epochs, synthetic panels)."""
import numpy as np
import pytest
import torch

import data_pipeline as dp
import dl_baselines as dl
from conftest import synth_registry, write_ser_panel

TINY = {**dl.DL_DEFAULTS, "dim": 16, "heads": 2, "layers": 1, "epochs": 2, "batch": 64, "patience": 2}


@pytest.fixture
def panel(tmp_path):
    write_ser_panel(tmp_path, "DL1", n_sites=14, per_site=2, drugs=("a", "b", "c", "d", "e"), seed=3)
    reg = synth_registry(("DL1",))
    return dp.load_panel(reg, "DL1", {"archive": tmp_path}, allow_lockbox=True)


@pytest.mark.parametrize("arch", dl.ARCHS)
def test_each_architecture_runs_and_is_finite(panel, arch):
    ep = dp.build_episodes(panel, 2, max_masks=None)
    folds = panel.folds[ep.i]
    fit = np.flatnonzero(folds >= 2)
    val = np.flatnonzero(folds == 1)
    pred = dl.train_predict(arch, panel, ep, fit, val, TINY)
    assert pred.shape == (len(val),) and np.isfinite(pred).all()


def test_forward_is_invariant_to_support_order(panel):
    ep = dp.build_episodes(panel, 3, max_masks=None)
    scale = dl.target_scale(panel, np.arange(panel.n))
    val, obs, q = dl.episode_tensors(panel, ep, scale)
    torch.manual_seed(0)
    for arch in dl.ARCHS:
        net = dl.build_model(arch, panel, np.arange(panel.n), TINY).eval()
        out = net(val[:32], obs[:32], q[:32])
        assert out.shape == (32,)
        # dense layout carries no support order: permuting the stored support list cannot change the tensors
        perm = ep.subset(np.arange(32))
        perm.s = perm.s[:, ::-1].copy()
        v2, o2, q2 = dl.episode_tensors(panel, perm, scale)
        assert torch.equal(v2, val[:32]) and torch.equal(o2, obs[:32])


def test_hidden_query_value_never_enters_inputs(panel):
    ep = dp.build_episodes(panel, 2, max_masks=None)
    scale = dl.target_scale(panel, np.arange(panel.n))
    val, obs, q = dl.episode_tensors(panel, ep, scale)
    assert torch.all(val[torch.arange(len(q)), q] == 0) and torch.all(obs[torch.arange(len(q)), q] == 0)


def test_validation_report_has_xgboost_and_dl(panel):
    rep = dl.validation_report(panel, 2, TINY, archs=("cnn", "lstm"))
    assert set(rep["dl_mae"]) == {"cnn", "lstm"} and rep["xgboost_best_mae"] > 0
    assert rep["validation_fold"] != rep["outer_fold"] and rep["best_dl"] in rep["dl_mae"]


def test_refuses_lockbox_panels(tmp_path):
    with pytest.raises(SystemExit):
        write_ser_panel(tmp_path, "LB", n_sites=6, drugs=("a", "b", "c"), seed=1)
        import json
        reg = {"seed": dp.SEED, "panels": {"LB": {"role": "lockbox_B", "root": "lockbox_b", "format": "ser_csv",
                                                    "multi_gene": False, "prior": "drug_median"}}}
        path = tmp_path / "r.json"
        path.write_text(json.dumps(reg))
        dl.main(["--archive-root", str(tmp_path), "--panels", "LB", "--registry", str(path), "--out", str(tmp_path / "o")])
