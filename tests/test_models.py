"""Release checks on synthetic panels: isolation, invariance, censoring identity, failure rows."""
import numpy as np
import pytest

import core_models as cm
import data_pipeline as dp
import routers as rt
from conftest import synth_registry, write_ser_panel


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("models")
    spec = {"T": dict(n_sites=30, per_site=2, drugs=tuple("abcde"), seed=5),
            "Pool1": dict(n_sites=36, per_site=2, drugs=tuple("abcdef"), seed=6),
            "Pool2": dict(n_sites=32, per_site=2, drugs=tuple("abcd"), seed=7, rank=1),
            "Cens": dict(n_sites=24, per_site=2, drugs=tuple("abcde"), seed=8, censor_frac=0.35)}
    for n, kw in spec.items():
        write_ser_panel(tmp, n, **kw)
    reg = synth_registry(tuple(spec))
    reg["panels"]["Few"] = dict(reg["panels"]["T"], min_groups=3)
    write_ser_panel(tmp, "Few", n_sites=3, per_site=14, drugs=tuple("abcd"), seed=9)
    panels = {n: dp.load_panel(reg, n, {"archive": tmp}, allow_lockbox=True) for n in list(spec) + ["Few"]}
    cfg = cm.load_config()
    pool = cm.PoolProvider({n: panels[n] for n in ("T", "Pool1", "Pool2")}, cfg)
    return panels, cfg, pool


def test_every_method_has_prediction_or_explicit_failure(world):
    panels, cfg, pool = world
    run = cm.run_panel(panels["T"], 2, cfg, pool=pool)
    for m, p in run.preds.items():
        missing = set(np.flatnonzero(~np.isfinite(p)).tolist())
        assert missing <= set(run.failures[m]), m
    assert all(len(set(run.ids)) == len(run.ids) for _ in [0])
    assert np.isfinite(run.preds["XGBoost"]).all() and np.isfinite(run.preds["SER"]).all()
    # same manifest for every method: predictions are aligned arrays of the same length
    assert {len(p) for p in run.preds.values()} == {len(run.ids)}


def test_hidden_target_isolation_baselines_and_routed(world):
    panels, cfg, pool = world
    panel = panels["T"]
    ep = dp.build_episodes(panel, 2)
    ef = panel.folds[ep.i]
    base = cm.run_baselines(panel, ep, cfg)
    got, _ = cm.routed_fold(panel, ep, 0, cfg, pool)
    test_idx = np.flatnonzero(ef == 0)
    i0 = int(ep.i[test_idx[0]])
    q0 = int(ep.q[test_idx[0]])
    changed = panel.y.copy()
    changed[i0, q0] += 37.0
    panel2 = type(panel)(**{**panel.__dict__, "y": changed})
    cm.clear_caches()
    base2 = cm.run_baselines(panel2, ep, cfg)
    got2, _ = cm.routed_fold(panel2, ep, 0, cfg, pool)
    untouched = np.array([not (ep.i[e] == i0 and (ep.q[e] == q0 or q0 in ep.s[e])) for e in test_idx])
    keep = test_idx[untouched]
    for name, a, b in (("XGBoost", base.xgb, base2.xgb), ("CovarianceLocalMedian", base.local, base2.local),
                       ("SER", base.ser, base2.ser)):
        np.testing.assert_allclose(a[keep], b[keep], atol=1e-9, err_msg=name)
    # the episode whose hidden query was changed must keep the identical prediction
    same_ep = test_idx[[ep.i[e] == i0 and ep.q[e] == q0 for e in test_idx]]
    for name, a, b in (("XGBoost", base.xgb, base2.xgb), ("SER", base.ser, base2.ser)):
        np.testing.assert_allclose(a[same_ep], b[same_ep], atol=1e-9, err_msg=name)
    for m in got:
        np.testing.assert_allclose(got[m][untouched], got2[m][untouched], atol=1e-7, err_msg=m)


def test_support_permutation_gives_same_aligned_features(world):
    panels, cfg, _ = world
    panel = panels["T"]
    ep = dp.build_episodes(panel, 3)
    rows = np.flatnonzero(panel.folds != 0)
    bank = cm.ExpertBank(panel, rows, cfg)
    sel = np.flatnonzero(panel.folds[ep.i] == 0)[:60]
    i, q, S = ep.i[sel], ep.q[sel], ep.s[sel]
    f1 = cm.featurize(bank, i, q, S)
    perm = [2, 0, 1]
    f2 = cm.featurize(bank, i, q, S[:, perm])
    np.testing.assert_allclose(f1["raw"], f2["raw"], atol=1e-8)
    x1 = rt.aligned_features(np.nan_to_num(f1["P"]), np.nan_to_num(f1["A"]), f1["rho"], f1["r"], np.zeros(len(i)), f1["prior_q"], q)
    x2 = rt.aligned_features(np.nan_to_num(f2["P"]), np.nan_to_num(f2["A"]), f2["rho"], f2["r"], np.zeros(len(i)), f2["prior_q"], q)
    np.testing.assert_allclose(x1, x2, atol=1e-8)


def test_censored_supports_and_zero_weight_keep_xgboost(world):
    panels, cfg, _ = world
    panel = panels["Cens"]
    ep = dp.build_episodes(panel, 2)
    out = cm.run_baselines(panel, ep, cfg)
    limit = (panel.censor[ep.i[:, None], ep.s] != 0).any(axis=1)
    assert limit.any() and (~limit).any()
    np.testing.assert_array_equal(out.ser[limit], out.xgb[limit])
    zero = cm.load_config(overrides={"ser": {"weights": [0.0]}})
    out0 = cm.run_baselines(panel, ep, zero)
    np.testing.assert_array_equal(out0.ser, out0.xgb)
    assert not out0.ser_intervened.any()


def test_interval_and_boundary_losses():
    p = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
    # lower bound 1.5: interval error 0.5 at p=1.0, zero at p=2.0; boundary error 0.5 at both
    lower = cm.interval_error(p[:2], np.array([1.5, 1.5]), np.array([1, 1]))
    np.testing.assert_allclose(lower, [0.5, 0.0])
    np.testing.assert_allclose(np.abs(p[:2] - 1.5), [0.5, 0.5])
    upper = cm.interval_error(np.array([1.0, 2.0]), np.array([1.5, 1.5]), np.array([-1, -1]))
    np.testing.assert_allclose(upper, [0.0, 0.5])
    np.testing.assert_allclose(cm.interval_error(np.array([1.0]), np.array([1.5]), np.array([0])), [0.5])


def test_site_lower_score_rules():
    sites = np.repeat(np.arange(6), 3)
    gain = np.full(18, 0.2)
    assert cm.site_lower_score(gain, sites) > 0
    assert cm.site_lower_score(gain[:6], sites[:6]) == -np.inf                 # only two sites
    one_site = gain.copy(); one_site[sites != 0] = -0.01; one_site[sites == 0] = 5.0
    assert cm.site_lower_score(one_site, sites) == -np.inf                     # gain depends on a single site


def test_sparse_groups_use_documented_fallbacks(world):
    panels, cfg, pool = world
    panel = panels["Few"]                          # three groups -> one fitting fold -> no auxiliary halves
    assert len(np.unique(panel.folds)) == 3
    ep = dp.build_episodes(panel, 2)
    got, info = cm.routed_fold(panel, ep, 0, cfg, pool)
    assert info["nested"]["head"] in ("cross_study_fallback", "none_zero_correction")
    assert not info["nested"]["used"]["local_router"]
    # without any pooled records and no local split, the routed models report an explicit failure
    got2, info2 = cm.routed_fold(panel, ep, 0, cfg, None)
    assert "NestedAligned" in info2["failures"] and info2["failures"]["NestedAligned"] == "no_router_available"
    run = cm.run_panel(panel, 2, cfg, methods=("XGBoost", "NestedAligned"), pool=None)
    assert len(run.failures["NestedAligned"]) == len(run.ids)


def test_neighbors_are_capped_in_small_folds(world):
    panels, cfg, _ = world
    panel = panels["Few"]
    ep = dp.build_episodes(panel, 2)
    rows = np.flatnonzero(panel.folds == 1)               # 14 training profiles < 96 requested neighbours
    sel = np.flatnonzero(panel.folds[ep.i] == 2)[:30]
    nb = cm.neighborhood(panel.y, rows, np.array([dp.site_key(s) for s in panel.sites])[rows], ep.i[sel], ep.q[sel],
                         ep.s[sel], [16, 48, 96])
    assert nb.ok.all() and (nb.eligible == len(rows)).all() and len(rows) < 16


def test_joint_expert_rule_follows_panel_size(world):
    panels, cfg, _ = world
    small = cm.ExpertBank(panels["T"], np.arange(panels["T"].n), cfg)
    assert not small.use_joint
    big_cfg = cm.load_config(overrides={"saer": {"joint_min_profiles": 10}})
    assert cm.ExpertBank(panels["T"], np.arange(panels["T"].n), big_cfg).use_joint


# ---------------------------------------------------------------------------------------
# Pre-freeze revisions of 2026-11-13 (amendments log, section 11)
# ---------------------------------------------------------------------------------------
from pathlib import Path


def test_three_inhibitor_panel_joint_expert_does_not_crash(tmp_path):
    """Revision R1: panels with only 3 inhibitors used to break the joint expert's sampler (shape (n,3) vs (n,2))."""
    write_ser_panel(tmp_path, "Tri", n_sites=40, per_site=2, drugs=tuple("abc"), seed=11)
    panel = dp.load_panel(synth_registry(("Tri",)), "Tri", {"archive": tmp_path}, allow_lockbox=True)
    cfg = cm.load_config(overrides={"saer": {"joint_min_profiles": 10, "joint_min_cells": 10}})
    cm.clear_caches()
    bank = cm.ExpertBank(panel, np.arange(panel.n), cfg)
    assert bank.use_joint and bank.D == 3
    ep = dp.build_episodes(panel, 2)
    out = bank.predict(ep.i[:60], ep.q[:60], ep.s[:60])
    assert np.isfinite(out[:, 6]).all()


def test_pool_provider_does_not_cache_a_failed_computation(world):
    """Revision R1: a failure must not be remembered as 'no records'."""
    panels, cfg, _ = world
    pool = cm.PoolProvider({"T": panels["T"]}, cfg)
    assert pool.panel_records("T", 2) is not None
    assert pool.cache.get(("T", 2)) is not None or any(v is not None for v in pool.cache.values())


def test_standardised_xgboost_is_scale_equivariant(world):
    """Revision R2: with the standardised target, rescaling every response rescales XGBoost's predictions."""
    panels, _, _ = world
    panel = panels["T"]
    ep = dp.build_episodes(panel, 2)
    cfg = cm.load_config(overrides={"revision": {"xgb_standardize_target": True}})
    cm.clear_caches()
    ref = cm.run_baselines(panel, ep, cfg)
    small = type(panel)(**{**panel.__dict__, "y": panel.y * 1e-4})
    cm.clear_caches()
    got = cm.run_baselines(small, ep, cfg)
    np.testing.assert_allclose(got.xgb, ref.xgb * 1e-4, rtol=1e-3, atol=1e-8)


def test_censor_modes_defer_or_use_alternative_on_censored_supports():
    """Revision R3: 'registered' defers to XGBoost on censored supports; 'no_defer_bi' lets the gate judge them."""
    n, g = 60, 6
    sites = np.repeat(np.arange(g), n // g)
    truth, anchor = np.zeros(n), np.ones(n)
    alts = {"bounded_local": np.zeros(n)}                       # the alternative is exactly right everywhere
    limit = np.ones(n, dtype=bool)                              # every episode has a censored support
    zeros = np.zeros(n, dtype=int)
    args = (truth, anchor, alts, sites, [0.5, 1.0], 0.90, 3, limit, zeros, True)
    reg = cm.ser_gate(*args, censor_mode="registered")
    assert reg["expert"] == "XGBoost"
    new = cm.ser_gate(*args, censor_mode="no_defer_bi")
    assert new["expert"] == "bounded_local" and new["weight"] == 1.0
    np.testing.assert_array_equal(cm.ser_apply(reg, anchor, alts["bounded_local"], limit, True, "registered"), anchor)
    np.testing.assert_array_equal(cm.ser_apply(new, anchor, alts["bounded_local"], limit, True, "no_defer_bi"), truth)


def test_production_config_enables_adopted_revisions_and_defaults_stay_registered():
    prod = cm.load_config(Path(__file__).resolve().parents[1] / "configs" / "models.json")["revision"]
    assert prod == {"xgb_standardize_target": True, "neighbors_whitening": "classic",
                    "ser_censor_mode": "no_defer_bi", "ser_selection": "max_lower_bound"}
    assert cm.load_config()["revision"] == {"xgb_standardize_target": False, "neighbors_whitening": "classic",
                                            "ser_censor_mode": "registered", "ser_selection": "max_lower_bound"}


def test_revised_pipeline_runs_on_censored_panel(world):
    panels, _, _ = world
    cfg = cm.load_config(Path(__file__).resolve().parents[1] / "configs" / "models.json")
    panel = panels["Cens"]
    ep = dp.build_episodes(panel, 2)
    cm.clear_caches()
    out = cm.run_baselines(panel, ep, cfg)
    assert np.isfinite(out.xgb).all() and np.isfinite(out.ser).all() and all(not v for v in out.failures.values())
