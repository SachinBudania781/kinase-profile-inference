"""Evaluator tests on synthetic panels with hand-made predictions (no model is trained here)."""
import numpy as np
import pytest

import data_pipeline as dp
import evaluator as ev
from conftest import synth_registry, write_ser_panel


def make_scored(panel, k, preds, failures=None):
    ep = dp.build_episodes(panel, k, max_masks=None)
    t = ep.truth(panel)["y"]
    return ep, ev.score_runs(panel, ep, {m: f(t, ep) for m, f in preds.items()}, failures)


def noise(scale, seed):
    def f(t, ep):
        return t + np.random.default_rng(seed).normal(scale=scale, size=len(t))
    return f


@pytest.fixture
def panel_factory(tmp_path):
    def build(name, **kw):
        write_ser_panel(tmp_path, name, **kw)
        reg = synth_registry((name,))
        return dp.load_panel(reg, name, {"archive": tmp_path}, allow_lockbox=True)
    return build


def test_scoring_join_requires_explicit_failures(panel_factory):
    p = panel_factory("S1", n_sites=8, per_site=2, drugs=("a", "b", "c", "d", "e"), seed=1)
    ep = dp.build_episodes(p, 2, max_masks=None)
    good = np.zeros(len(ep))
    bad = good.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError):
        ev.score_runs(p, ep, {"XGBoost": good, "SER": bad})
    sc = ev.score_runs(p, ep, {"XGBoost": good, "SER": bad}, {"SER": {3: "no_router_available"}})
    assert np.isnan(sc.preds["SER"][3])
    with pytest.raises(ValueError):
        ev.score_runs(p, ep, {"XGBoost": good[:-1]})


def test_gain_formula_and_equal_study_weights(panel_factory):
    studies = {}
    for n, (name, scale_ref, scale_ser) in enumerate((("S1", 0.5, 0.45), ("S2", 0.4, 0.40))):
        p = panel_factory(name, n_sites=10, per_site=2, drugs=("a", "b", "c", "d"), seed=n + 10)
        _, sc = make_scored(p, 2, {"XGBoost": noise(scale_ref, 1), "SER": noise(scale_ser, 2)})
        studies[name] = {2: sc}
    rows, _ = ev.study_table(studies, "SER", None)
    for r in rows:
        assert r["gain_percent"] == pytest.approx(100 * (r["mae_xgboost"] - r["mae_method"]) / r["mae_xgboost"])
    per, _ = ev.study_gains(rows, {})
    assert np.mean(list(per.values())) == pytest.approx((rows[0]["gain_percent"] + rows[1]["gain_percent"]) / 2)


def test_identical_predictions_are_ties_and_not_wins(panel_factory):
    studies = {}
    for n in range(7):
        p = panel_factory(f"T{n}", n_sites=8, per_site=2, drugs=("a", "b", "c", "d"), seed=n)
        _, sc = make_scored(p, 2, {"XGBoost": noise(0.4, n), "SER": noise(0.4, n)})
        studies[f"T{n}"] = {2: sc}
    res = ev.evaluate(studies, {"bootstrap_draws": 200})
    wt = res["win_tie"]
    assert wt["studies_tied"] == 7 and wt["studies_won"] == 0 and wt["studies_lost"] == 0
    assert wt["meets_descriptive_5_of_7"] is False
    assert res["primary"]["gain_percent"] == pytest.approx(0.0)
    assert res["primary"]["meets_target"] is False
    assert all(r["intervention_rate"] == 0 for r in res["intervention"])


def test_win_rule_counts_strict_wins_only(panel_factory):
    studies = {}
    for n in range(7):
        p = panel_factory(f"W{n}", n_sites=8, per_site=2, drugs=("a", "b", "c", "d"), seed=20 + n)
        ser_scale = 0.2 if n < 5 else 0.4                 # first five strictly better, last two identical
        _, sc = make_scored(p, 2, {"XGBoost": noise(0.4, n), "SER": noise(ser_scale, n)})
        studies[f"W{n}"] = {2: sc}
    res = ev.evaluate(studies, {"bootstrap_draws": 200})
    wt = res["win_tie"]
    assert wt["studies_won"] == 5 and wt["studies_tied"] == 2
    assert wt["meets_descriptive_5_of_7"] is True
    assert wt["sign_test_upper_tail"] == pytest.approx(29 / 128)
    assert res["primary"]["gain_percent"] > 2


def test_bootstrap_is_deterministic_and_paired(panel_factory):
    p = panel_factory("B1", n_sites=12, per_site=2, drugs=("a", "b", "c", "d", "e"), seed=5)
    _, sc2 = make_scored(p, 2, {"XGBoost": noise(0.5, 1), "SER": noise(0.4, 2)})
    _, sc3 = make_scored(p, 3, {"XGBoost": noise(0.5, 3), "SER": noise(0.4, 4)})
    studies = {"B1": {2: sc2, 3: sc3}}
    a = ev.evaluate(studies, {"bootstrap_draws": 300})
    b = ev.evaluate(studies, {"bootstrap_draws": 300})
    assert a["primary"]["ci95"] == b["primary"]["ci95"]
    M = ev.bootstrap_matrices(studies, 50, ev.SEED)["B1"]
    assert M.shape == (50, sc2.n_sites) and np.all(M.sum(axis=1) == sc2.n_sites)
    lo, hi = a["primary"]["ci95"]
    assert lo <= a["primary"]["gain_percent"] <= hi


def test_holm_step_down():
    out = ev.holm({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5}, 0.05)
    assert out["a"]["reject"] and out["a"]["threshold"] == pytest.approx(0.0125)
    assert not out["b"]["reject"]                     # 0.02 > 0.05/3
    assert not out["c"]["reject"] and not out["d"]["reject"]  # stop at first failure
    assert ev.holm({"a": 0.01, "b": 0.02}, 0.05)["b"]["reject"]


def test_robustness_leave_one_out_and_weights(panel_factory):
    studies = {}
    for n, s in enumerate((0.2, 0.3, 0.45)):
        p = panel_factory(f"R{n}", n_sites=6 + 3 * n, per_site=2, drugs=("a", "b", "c", "d"), seed=40 + n)
        _, sc = make_scored(p, 2, {"XGBoost": noise(0.5, n), "SER": noise(s, n + 9)})
        studies[f"R{n}"] = {2: sc}
    res = ev.evaluate(studies, {"bootstrap_draws": 200})
    rob = res["robustness"]
    per = res["primary"]["per_study_gain"]
    assert rob["median_study_gain"] == pytest.approx(np.median(list(per.values())))
    assert set(rob["leave_one_study_out"]) == set(per)
    for left, rec in rob["leave_one_study_out"].items():
        assert rec["gain_percent"] == pytest.approx(np.mean([v for k, v in per.items() if k != left]))
    assert sum(rob["episode_weighted"]["weights"].values()) == pytest.approx(1.0)
    assert rob["site_weighted"]["weights"]["R2"] > rob["site_weighted"]["weights"]["R0"]
    assert res["primary"]["gain_percent"] == pytest.approx(np.mean(list(per.values())))   # primary stays equal-weight


def test_failures_are_compared_on_common_episodes(panel_factory):
    p = panel_factory("F1", n_sites=8, per_site=2, drugs=("a", "b", "c", "d"), seed=7)
    ep = dp.build_episodes(p, 2, max_masks=None)
    t = ep.truth(p)["y"]
    ser = t + 0.1
    ser[:5] = np.nan
    sc = ev.score_runs(p, ep, {"XGBoost": t + 0.5, "SER": ser}, {"SER": {e: "x" for e in range(5)}})
    rows, _ = ev.study_table({"F1": {2: sc}}, "SER", None)
    assert rows[0]["failed_episodes"] == 5 and rows[0]["episodes"] == len(ep) - 5
    assert rows[0]["mae_method"] == pytest.approx(0.1) and rows[0]["mae_xgboost"] == pytest.approx(0.5)


def test_interval_error_on_censored_truth(panel_factory):
    p = panel_factory("C1", n_sites=10, per_site=2, drugs=("a", "b", "c", "d"), seed=8, censor_frac=0.4)
    ep = dp.build_episodes(p, 2, max_masks=None)
    t = ep.truth(p)
    sc = ev.score_runs(p, ep, {"XGBoost": t["y"] + 1.0, "SER": t["y"] + 1.0})
    ie = ev.interval_err(sc, "SER")
    assert np.all(ie[t["censor"] == 1] == 0)                       # lower bound, prediction above it: no penalty
    assert np.all(ie[t["censor"] == -1] == pytest.approx(1.0))     # upper bound, prediction above it: penalised
    assert np.all(ie[t["censor"] == 0] == pytest.approx(1.0))
    assert np.all(ev.abs_err(sc, "SER") == pytest.approx(1.0))     # primary loss ignores censor direction
    assert ev.exact_and_interval_tables({"C1": {2: sc}}, "SER")


# ---------------------------------------------------------------------------------- T1
def _toy_scored(panel, k, preds_fn):
    ep = dp.build_episodes(panel, k, max_masks=None)
    t = ep.truth(panel)["y"]
    return ev.score_runs(panel, ep, {m: f(t, ep) for m, f in preds_fn.items()})


def test_t1_min_eligible_enrichment_and_ap(panel_factory):
    p = panel_factory("T1a", n_sites=30, per_site=2, drugs=("a", "b", "c", "d"), seed=11)
    p.scale_kind = "ic50_log10_ratio"
    oracle = lambda t, ep: t                                       # noqa: E731
    anti = lambda t, ep: -t                                        # noqa: E731
    sc = _toy_scored(p, 2, {"XGBoost": anti, "SER": oracle})
    cfg = {**ev.EVAL_DEFAULTS}
    out = ev.t1_screening(sc, ["XGBoost", "SER"], cfg)
    assert out["defined"]
    assert out["groups_used"] + out["groups_below_minimum"] == out["groups_total"]
    if out["groups_used"]:
        o, a = out["methods"]["SER"], out["methods"]["XGBoost"]
        assert o["precision"] >= a["precision"]
        assert o["enrichment_factor"] == pytest.approx(o["precision"] / out["base_rate"])
    strict = ev.t1_screening(sc, ["SER"], {**cfg, "t1_min_eligible": 10 ** 6})
    assert strict["groups_used"] == 0 and strict["groups_below_minimum"] == strict["groups_total"]


def test_average_precision_values():
    assert ev.average_precision(np.array([1, 0, 0])) == pytest.approx(1.0)
    assert ev.average_precision(np.array([0, 0, 1])) == pytest.approx(1 / 3)
    assert ev.average_precision(np.array([1, 0, 1, 0])) == pytest.approx((1 + 2 / 3) / 2)
    assert np.isnan(ev.average_precision(np.array([0, 0])))


def test_t1_undefined_for_unsupported_scales(panel_factory):
    p = panel_factory("T1b", n_sites=10, per_site=2, drugs=("a", "b", "c", "d"), seed=12)
    p.scale_kind = "other"
    sc = _toy_scored(p, 2, {"XGBoost": lambda t, e: t, "SER": lambda t, e: t})
    assert ev.t1_screening(sc, ["SER"], ev.EVAL_DEFAULTS)["defined"] is False


def test_t1_ambiguous_censoring_is_counted_not_guessed():
    # statement: hidden response >= 1.0
    det, val = ev._at_least(np.array([1.2, 0.5, 1.5, 0.4]), np.array([-1, 1, 1, -1]), 1.0)
    assert det.tolist() == [False, False, True, True]        # '<=1.2' and '>=0.5' cannot decide; '>=1.5' and '<=0.4' can
    assert val.tolist()[2] is True and val.tolist()[3] is False
    # statement: support response < log10(3)
    det, val = ev._below(np.array([0.3, 0.6, 0.2]), np.array([-1, 1, 1]), np.log10(3))
    assert det.tolist() == [True, True, False]
    assert val.tolist()[0] is True and val.tolist()[1] is False


# ---------------------------------------------------------------------------------- T2
def test_t2_requires_two_remaining_inhibitors(panel_factory):
    p3 = panel_factory("T2a", n_sites=8, per_site=2, drugs=("a", "b", "c"), seed=13)
    sc = _toy_scored(p3, 2, {"XGBoost": lambda t, e: t, "SER": lambda t, e: t})
    out = ev.t2_choice(sc, ["SER"], ev.EVAL_DEFAULTS)
    assert out["defined"] is False and "remaining" in out["reason"]
    p4 = panel_factory("T2b", n_sites=8, per_site=2, drugs=("a", "b", "c", "d"), seed=14)
    sc4 = _toy_scored(p4, 3, {"XGBoost": lambda t, e: t, "SER": lambda t, e: t})
    assert ev.t2_choice(sc4, ["SER"], ev.EVAL_DEFAULTS)["defined"] is False


def test_t2_oracle_has_zero_regret_and_anti_oracle_positive(panel_factory):
    p = panel_factory("T2c", n_sites=10, per_site=2, drugs=("a", "b", "c", "d", "e"), seed=15)
    sc = _toy_scored(p, 2, {"XGBoost": lambda t, e: -t, "SER": lambda t, e: t})
    out = ev.t2_choice(sc, ["XGBoost", "SER"], ev.EVAL_DEFAULTS)
    assert out["defined"] and out["support_sets"] > 0
    assert out["methods"]["SER"]["mean_regret"] == pytest.approx(0.0)
    assert out["methods"]["SER"]["chosen_is_best_fraction"] == pytest.approx(1.0)
    assert out["methods"]["XGBoost"]["mean_regret"] > 0
    assert out["methods"]["SER"]["eligible_variants"] == p.n


def test_t2_excludes_sets_with_censored_responses(panel_factory):
    p = panel_factory("T2d", n_sites=10, per_site=2, drugs=("a", "b", "c", "d", "e"), seed=16, censor_frac=0.6)
    sc = _toy_scored(p, 2, {"XGBoost": lambda t, e: t, "SER": lambda t, e: t})
    out = ev.t2_choice(sc, ["SER"], ev.EVAL_DEFAULTS)
    assert out["excluded_censored"] > 0


# ---------------------------------------------------------------------------------- T3
def test_t3_pairs_two_drug_base_with_three_drug_supersets(panel_factory):
    p = panel_factory("T3a", n_sites=10, per_site=2, drugs=("a", "b", "c", "d", "e"), seed=17)
    _, s2 = make_scored(p, 2, {"XGBoost": noise(0.6, 1), "SER": noise(0.6, 1)})
    ep3 = dp.build_episodes(p, 3, max_masks=None)
    t3 = ep3.truth(p)["y"]
    s3 = ev.score_runs(p, ep3, {"XGBoost": t3 + 0.2, "SER": t3 + 0.2})      # third measurement makes it much better
    out = ev.t3_third_measurement(s2, s3, ["SER"])
    assert out["defined"] and out["base_episodes"] > 0
    assert out["methods"]["SER"]["mean_error_reduction"] > 0
    assert out["methods"]["SER"]["fraction_helped"] > 0.5


# ---------------------------------------------------------------------------------- T4
def test_t4_strata_compare_within_query_inhibitor(panel_factory):
    p = panel_factory("T4a", n_sites=12, per_site=2, drugs=("a", "b", "c", "d", "e", "f"), seed=18)
    sc = _toy_scored(p, 2, {"XGBoost": noise(0.5, 1), "SER": noise(0.3, 2)})
    out = ev.t4_modes(sc, ["SER"], ev.EVAL_DEFAULTS)
    assert out["defined"]
    s = out["strata"]
    total = sum(v.get("observations", 0) for v in s.values())
    assert total == len(sc.ids)
    for label, entry in s.items():
        if entry.get("observations"):
            qs = [r["query"] for r in entry["by_query"]["SER"]]
            assert len(qs) == len(set(qs))                              # one row per query inhibitor
            assert all(r["observations"] > 0 for r in entry["by_query"]["SER"])


def test_t4_undefined_without_mode_annotation(panel_factory):
    p = panel_factory("T4b", n_sites=8, per_site=2, drugs=("a", "b", "c", "d"), seed=19)
    p.modes = tuple("unspecified" for _ in p.drugs)
    sc = _toy_scored(p, 2, {"XGBoost": lambda t, e: t, "SER": lambda t, e: t})
    assert ev.t4_modes(sc, ["SER"], ev.EVAL_DEFAULTS)["defined"] is False


# ---------------------------------------------------------------------------------- output
def test_results_are_json_serialisable(tmp_path, panel_factory):
    p = panel_factory("J1", n_sites=10, per_site=2, drugs=("a", "b", "c", "d", "e"), seed=21)
    _, s2 = make_scored(p, 2, {"XGBoost": noise(0.5, 1), "SER": noise(0.4, 2), "QueryMedian": noise(0.7, 3)})
    _, s3 = make_scored(p, 3, {"XGBoost": noise(0.5, 4), "SER": noise(0.4, 5), "QueryMedian": noise(0.7, 6)})
    res = ev.evaluate({"J1": {2: s2, 3: s3}}, {"bootstrap_draws": 100})
    ev.write_results(res, tmp_path)
    import json
    back = json.loads((tmp_path / "results.json").read_text())
    assert back["primary"]["complete"] is True
    assert "QueryMedian" in back["plain_baselines"]
    assert (tmp_path / "panel_budget_table.csv").exists()
