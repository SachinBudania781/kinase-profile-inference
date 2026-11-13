import hashlib
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pytest

import data_pipeline as dp
from conftest import dev_root, requires_dev_data, synth_registry, write_ser_panel


def test_fold_rule_matches_registered_formula():
    sites = [101, 102, 103, 104, 105, 106, 107, 108]
    folds = dp.fold_assignment(sites)
    order = sorted(set(sites), key=lambda p: hashlib.sha256(f"{dp.SEED}:{p}".encode()).digest())
    expected = {p: i % 5 for i, p in enumerate(order)}
    assert folds.tolist() == [expected[s] for s in sites]


def test_fold_rule_matches_released_reference_when_available():
    ref = os.environ.get("KPI_REFERENCE_SER")
    if not ref:
        pytest.skip("set KPI_REFERENCE_SER to the folder that contains the released 'ser' package")
    import sys
    sys.path.insert(0, ref)
    from ser.models import fold_assignment as released
    rng = np.random.default_rng(0)
    positions = rng.integers(1, 400, size=300)
    assert dp.fold_assignment(positions).tolist() == released(positions).tolist()


def test_whole_sites_share_a_fold():
    sites = np.repeat(np.arange(30), 4)
    folds = dp.fold_assignment(sites)
    for s in range(30):
        assert len(set(folds[sites == s])) == 1


def test_at_least_four_groups_required():
    with pytest.raises(dp.PanelValidationError):
        dp.fold_assignment([1, 1, 2, 2, 3, 3])
    assert dp.fold_assignment([1, 2, 3], min_groups=3).max() == 2


def test_lockbox_panel_needs_explicit_access(synth_root):
    reg = synth_registry()
    with pytest.raises(dp.LockboxAccessError):
        dp.load_panel(reg, "SynthA", {"archive": synth_root})
    assert dp.load_panel(reg, "SynthA", {"archive": synth_root}, allow_lockbox=True).n == 24


def test_episode_counts_match_formula(synth_panels):
    p = synth_panels["SynthA"]                       # 24 variants, 5 drugs
    for k in (2, 3):
        ep = dp.build_episodes(p, k)
        assert len(ep) == dp.expected_full_episodes(p.n, p.n_drugs, k) == p.n * 5 * math.comb(4, k)
    p3 = synth_panels["Synth3"]                      # 3 drugs: only k=2
    assert p3.budgets() == (2,)
    assert len(dp.build_episodes(p3, 2)) == p3.n * 3
    with pytest.raises(dp.PanelValidationError):
        dp.build_episodes(p3, 3)


def test_query_never_in_support_and_ids_unique(synth_panels):
    for p in synth_panels.values():
        for k in p.budgets():
            ep = dp.build_episodes(p, k)
            assert not (ep.s == ep.q[:, None]).any()
            assert (np.diff(ep.s, axis=1) > 0).all()          # ascending inhibitor order
            assert len(set(ep.ids.tolist())) == len(ep)


def test_episode_set_has_no_hidden_target(synth_panels):
    p = synth_panels["SynthA"]
    ep = dp.build_episodes(p, 2)
    inputs = ep.inputs(p)
    assert set(inputs) == {"i", "q", "s", "support_y", "support_censor"}
    t = ep.truth(p)
    np.testing.assert_array_equal(t["y"], p.y[ep.i, ep.q])
    # changing the hidden target leaves the inputs unchanged
    p2 = type(p)(**{**p.__dict__, "y": p.y.copy()})
    p2.y[ep.i[0], ep.q[0]] += 100
    np.testing.assert_array_equal(ep.inputs(p2)["support_y"][0], inputs["support_y"][0])


def test_sampling_is_deterministic_and_uniform_in_size():
    obs = np.arange(40)
    a = dp.enumerate_profile_masks(obs, 3, 25, np.random.default_rng(5))
    b = dp.enumerate_profile_masks(obs, 3, 25, np.random.default_rng(5))
    assert a == b and len(a) == len(set(a)) == 25
    assert all(q not in s and list(s) == sorted(s) for q, s in a)


def test_missing_cells_are_never_used(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.normal(size=(30, 6)); y[::3, 2] = np.nan; y[::5, 4] = np.nan
    d = tmp_path / "Dev"; d.mkdir()
    np.savez(d / "response_matrix.npz", Y=y, W=np.zeros_like(y), labels=np.array([f"v{i}" for i in range(30)]),
             genes=np.array(["G"] * 30), groups=np.arange(30) // 2, drugs=np.array(list("abcdef")))
    (d / "study.json").write_text('{"study": "Dev", "kinase": "G"}')
    reg = {"panels": {"Dev": {"role": "core_dev", "root": "archive", "format": "matrix_npz", "min_groups": 4,
                               "max_masks_per_profile": {"2": None, "3": None}}}}
    p = dp.load_panel(reg, "Dev", {"archive": tmp_path})
    ep = dp.build_episodes(p, 2)
    cells = np.column_stack([ep.q[:, None], ep.s])
    assert np.isfinite(p.y[ep.i[:, None], cells]).all()


def test_invalid_inputs_are_rejected(tmp_path):
    reg = synth_registry(("Bad",))
    d = write_ser_panel(tmp_path, "Bad")
    text = (d / "SER_responses.csv").read_text().splitlines()
    (d / "SER_responses.csv").write_text("\n".join(text[:-1]) + "\n")          # drop one row => incomplete profile
    with pytest.raises(dp.PanelValidationError):
        dp.load_panel(reg, "Bad", {"archive": tmp_path}, allow_lockbox=True)
    d = write_ser_panel(tmp_path, "Bad2")
    (d / "SER_responses.csv").write_text((d / "SER_responses.csv").read_text().replace(",0\n", ",7\n", 1))
    with pytest.raises(dp.PanelValidationError):
        dp.load_panel(synth_registry(("Bad2",)), "Bad2", {"archive": tmp_path}, allow_lockbox=True)


def test_manifest_is_deterministic_and_target_free(synth_panels, tmp_path):
    p = synth_panels["SynthA"]
    ep = dp.build_episodes(p, 2)
    r1 = dp.write_manifest(p, ep, tmp_path / "one", include_inputs=False)
    r2 = dp.write_manifest(p, ep, tmp_path / "two", include_inputs=False)
    assert r1["sha256_uncompressed"] == r2["sha256_uncompressed"]
    assert (tmp_path / "one" / r1["file"]).read_bytes() == (tmp_path / "two" / r2["file"]).read_bytes()
    text = dp.manifest_text(p, ep, include_inputs=True)
    assert "y_support1" in text and "y_query" not in text and "truth" not in text


def test_lockbox_cli_writes_private_files_and_declared_counts(synth_root, tmp_path, monkeypatch, capsys):
    reg = {"panels": {"SynthA": {"role": "lockbox_A", "root": "archive", "format": "ser_csv", "min_groups": 4,
                                  "declared": {"variants": 24, "sites": 12, "drugs": 5, "budgets": [2, 3]}}}}
    regpath = tmp_path / "reg.json"; regpath.write_text(__import__("json").dumps(reg))
    priv = tmp_path / "private"
    dp.main(["--registry", str(regpath), "lockbox", "--archive-root", str(synth_root), "--roles", "lockbox_A",
             "--private-dir", str(priv), "--public-out", str(tmp_path / "counts.json")])
    assert (priv / "SynthA" / "SynthA_k2_manifest.csv.gz").exists()
    assert (priv / "SynthA" / "SynthA_k2_truth.csv.gz").exists()
    counts = __import__("json").loads((tmp_path / "counts.json").read_text())
    assert counts["cohort_counts"]["SynthA"]["episodes"] == {"2": 24 * 5 * 6, "3": 24 * 5 * 4}
    # declared-cohort mismatch aborts
    reg["panels"]["SynthA"]["declared"]["variants"] = 25
    regpath.write_text(__import__("json").dumps(reg))
    with pytest.raises(SystemExit):
        dp.main(["--registry", str(regpath), "lockbox", "--archive-root", str(synth_root), "--roles", "lockbox_A",
                 "--private-dir", str(priv), "--public-out", str(tmp_path / "counts2.json")])


def test_bundle_hash_changes_with_content(synth_root):
    reg = synth_registry(("SynthA",))
    h1 = dp.bundle_hash(reg, {"archive": synth_root}, ("lockbox_A",))["bundle_sha256"]
    (synth_root / "SynthA" / "EDA.json").write_text("{}")
    h2 = dp.bundle_hash(reg, {"archive": synth_root}, ("lockbox_A",))["bundle_sha256"]
    assert h1 != h2


@requires_dev_data
def test_development_panels_load_and_split():
    reg = dp.load_registry()
    expected = {"Estevam2025_MET": (3214, 10), "DuongLy2016_multi_kinase": (76, 183), "Robichaux2021_EGFR": (77, 18),
                "KiRHub2026_multi_kinase": (314, 89), "OKL2026_multi_kinase": (59, 158), "Jung2025_ABL1": (1836, 5)}
    for name, (n_all, d) in expected.items():
        p = dp.load_panel(reg, name, {"archive": dev_root()})
        assert p.n_drugs == d and p.n <= n_all
        assert len(np.unique(p.folds)) >= 4
        for k in p.budgets():
            ep = dp.build_episodes(p, k)
            assert np.isfinite(p.y[ep.i, ep.q]).all()


@requires_dev_data
def test_lockbox_roles_cannot_be_loaded_by_dev_commands():
    reg = dp.load_registry()
    with pytest.raises(SystemExit):
        dp.main(["episodes", "--archive-root", str(dev_root()), "--roles", "lockbox_A"])
    with pytest.raises(dp.LockboxAccessError):
        dp.load_panel(reg, "An2023_EGFR", {"archive": dev_root()})
