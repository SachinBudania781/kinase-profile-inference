import json
import shutil
from pathlib import Path

import pytest

import data_pipeline as dp
import integrity as ig
from conftest import synth_registry, write_ser_panel


@pytest.fixture
def mini_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "a.py").write_text("x = 1\n")
    (repo / "tests" / "test_a.py").write_text("def test_a():\n    pass\n")
    (repo / "configs" / "models.json").write_text('{"eta": 0.055}\n')
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "a.cpython.pyc").write_bytes(b"junk")
    (repo / "lockbox_private").mkdir()
    (repo / "lockbox_private" / "secret.py").write_text("y = 2\n")
    return repo


@pytest.fixture
def registry_file(tmp_path):
    path = tmp_path / "reg.json"
    reg = synth_registry(("S1", "S2"))
    reg["panels"]["S1"]["role"] = "core_dev"
    reg["panels"]["S2"]["role"] = "lockbox_B"
    reg["panels"]["S2"]["root"] = "lockbox_b"
    path.write_text(json.dumps(reg))
    return path


def test_code_hash_ignores_caches_and_private_folders(mini_repo):
    h1 = ig.code_hash(mini_repo)
    assert set(h1["files"]) == {"a.py", "tests/test_a.py", "configs/models.json"}
    (mini_repo / "__pycache__" / "b.pyc").write_bytes(b"more")
    (mini_repo / "lockbox_private" / "other.py").write_text("z = 3\n")
    assert ig.code_hash(mini_repo)["bundle_sha256"] == h1["bundle_sha256"]
    (mini_repo / "a.py").write_text("x = 2\n")
    assert ig.code_hash(mini_repo)["bundle_sha256"] != h1["bundle_sha256"]


def test_record_verify_roundtrip_and_tamper_detection(mini_repo, tmp_path, registry_file):
    archive, lock_b = tmp_path / "archive", tmp_path / "lockB"
    write_ser_panel(archive, "S1", n_sites=6, drugs=("a", "b", "c"), seed=1)
    write_ser_panel(lock_b, "S2", n_sites=6, drugs=("a", "b", "c"), seed=2)
    rec = ig.compute(mini_repo, archive, lock_b, registry_file)
    assert rec["development_data"]["bundle_sha256"] and rec["lockbox_B"]["bundle_sha256"]
    assert rec["lockbox_A"]["bundle_sha256"] is None
    assert ig.compare(rec, ig.compute(mini_repo, archive, lock_b, registry_file)) == []
    # code change
    (mini_repo / "a.py").write_text("x = 99\n")
    assert any("a.py" in p for p in ig.compare(rec, ig.compute(mini_repo, archive, lock_b, registry_file)))
    (mini_repo / "a.py").write_text("x = 1\n")
    # config change
    (mini_repo / "configs" / "models.json").write_text('{"eta": 0.1}\n')
    assert any("configuration" in p for p in ig.compare(rec, ig.compute(mini_repo, archive, lock_b, registry_file)))
    (mini_repo / "configs" / "models.json").write_text('{"eta": 0.055}\n')
    # lockbox data change
    f = lock_b / "S2" / "SER_responses.csv"
    f.write_text(f.read_text() + "X999Z,KIN,999,a,0.0,0\n")
    assert any("lockbox_B" in p and "S2" in p for p in ig.compare(rec, ig.compute(mini_repo, archive, lock_b, registry_file)))


def test_missing_lockbox_root_is_reported_not_silently_passed(mini_repo, tmp_path, registry_file):
    archive, lock_b = tmp_path / "archive", tmp_path / "lockB"
    write_ser_panel(archive, "S1", n_sites=6, drugs=("a", "b", "c"), seed=1)
    write_ser_panel(lock_b, "S2", n_sites=6, drugs=("a", "b", "c"), seed=2)
    rec = ig.compute(mini_repo, archive, lock_b, registry_file)
    probs = ig.compare(rec, ig.compute(mini_repo, archive, None, registry_file))
    assert any("lockbox_B" in p and "not provided" in p for p in probs)


def test_cli_record_and_verify(mini_repo, tmp_path, registry_file, capsys):
    archive = tmp_path / "archive"
    write_ser_panel(archive, "S1", n_sites=6, drugs=("a", "b", "c"), seed=1)
    out = tmp_path / "rec.json"
    ig.main(["--registry", str(registry_file), "record", "--repo", str(mini_repo), "--archive-root", str(archive), "--out", str(out)])
    ig.main(["--registry", str(registry_file), "verify", "--repo", str(mini_repo), "--archive-root", str(archive), "--record", str(out)])
    assert "passed" in capsys.readouterr().out
    (mini_repo / "a.py").write_text("tampered = True\n")
    with pytest.raises(SystemExit) as e:
        ig.main(["--registry", str(registry_file), "verify", "--repo", str(mini_repo), "--archive-root", str(archive), "--record", str(out)])
    assert e.value.code == 1


def test_hash_dir_is_order_independent(tmp_path):
    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    for d, order in ((d1, ("a", "b")), (d2, ("b", "a"))):
        d.mkdir()
        for n in order:
            (d / f"{n}.csv").write_text(n)
    assert ig.directory_hash(d1)["bundle_sha256"] == ig.directory_hash(d2)["bundle_sha256"]
