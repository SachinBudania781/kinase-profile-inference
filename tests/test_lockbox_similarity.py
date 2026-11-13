import json

import numpy as np
import pytest

import data_pipeline as dp
import lockbox_similarity as ls
from conftest import write_ser_panel


def reg(names_roles):
    return {"seed": dp.SEED, "panels": {n: {"role": r, "root": "archive" if r == "lockbox_A" else "lockbox_b", "format": "ser_csv",
                                              "multi_gene": False, "prior": "drug_median"} for n, r in names_roles.items()}}


def test_metrics_basic_values(tmp_path):
    write_ser_panel(tmp_path, "P", n_sites=12, per_site=2, drugs=("a", "b", "c", "d"), seed=1, censor_frac=0.2)
    panel = dp.load_panel(reg({"P": "lockbox_A"}), "P", {"archive": tmp_path}, allow_lockbox=True)
    m = ls.panel_metrics(panel)
    assert m["variants"] == 24 and m["sites"] == 12 and m["drugs"] == 4 and m["variants_per_site"] == pytest.approx(2.0)
    assert 0 < m["censored_fraction"] < 0.5
    assert 1.0 <= m["effective_rank"] <= 4.0 and 0.25 <= m["first_pc_share"] <= 1.0
    assert m["min_interdrug_spearman"] <= m["mean_interdrug_spearman"]


def test_similarity_flags_an_outlier(tmp_path):
    roots = {"archive": tmp_path / "A", "lockbox_b": tmp_path / "B"}
    for i in range(3):
        write_ser_panel(roots["archive"], f"A{i}", n_sites=14, per_site=2, drugs=("a", "b", "c", "d"), seed=10 + i, rank=1)
    write_ser_panel(roots["lockbox_b"], "Bgood", n_sites=14, per_site=2, drugs=("a", "b", "c", "d"), seed=11, rank=1)  # same generator as A1: inside range by construction
    write_ser_panel(roots["lockbox_b"], "Bheavy", n_sites=14, per_site=2, drugs=("a", "b", "c", "d"), seed=21, rank=1, censor_frac=0.9)
    r = reg({"A0": "lockbox_A", "A1": "lockbox_A", "A2": "lockbox_A", "Bgood": "lockbox_B", "Bheavy": "lockbox_B"})
    res = ls.audit(r, roots, ["Bgood", "Bheavy"])
    assert res["comparison"]["Bgood"]["all_inside"] is True
    assert res["comparison"]["Bheavy"]["censored_fraction"]["inside"] is False and res["comparison"]["Bheavy"]["all_inside"] is False
    text = ls.markdown_table(res)
    assert "Bgood" in text and "| A0 | reference" in text


def test_cli_writes_outputs(tmp_path, capsys):
    roots = {"archive": tmp_path / "A", "lockbox_b": tmp_path / "B"}
    for i in range(2):
        write_ser_panel(roots["archive"], f"A{i}", n_sites=12, per_site=2, drugs=("a", "b", "c"), seed=1 + i)
    write_ser_panel(roots["lockbox_b"], "B0", n_sites=12, per_site=2, drugs=("a", "b", "c"), seed=9)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(reg({"A0": "lockbox_A", "A1": "lockbox_A", "B0": "lockbox_B"})))
    ls.main(["--archive-root", str(roots["archive"]), "--lockbox-b-root", str(roots["lockbox_b"]), "--registry", str(path), "--out", str(tmp_path / "o")])
    assert (tmp_path / "o" / "similarity.json").exists() and (tmp_path / "o" / "similarity_table.md").exists()
