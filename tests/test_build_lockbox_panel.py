import json
import math

import pytest

import build_lockbox_panel as bl
import data_pipeline as dp


def spec(**over):
    base = {"study": "SynB", "kinase": "KIN", "source": "https://doi.org/10.0000/test", "native_label": "WT", "unit": "nM",
            "drugs": ["d1", "d2", "d3"], "modes": ["ATP-site", "ATP-site", "allosteric"],
            "rows": [["WT", "10", "20", "40"],
                     ["A100B", "100", "40", "400"],
                     ["C200D", ">3,000", "2,000 ± 150", "80"],
                     ["E300F + G310H", "10", "20", "<4"],
                     ["I400J", "na", "5", "5"],
                     ["K500L", "20", "20", "40"],
                     ["M600N", "10", "10", "10"]]}
    base.update(over)
    return base


def test_parse_literal_variants():
    assert bl.parse_literal("205") == (205.0, "=")
    assert bl.parse_literal(">10,000") == (10000.0, ">")
    assert bl.parse_literal("1,109 ± 453^") == (1109.0, "=")
    assert bl.parse_literal("< 0.2") == (0.2, "<")
    assert bl.parse_literal("na") is None and bl.parse_literal("-") is None
    with pytest.raises(bl.BuildError):
        bl.parse_literal("about 5")


def test_ratio_scale_censoring_and_drops():
    resp, source, dropped = bl.build(spec())
    by = {(r["variant"], r["drug"]): r for r in resp}
    assert by[("A100B", "d1")]["response"] == pytest.approx(1.0)
    assert by[("A100B", "d1")]["censor"] == 0
    assert by[("C200D", "d1")]["response"] == pytest.approx(math.log10(3000 / 10)) and by[("C200D", "d1")]["censor"] == 1
    assert by[("C200D", "d2")]["response"] == pytest.approx(2.0)
    assert by[("E300F+G310H", "d3")]["censor"] == -1 and by[("E300F+G310H", "d3")]["position"] == 300
    assert "I400J" in {n for n, _ in dropped}                      # missing d1 value -> not admitted
    assert not any(r["variant"] == "WT" for r in resp)
    assert len({r["variant"] for r in resp}) == 5


def test_native_must_be_exact():
    bad = spec(rows=[["WT", "10", ">20", "40"], ["A100B", "1", "1", "1"]])
    with pytest.raises(bl.BuildError):
        bl.build(bad)
    with pytest.raises(bl.BuildError):
        bl.build(spec(rows=[["A100B", "1", "1", "1"]]))


def test_unit_conversion_and_exclusions():
    resp, _, dropped = bl.build(spec(unit="uM", exclude={"M600N": "overlap with earlier study"}))
    assert ("M600N", "overlap with earlier study") in dropped
    assert next(r for r in resp if r["variant"] == "A100B" and r["drug"] == "d1")["response"] == pytest.approx(1.0)


def test_written_panel_loads_with_the_pipeline(tmp_path):
    folder = bl.write_panel(spec(rows=[["WT", "10", "20", "40"]] + [[f"A{100 + 10 * i}B", str(10 + i), str(20 + 2 * i), str(40 + 3 * i)] for i in range(10)]), tmp_path)
    assert {p.name for p in folder.iterdir()} >= {"SER_responses.csv", "source_IC50.csv", "study.json", "EDA.json"}
    reg = {"seed": dp.SEED, "panels": {"SynB": {"role": "lockbox_B", "root": "lockbox_b", "format": "ser_csv", "multi_gene": False, "prior": "drug_median"}}}
    panel = dp.load_panel(reg, "SynB", {"lockbox_b": tmp_path}, allow_lockbox=True)
    assert panel.n == 10 and panel.n_drugs == 3 and panel.n_sites == 10
    eda = json.loads((folder / "EDA.json").read_text())
    assert eda["SER_responses.csv"]["profiles"] == 10 and eda["SER_responses.csv"]["matrix_cells"] == 30
    src = (folder / "source_IC50.csv").read_text().splitlines()
    assert src[0] == "construct,drug,reported_ic50,relation,source_literal,in_SER_cohort" and src[1].startswith("WT,d1,10.0,=")


def test_two_native_backgrounds_scale_each_variant_by_its_own_native():
    rows = [["WT@A", "10", "20", "40"], ["WT@B", "100", "200", "400"],
            ["X100Y@A", "100", "200", "400"],        # 10x its native on every drug
            ["X100Y@B", "100", "200", "400"],        # same IC50 as native B: response 0
            ["P200Q@A", "10", "20", "40"]]
    s = spec(rows=rows, native_labels=["WT@A", "WT@B"], native_of={"X100Y@B": "WT@B"})
    resp, _, dropped = bl.build(s)
    by = {(r["variant"], r["drug"]): r["response"] for r in resp}
    assert by[("X100Y@A", "d1")] == pytest.approx(1.0)
    assert by[("X100Y@B", "d1")] == pytest.approx(0.0)
    assert by[("P200Q@A", "d3")] == pytest.approx(0.0)
    assert not dropped
    # both backgrounds share residue 100, so they land on one site
    assert {r["position"] for r in resp if r["variant"].startswith("X100Y")} == {100}


def test_native_of_must_point_to_a_native_row():
    s = spec(native_labels=["WT"], native_of={"A100B": "WT2"})
    with pytest.raises(bl.BuildError):
        bl.build(s)


def test_background_suffix_is_ignored_for_the_position():
    assert bl.variant_position("G2032R@CD74") == 2032
    assert bl.variant_position("S1986F+G2032R@SLC34A2", "last_position") == 2032
    assert bl.variant_position("S1986F+G2032R@SLC34A2", "first_position") == 1986
