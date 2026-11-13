"""Parity of XGBoost, CovarianceLocalMedian and SER with the released ``ser`` package (synthetic panels only).

Set KPI_REFERENCE_SER to the folder that contains the released 'ser' package.
"""
import os
import sys

import numpy as np
import pytest

import core_models as cm
import data_pipeline as dp
from conftest import synth_registry, write_ser_panel

REF = os.environ.get("KPI_REFERENCE_SER")
pytestmark = pytest.mark.skipif(not REF, reason="set KPI_REFERENCE_SER to compare with the released package")


@pytest.mark.parametrize("censor_frac,k", [(0.0, 2), (0.0, 3), (0.25, 2)])
def test_matches_released_reference(tmp_path, censor_frac, k):
    sys.path.insert(0, REF)
    from ser import evaluation, models as ref_models
    from ser.data import load_study
    write_ser_panel(tmp_path, "P", n_sites=16, per_site=2, drugs=("a", "b", "c", "d", "e"), seed=21, censor_frac=censor_frac)
    panel = dp.load_panel(synth_registry(("P",)), "P", {"archive": tmp_path}, allow_lockbox=True)
    study = load_study(tmp_path / "P")
    ep = dp.build_episodes(panel, k)
    out = cm.run_baselines(panel, ep, cm.load_config())
    ref, dec = evaluation.evaluate(study, k)
    np.testing.assert_array_equal(ref["i"], ep.i)
    np.testing.assert_allclose(out.xgb, ref["XGBoost"], atol=1e-9)
    np.testing.assert_allclose(out.ser, ref["SER"], atol=1e-9)
    assert [(d["fold"], d["expert"], d["weight"]) for d in out.decisions] == \
           [(d["fold"], d["expert"], d["weight"]) for d in dec["SER_gate"]]
    # CovarianceLocalMedian: compare with the released neighbourhood fit using the same selected size per outer fold
    for rec in out.selected:
        outer = rec["outer_fold"]
        test = np.flatnonzero(panel.folds[ep.i] == outer)
        train = np.flatnonzero(panel.folds != outer)
        pred = ref_models.neighborhood_fit(study.y, train, ep.i[test], ep.q[test], ep.s[test], ("LocalCandidate",))
        j = list(cm.DEFAULT_CONFIG["neighbors"]).index(rec["neighbors"])
        np.testing.assert_allclose(out.local[test], pred["LocalCandidate"][:, j], atol=1e-9)
