"""Shared fixtures. Synthetic panels only; no lockbox response tables are ever read by the tests."""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_pipeline as dp  # noqa: E402


def write_ser_panel(root: Path, name: str, n_sites=12, per_site=2, drugs=("a", "b", "c", "d", "e"), seed=0,
                    censor_frac=0.0, role="lockbox_A", rank=2):
    """Write a lockbox-format synthetic panel (low-rank signal + noise) and return its directory."""
    rng = np.random.default_rng(seed)
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    n = n_sites * per_site
    load = rng.normal(size=(len(drugs), rank))
    scores = rng.normal(size=(n, rank))
    y = scores @ load.T + 0.3 * rng.normal(size=(n, len(drugs)))
    cens = np.zeros(y.shape, dtype=int)
    if censor_frac:
        cens = rng.choice([-1, 0, 1], size=y.shape, p=[censor_frac / 2, 1 - censor_frac, censor_frac / 2])
    modes = ["ATP-site" if j % 2 == 0 else "allosteric" for j in range(len(drugs))]
    with (d / "SER_responses.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["variant", "kinase", "position", "drug", "response", "censor"])
        for i in range(n):
            site = 100 + i // per_site
            for j, drug in enumerate(drugs):
                w.writerow([f"X{site}{'ACDEFGHIKLMNPQRSTVWY'[i % per_site]}", "KIN", site, drug, repr(float(y[i, j])), int(cens[i, j])])
    (d / "study.json").write_text(json.dumps({
        "study": name, "kinase": "KIN",
        "ser": {"table": "SER_responses.csv", "drugs": list(drugs), "modes": modes,
                "response": "log10 mutant/native IC50", "censor": {"-1": "upper_bound", "0": "exact", "1": "lower_bound"}}}))
    return d


@pytest.fixture
def registry():
    return dp.load_registry()


@pytest.fixture
def synth_root(tmp_path):
    write_ser_panel(tmp_path, "SynthA", n_sites=12, per_site=2, drugs=("a", "b", "c", "d", "e"), seed=1)
    write_ser_panel(tmp_path, "SynthCens", n_sites=14, per_site=2, drugs=("a", "b", "c", "d"), seed=2, censor_frac=0.3)
    write_ser_panel(tmp_path, "Synth3", n_sites=10, per_site=3, drugs=("a", "b", "c"), seed=3)
    return tmp_path


def synth_registry(names=("SynthA", "SynthCens", "Synth3")):
    return {"seed": dp.SEED, "panels": {n: {"role": "lockbox_A", "root": "archive", "format": "ser_csv",
                                              "multi_gene": False, "prior": "drug_median"} for n in names}}


@pytest.fixture
def synth_panels(synth_root):
    reg = synth_registry()
    return {n: dp.load_panel(reg, n, {"archive": synth_root}, allow_lockbox=True) for n in reg["panels"]}


def dev_root():
    path = os.environ.get("KPI_DATA_ROOT")
    return Path(path) if path and Path(path).is_dir() else None


requires_dev_data = pytest.mark.skipif(dev_root() is None, reason="set KPI_DATA_ROOT to the cleaned archive")
