"""Synthetic stress-test bank for the Kinase Profile Inference study (development tool only).

Artificial studies only. No lockbox information is used. See SYNTHETIC_STRESS_SPEC.md.

Usage:
    python synth_bank.py --out synth_data --n 400 [--seed 20261114]

Writes <out>/data/<study>/ (matrix_npz or ser_csv format), <out>/registry.json (same schema as
configs/panels.json, role "stress_test", root "synth") and <out>/bank_manifest.json (all drawn parameters).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

BANK_SEED = 20261114
D_CHOICES = [3, 4, 5, 6, 8, 10, 14, 18, 30, 60, 120]
D_PROBS = [.10, .08, .10, .08, .10, .10, .10, .10, .10, .07, .07]
N_CHOICES = [40, 60, 100, 200, 400, 800, 1600, 3000]
SCALE_KINDS = ["other", "ic50_log10_ratio", "enrichment", "remaining_activity", "fitness_score"]
EPISODE_TARGET = 8000          # approximate episodes per (study, k): keeps run time affordable


def sample_params(rng: np.random.Generator, idx: int) -> dict:
    D = int(rng.choice(D_CHOICES, p=D_PROBS))
    n = int(rng.choice(N_CHOICES))
    vps = float(rng.choice([1, 2, 4, 8, 16], p=[.15, .2, .3, .2, .15]))       # variants per site
    G = int(np.clip(round(n / vps), 4, 400))
    fmt = "ser_csv" if rng.random() < 0.3 else "matrix_npz"
    p = {
        "idx": idx, "D": D, "n": n, "G": G, "format": fmt,
        "rank": int(rng.choice([1, 2, 3, 5], p=[.3, .3, .25, .15])),
        "families": int(np.clip(D // 3, 1, 6)),
        "common_loading": float(rng.uniform(0.2, 0.95)),
        "p_negative_drug": float(rng.choice([0.0, 0.0, 0.05, 0.1])),
        "site_var_frac": float(rng.choice([0.0, 0.1, 0.3, 0.5, 0.8])),
        "p_resistant": float(rng.choice([0.05, 0.15, 0.3, 0.5])),
        "resistant_shift": float(rng.uniform(1.5, 4.0)),
        "p_saturating": float(rng.choice([0.0, 0.3, 0.7, 1.0])),
        "noise_mult": float(rng.choice([0.3, 0.6, 1.0, 1.6])),
        "heteroscedastic": float(rng.choice([0.0, 0.0, 0.3, 0.8])),
        "p_heavy_tail": float(rng.choice([0.0, 0.0, 0.3])),
        "p_outlier": float(rng.choice([0.0, 0.0, 0.01, 0.04])),
        "batch_frac": float(rng.choice([0.0, 0.15, 0.4])),
        "study_scale": float(10 ** rng.uniform(-1.5, 1.5)),
        "multi_gene": bool(rng.random() < 0.3 and D >= 10),
        "scale_kind": str(rng.choice(SCALE_KINDS)),
        "fill": 1.0 if fmt == "ser_csv" else float(rng.choice([1.0, 1.0, 0.9, 0.7, 0.5])),
        "missing_mnar": bool(rng.random() < 0.4),
        "censor_frac": 0.0 if fmt == "matrix_npz" else float(rng.choice([0.0, 0.05, 0.15, 0.35, 0.6])),
        "n_modes": int(rng.choice([1, 2, 3])) if fmt == "ser_csv" else 1,
        "adversarial": {k: bool(rng.random() < 0.08) for k in
                        ["duplicate_drug", "constant_drug", "extreme_scale_drug", "pure_noise_drug",
                         "context_flip_drug", "sparse_drug"]},
    }
    if D < 6:                                           # adversarial columns need room to stay meaningful
        for k in ("duplicate_drug", "constant_drug", "pure_noise_drug", "sparse_drug"):
            p["adversarial"][k] = False
    if fmt == "ser_csv":                                # the CSV contract requires complete profiles
        p["adversarial"]["sparse_drug"] = False
    masks_k2 = D * math.comb(D - 1, 2)
    cap_target = max(4, EPISODE_TARGET // n)
    p["max_masks"] = None if masks_k2 <= cap_target else int(cap_target)
    p["name"] = f"Syn{idx:04d}_D{D}_n{n}_{fmt[:3]}"
    return p


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate(p: dict, rng: np.random.Generator) -> dict:
    D, n, G, K = p["D"], p["n"], p["G"], p["rank"]
    site_of = np.concatenate([np.arange(G), rng.integers(0, G, max(n - G, 0))])[:n]
    rng.shuffle(site_of)
    positions = np.sort(rng.choice(np.arange(1, 2000), size=G, replace=False))
    sf = p["site_var_frac"]
    z = rng.normal(0, math.sqrt(sf), (G, K))[site_of] + rng.normal(0, math.sqrt(1 - sf), (n, K))
    res = rng.random(n) < p["p_resistant"]
    z[res, 0] += rng.normal(p["resistant_shift"], 0.5, res.sum())
    fam = rng.integers(0, p["families"], D)
    fam_load = rng.normal(0, 1, (p["families"], K))
    L = np.zeros((D, K))
    L[:, 0] = p["common_loading"] * (1 + 0.3 * rng.normal(size=D))
    for f in range(1, K):
        L[:, f] = fam_load[fam, f] * 0.8 + 0.3 * rng.normal(size=D)
    neg = rng.random(D) < p["p_negative_drug"]
    L[neg] *= -1
    L /= np.maximum(np.linalg.norm(L, axis=1, keepdims=True), 1e-9)
    signal = z @ L.T
    sd_sig = signal.std(axis=0) + 1e-9
    sigma = np.exp(rng.uniform(math.log(0.1), math.log(1.0), D)) * p["noise_mult"] * sd_sig
    y = signal.copy()
    # adversarial structure on the signal
    adv = p["adversarial"]
    cols = {}
    free = list(rng.permutation(D))
    def take():
        return free.pop() if free else None
    if adv["pure_noise_drug"]:
        j = take()
        if j is not None:
            y[:, j] = rng.normal(0, 1, n); sigma[j] = 0.0; cols["pure_noise"] = j
    if adv["context_flip_drug"]:
        j = take()
        if j is not None:
            flip = np.where((positions[site_of] % 2) == 0, 1.0, -1.0)
            y[:, j] = signal[:, j] * flip; cols["context_flip"] = j
    # saturating link (applied to a random subset of drugs)
    sat = rng.random(D) < p["p_saturating"]
    for j in np.flatnonzero(sat):
        a, b = rng.uniform(0.8, 2.5), rng.uniform(-0.5, 1.0)
        y[:, j] = 4.0 * _sigmoid(a * (y[:, j] - b)) - 2.0
    # measurement noise
    het = 1.0 + p["heteroscedastic"] * np.abs(y)
    noise = rng.normal(0, 1, (n, D))
    heavy = rng.random(D) < p["p_heavy_tail"]
    if heavy.any():
        noise[:, heavy] = rng.standard_t(3, (n, int(heavy.sum()))) / math.sqrt(3)
    y = y + noise * sigma[None, :] * het
    if p["batch_frac"] > 0:
        y = y + rng.normal(0, 1, (G, D))[site_of] * (p["batch_frac"] * sd_sig)[None, :]
    if p["p_outlier"] > 0:
        out = rng.random((n, D)) < p["p_outlier"]
        y = y + out * rng.choice([-1, 1], (n, D)) * rng.uniform(3, 8, (n, D)) * sd_sig[None, :]
    # drug scale / offset, study scale
    y = y * np.exp(rng.normal(0, 0.4, D))[None, :] + rng.normal(0, 2, D)[None, :]
    if adv["extreme_scale_drug"]:
        j = take()
        if j is not None:
            y[:, j] *= 500.0; cols["extreme_scale"] = j
    if adv["constant_drug"]:
        j = take()
        if j is not None:
            y[:, j] = np.full(n, y[:, j].mean()) + rng.normal(0, 1e-6, n); cols["constant"] = j
    if adv["duplicate_drug"]:
        j = take()
        if j is not None:
            src = int(rng.integers(0, D))
            if src != j:
                y[:, j] = y[:, src] + rng.normal(0, 1e-3 * (y[:, src].std() + 1e-9), n); cols["duplicate_of"] = [j, src]
    y = y * p["study_scale"]
    # missing cells (matrix format only)
    miss = np.zeros((n, D), bool)
    if p["fill"] < 1.0:
        base = 1.0 - p["fill"]
        if p["missing_mnar"]:
            r = (y - y.mean(0)) / (y.std(0) + 1e-9)
            pm = np.clip(base * 2.0 * _sigmoid(r), 0, 0.95)
        else:
            pm = np.full((n, D), base)
        miss = rng.random((n, D)) < pm
    if adv["sparse_drug"]:
        j = take()
        if j is not None:
            miss[:, j] |= rng.random(n) < 0.7; cols["sparse"] = j
    # every profile keeps at least 3 observed cells
    for i in np.flatnonzero((~miss).sum(1) < 3):
        keep = rng.choice(D, size=min(3, D), replace=False)
        miss[i, keep] = False
    # censoring (CSV format only; complete profiles)
    censor = np.zeros((n, D), np.int8)
    if p["censor_frac"] > 0:
        f = p["censor_frac"]
        up = np.quantile(y, 1 - 0.7 * f, axis=0)
        lo = np.quantile(y, 0.3 * f, axis=0)
        hi_mask, lo_mask = y > up[None, :], y < lo[None, :]
        y = np.where(hi_mask, up[None, :], np.where(lo_mask, lo[None, :], y))
        censor[hi_mask] = 1
        censor[lo_mask] = -1
    y_out = np.where(miss, np.nan, y)
    labels = [f"S{positions[site_of[i]]}v{i:05d}" for i in range(n)]
    fam_names = np.array([f"fam{f}" for f in fam])
    drugs = [f"D{j:03d}" for j in range(D)]
    modes = [f"mode{int(f) % p['n_modes']}" for f in fam]
    genes = (np.array([f"G{int(s) % max(2, D // 10)}" for s in site_of]) if p["multi_gene"]
             else np.array(["GENE"] * n))
    return {"Y": y_out, "censor": censor, "labels": labels, "genes": genes, "positions": positions[site_of],
            "drugs": drugs, "modes": modes, "planted": cols, "signal_sd": float(signal.std())}


def write_study(root: Path, p: dict, g: dict) -> None:
    d = root / "data" / p["name"]
    d.mkdir(parents=True, exist_ok=True)
    if p["format"] == "matrix_npz":
        np.savez_compressed(d / "response_matrix.npz", Y=g["Y"], labels=np.array(g["labels"]), genes=g["genes"],
                            groups=g["positions"].astype(int), drugs=np.array(g["drugs"]), name=np.array(p["name"]))
        (d / "study.json").write_text(json.dumps({"study": p["name"], "kinase": "SYNTHETIC", "source": "synthetic"}))
    else:
        resp = {"ic50_log10_ratio": "ic50 log10 ratio", "enrichment": "enrichment"}.get(p["scale_kind"], "other")
        with (d / "SER_responses.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["variant", "drug", "response", "censor", "position", "kinase"])
            for i, lab in enumerate(g["labels"]):
                for j, dn in enumerate(g["drugs"]):
                    w.writerow([lab, dn, repr(float(g["Y"][i, j])), int(g["censor"][i, j]), int(g["positions"][i]), g["genes"][i]])
        (d / "study.json").write_text(json.dumps({"study": p["name"], "kinase": "SYNTHETIC", "source": "synthetic",
                                                  "ser": {"drugs": g["drugs"], "modes": g["modes"], "table": "SER_responses.csv",
                                                          "response": resp}}))


def build_bank(out: Path, n_studies: int, seed: int = BANK_SEED) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    panels, manifest = {}, []
    for idx in range(n_studies):
        rng = np.random.default_rng([seed, idx])
        p = sample_params(rng, idx)
        g = generate(p, rng)
        write_study(out, p, g)
        spec = {"role": "stress_test", "root": "synth", "format": p["format"], "multi_gene": p["multi_gene"],
                "prior": "drug_median", "scale_kind": p["scale_kind"],
                "max_masks_per_profile": {"2": p["max_masks"], "3": p["max_masks"]}, "min_groups": 4}
        panels[p["name"]] = spec
        manifest.append({**p, "planted_columns": g["planted"]})
    reg = {"schema_version": "1.0", "note": "SYNTHETIC stress-test bank (artificial data; dev tool only).",
           "seed": 20261112, "roots": {"synth": "synthetic bank data folder"}, "panels": panels}
    (out / "registry.json").write_text(json.dumps(reg, indent=1))
    man = {"bank_seed": seed, "n_studies": n_studies, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "studies": manifest}
    (out / "bank_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=BANK_SEED)
    a = ap.parse_args()
    m = build_bank(Path(a.out), a.n, a.seed)
    print(f"wrote {m['n_studies']} synthetic studies to {a.out}; generator sha256 {m['generator_sha256'][:12]}")
