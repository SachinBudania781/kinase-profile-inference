"""Structural-similarity audit between candidate Lockbox B panels and the Lockbox A / development panels.

Curator step. It reads response tables only to compute descriptive summaries (sizes, censoring, response
distribution, inter-inhibitor correlation and effective rank). It trains no prediction model and computes no
method error. The public output is a table of coarse per-panel metrics; response values never leave the curator folders.

    python lockbox_similarity.py --archive-root <data> --lockbox-b-root <B> --out LockboxB_Data/_similarity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import data_pipeline as dp

METRICS = ("variants", "sites", "drugs", "variants_per_site", "censored_fraction", "median_response", "iqr_response",
           "p95_response", "fraction_resistant_ratio_ge_1", "mean_interdrug_spearman", "min_interdrug_spearman",
           "effective_rank", "first_pc_share", "drug_sd_ratio")
# Metrics used for the "inside the Lockbox A range" screen. Counts are reported but not screened (A spans 8-370 variants).
SCREENED = ("censored_fraction", "median_response", "iqr_response", "mean_interdrug_spearman", "effective_rank", "first_pc_share")


def panel_metrics(panel: dp.Panel) -> dict:
    y = panel.y
    n, D = y.shape
    finite = np.isfinite(y)
    flat = y[finite]
    c = panel.censor[finite]
    rho = np.full((D, D), np.nan)
    for a in range(D):
        for b in range(a + 1, D):
            ok = finite[:, a] & finite[:, b]
            if ok.sum() >= 5 and np.ptp(y[ok, a]) > 0 and np.ptp(y[ok, b]) > 0:
                rho[a, b] = rho[b, a] = spearmanr(y[ok, a], y[ok, b])[0]
    off = rho[np.triu_indices(D, 1)]
    off = off[np.isfinite(off)]
    complete = finite.all(axis=1)
    z = y[complete]
    if len(z) >= 3 and D >= 2:
        corr = np.corrcoef(np.argsort(np.argsort(z, axis=0), axis=0).T.astype(float))      # Spearman via ranks
        corr = np.nan_to_num(corr, nan=0.0)
        eig = np.clip(np.linalg.eigvalsh(corr), 0, None)
        eff_rank = float(eig.sum() ** 2 / (eig ** 2).sum())
        pc1 = float(eig.max() / eig.sum())
    else:
        eff_rank = pc1 = float("nan")
    sd = np.nanstd(y, axis=0)
    return {"variants": int(n), "sites": int(panel.n_sites), "drugs": int(D), "variants_per_site": float(n / max(panel.n_sites, 1)),
            "censored_fraction": float((c != 0).mean()), "median_response": float(np.median(flat)),
            "iqr_response": float(np.percentile(flat, 75) - np.percentile(flat, 25)), "p95_response": float(np.percentile(flat, 95)),
            "fraction_resistant_ratio_ge_1": float((flat >= 1.0).mean()),
            "mean_interdrug_spearman": float(off.mean()) if len(off) else float("nan"),
            "min_interdrug_spearman": float(off.min()) if len(off) else float("nan"),
            "effective_rank": eff_rank, "first_pc_share": pc1,
            "drug_sd_ratio": float(sd.max() / max(sd.min(), 1e-9)), "scale_kind": panel.scale_kind,
            "genes": int(len({str(g) for g in panel.genes}))}


def compare_to_reference(candidate: dict, reference: dict[str, dict], slack: float = 0.25) -> dict:
    """Is each screened metric inside the reference range, widened by ``slack`` of the range on both sides?"""
    out = {}
    for m in SCREENED:
        vals = [r[m] for r in reference.values() if np.isfinite(r[m])]
        lo, hi = min(vals), max(vals)
        pad = slack * (hi - lo)
        v = candidate[m]
        out[m] = {"value": v, "reference_min": lo, "reference_max": hi, "inside": bool(np.isfinite(v) and lo - pad <= v <= hi + pad)}
    out["all_inside"] = bool(all(out[m]["inside"] for m in SCREENED))
    return out


def audit(registry: dict, roots: dict, candidate_names, reference_roles=("lockbox_A",), context_names=()):
    ref = {n: panel_metrics(dp.load_panel(registry, n, roots, allow_lockbox=True))
           for n in dp.panels_by_role(registry, reference_roles) if registry["panels"][n]["root"] in roots}
    cand = {n: panel_metrics(dp.load_panel(registry, n, roots, allow_lockbox=True)) for n in candidate_names}
    ctx = {n: panel_metrics(dp.load_panel(registry, n, roots, allow_lockbox=False)) for n in context_names}
    return {"reference": ref, "candidates": cand, "context": ctx,
            "comparison": {n: compare_to_reference(m, ref) for n, m in cand.items()} if ref else {}}


def markdown_table(res: dict) -> str:
    cols = ["variants", "sites", "drugs", "censored_fraction", "median_response", "iqr_response", "mean_interdrug_spearman",
            "effective_rank", "first_pc_share"]
    head = "| Panel | Group | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 2) + "\n"
    rows = []
    for group in ("reference", "candidates", "context"):
        for n, m in res[group].items():
            rows.append(f"| {n} | {group} | " + " | ".join(f"{m[c]:.2f}" if isinstance(m[c], float) else str(m[c]) for c in cols) + " |")
    return head + "\n".join(rows) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive-root", required=True)
    p.add_argument("--lockbox-b-root", required=True)
    p.add_argument("--registry", default=None)
    p.add_argument("--out", default="LockboxB_Data/_similarity")
    args = p.parse_args(argv)
    registry = dp.load_registry(args.registry)
    roots = {"archive": args.archive_root, "lockbox_b": args.lockbox_b_root}
    cand = [n for n in dp.panels_by_role(registry, ("lockbox_B",)) if (Path(args.lockbox_b_root) / n).is_dir()]
    ctx = [n for n in ("Robichaux2021_EGFR", "Estevam2025_MET", "Nakamura2021_FGFR1_FGFR2_FGFR3") if n in registry["panels"]]
    res = audit(registry, roots, cand, context_names=ctx)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "similarity.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / "similarity_table.md").write_text(markdown_table(res), encoding="utf-8")
    print(markdown_table(res))
    for n, c in res["comparison"].items():
        print(n, "inside Lockbox A range on all screened metrics:", c["all_inside"],
              [m for m in SCREENED if not c[m]["inside"]])


if __name__ == "__main__":
    main()
