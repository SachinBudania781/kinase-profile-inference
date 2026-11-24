"""Replay SER gate variants on stored records (no refitting). Development tool only.

    python dev_tools/replay_ser.py --records dev_records/v1 --repo . [--variants all]
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
import numpy as np

VARIANTS = {
    "registered":        {"ser_censor_mode": "registered",   "ser_selection": "max_lower_bound"},
    "nodefer_bi":        {"ser_censor_mode": "no_defer_bi",  "ser_selection": "max_lower_bound"},
    "nodefer_all3":      {"ser_censor_mode": "no_defer_all3", "ser_selection": "max_lower_bound"},
    "registered+meansafe": {"ser_censor_mode": "registered", "ser_selection": "max_mean_safe"},
    "nodefer_bi+meansafe": {"ser_censor_mode": "no_defer_bi", "ser_selection": "max_mean_safe"},
}


def interval_err(pred, truth, cens):
    r = pred - truth
    return np.where(cens == 1, np.maximum(-r, 0), np.where(cens == -1, np.maximum(r, 0), np.abs(r)))


def replay(rec, variant, cm, cfg):
    sers = np.full(rec["episodes"], np.nan)
    choices = []
    for outer, r in rec["records"].items():
        v, t = r["val"], r["test"]
        ch = cm.ser_gate(v["truth"], v["anchor"], v["alts"], v["sites"], cfg["ser"]["weights"], cfg["ser"]["confidence"],
                         cfg["ser"]["min_validation_sites"], v["obs_limit"], v["query_censor"], r["censored_panel"],
                         variant["ser_censor_mode"], variant["ser_selection"])
        sers[t["idx"]] = cm.ser_apply(ch, t["xgb"], t["alts"].get(ch["expert"]), t["obs_limit"], r["censored_panel"],
                                      variant["ser_censor_mode"])
        choices.append(ch)
    return sers, choices


def site_boot(sites, ex, es, draws=300, seed=1):
    u, inv = np.unique(sites, return_inverse=True)
    sx, ss = np.bincount(inv, ex), np.bincount(inv, es)
    idx = np.random.default_rng(seed).integers(0, len(u), (draws, len(u)))
    g = 100 * (1 - ss[idx].sum(1) / np.maximum(sx[idx].sum(1), 1e-300))
    return float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True); ap.add_argument("--repo", required=True)
    ap.add_argument("--variants", default="all"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    import core_models as cm
    cfg = cm.load_config(str(Path(a.repo) / "configs" / "models.json"))
    names = list(VARIANTS) if a.variants == "all" else a.variants.split(",")
    rows = []
    for f in sorted(Path(a.records).glob("*.pkl")):
        rec = pickle.load(open(f, "rb"))
        truth, cq, xg = rec["truth"], rec["query_censor"], rec["preds"]["XGBoost"]
        ok0 = np.isfinite(xg)
        row = {"target": f"{rec['name']}:{rec['k']}", "censored": rec["censored_panel"], "D": rec["n_drugs"], "n": rec["n"],
               "local_gain": 100 * (1 - np.nanmean(np.abs(rec["preds"]["CovarianceLocalMedian"] - truth)[ok0]) / np.mean(np.abs(xg - truth)[ok0])),
               "ridge_gain": 100 * (1 - np.nanmean(np.abs(rec["preds"]["RidgeSupports"] - truth)[ok0]) / np.mean(np.abs(xg - truth)[ok0]))}
        for vn in names:
            sers, ch = replay(rec, VARIANTS[vn], cm, cfg)
            ok = np.isfinite(sers) & np.isfinite(xg)
            ex, es = np.abs(xg - truth)[ok], np.abs(sers - truth)[ok]
            g = 100 * (1 - es.sum() / ex.sum())
            lo, hi = site_boot(rec["sites"][ok], ex, es)
            ix, is_ = interval_err(xg, truth, cq)[ok], interval_err(sers, truth, cq)[ok]
            row[vn] = {"gain": g, "ci": (lo, hi), "interval_gain": 100 * (1 - is_.sum() / max(ix.sum(), 1e-300)),
                       "intervened": float(np.mean(sers[ok] != xg[ok]))}
        rows.append(row)
    if a.out:
        import json
        Path(a.out).write_text(json.dumps(rows, indent=1, default=float))
    print(f"{'target':36s} cens  local  ridge | " + " | ".join(f"{vn[:14]:>14s}" for vn in names))
    for r in rows:
        print(f"{r['target']:36s} {'Y' if r['censored'] else '-':4s} {r['local_gain']:6.1f} {r['ridge_gain']:6.1f} | " +
              " | ".join(f"{r[vn]['gain']:6.2f} ({r[vn]['intervened']:.2f})" for vn in names))
    print("\nmean gain over targets:", {vn: round(float(np.mean([r[vn]['gain'] for r in rows])), 2) for vn in names})
    print("harm (CI hi < 0):", {vn: sum(r[vn]['ci'][1] < 0 for r in rows) for vn in names})
    print("studies with negative gain:", {vn: sum(r[vn]['gain'] < 0 for r in rows) for vn in names})
    print("worst study gain:", {vn: round(min(r[vn]['gain'] for r in rows), 2) for vn in names})
    print("\ninterval-error gain (sensitivity analysis), mean over targets / worst target:")
    print({vn: (round(float(np.mean([r[vn]['interval_gain'] for r in rows])), 2),
                round(min(r[vn]['interval_gain'] for r in rows), 2)) for vn in names})


if __name__ == "__main__":
    main()
