"""Run the baseline engine (XGBoost, SER, CovarianceLocalMedian, Ridge, QueryMedian) on the synthetic bank.

    python run_synth.py --bank synth_data --repo <repo folder> --out synth_results.jsonl --jobs 4
    python run_synth.py ... --only Syn0001,Syn0002 --scale-check

Development tool only: artificial data, never a lockbox panel. Resumable (skips finished study/k pairs).
"""
from __future__ import annotations
import argparse, json, sys, time, traceback, dataclasses
from multiprocessing import Pool
from pathlib import Path
import numpy as np

METHODS = ("XGBoost", "SER", "CovarianceLocalMedian", "QueryMedian", "RidgeSupports")


def boot_gain(sites, err_x, err_s, draws=400, seed=1):
    u, inv = np.unique(sites, return_inverse=True)
    sx, ss = np.bincount(inv, err_x), np.bincount(inv, err_s)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), (draws, len(u)))
    g = 100 * (1 - ss[idx].sum(1) / np.maximum(sx[idx].sum(1), 1e-300))
    return float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))


def one(job):
    name, k, bank, repo, scale = job
    sys.path.insert(0, repo)
    import core_models as cm, data_pipeline as dp
    t0 = time.time()
    row = {"study": name, "k": k, "scale": scale}
    try:
        reg = dp.load_registry(Path(bank) / "registry.json")
        P = dp.load_panel(reg, name, {"synth": str(Path(bank) / "data")}, allow_lockbox=False)
        if scale != 1.0:
            P = dataclasses.replace(P, y=P.y * scale)
        if k not in P.budgets():
            row["skipped"] = "budget not allowed"; return row
        cfg = cm.load_config()
        ep = dp.build_episodes(P, k)
        res = cm.run_panel(P, k, cfg, METHODS, pool=None, ep=ep)
        truth = P.y[ep.i, ep.q]
        cq = P.censor[ep.i, ep.q]
        sites = np.array([dp.site_key(s) for s in P.sites], dtype=object)[ep.i]
        row.update({"episodes": int(len(ep)), "D": int(P.n_drugs), "n": int(P.n), "sites": int(P.n_sites),
                    "censored_panel": bool((P.censor != 0).any())})
        mae, nonfinite = {}, {}
        for m in METHODS:
            pr = res.preds[m]
            nonfinite[m] = int((~np.isfinite(pr)).sum())
            mae[m] = float(np.nanmean(np.abs(pr - truth)))
        row["mae"], row["nonfinite"] = mae, nonfinite
        row["failures"] = {m: len(v) for m, v in res.failures.items() if v}
        ok = np.isfinite(res.preds["SER"]) & np.isfinite(res.preds["XGBoost"])
        ex, es = np.abs(res.preds["XGBoost"] - truth)[ok], np.abs(res.preds["SER"] - truth)[ok]
        row["ser_gain_pct"] = float(100 * (1 - es.sum() / ex.sum())) if ok.any() else None
        if ok.sum() > 10:
            row["ser_gain_ci"] = boot_gain(sites[ok], ex, es)
        if row["censored_panel"]:                               # interval error treats censored truth as a bound
            def ie(p):
                r = p - truth
                return np.where(cq == 1, np.maximum(-r, 0), np.where(cq == -1, np.maximum(r, 0), np.abs(r)))
            ix, is_ = ie(res.preds["XGBoost"])[ok], ie(res.preds["SER"])[ok]
            row["ser_gain_interval_pct"] = float(100 * (1 - is_.sum() / max(ix.sum(), 1e-300)))
        for alt, nm in (("CovarianceLocalMedian", "local"), ("RidgeSupports", "ridge"), ("QueryMedian", "qmed")):
            a = res.preds[alt]; o = np.isfinite(a) & np.isfinite(res.preds["XGBoost"])
            row[f"{nm}_gain_pct"] = float(100 * (1 - np.abs(a - truth)[o].sum() / np.abs(res.preds["XGBoost"] - truth)[o].sum())) if o.any() else None
        row["intervention_rate"] = float(res.ser_intervened.mean())
        row["gate"] = [d.get("expert") for d in res.decisions.get("SER_gate", [])]
        row["gate_weights"] = [d.get("weight") for d in res.decisions.get("SER_gate", [])]
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = "CRASH"; row["error"] = repr(exc)[:300]; row["trace"] = traceback.format_exc()[-1200:]
    row["seconds"] = round(time.time() - t0, 1)
    try:
        cm.clear_caches()
    except Exception:
        pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True); ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="synth_results.jsonl"); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", default=None, help="comma-separated study-name prefixes, e.g. Syn0001,Syn0002")
    ap.add_argument("--budgets", default="2,3"); ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--scale-check", action="store_true", help="also run each study at x0.001 and x1000")
    a = ap.parse_args()
    man = json.loads((Path(a.bank) / "bank_manifest.json").read_text())
    names = [s["name"] for s in man["studies"]]
    if a.only:
        pre = a.only.split(","); names = [n for n in names if any(n.startswith(p) for p in pre)]
    out = Path(a.out); done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line); done.add((r["study"], r["k"], r.get("scale", 1.0)))
    scales = [1.0, 0.001, 1000.0] if a.scale_check else [1.0]
    jobs = [(n, int(k), a.bank, a.repo, sc) for n in names for k in a.budgets.split(",") for sc in scales
            if (n, int(k), sc) not in done]
    print(f"{len(jobs)} jobs, {a.jobs} workers", flush=True)
    t0 = time.time()
    with Pool(a.jobs) as pool, out.open("a") as fh:
        for row in pool.imap_unordered(one, jobs):
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(row["study"], "k", row["k"], row.get("status", row.get("skipped")), row.get("ser_gain_pct"), row.get("seconds"), flush=True)
            if a.max_seconds and time.time() - t0 > a.max_seconds:
                print("time budget reached; stopping (resume later)"); pool.terminate(); break


if __name__ == "__main__":
    main()
