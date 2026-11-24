"""Run the baseline engine (XGBoost, SER, local, ridge, medians) and store the per-fold records so gate variants
can be replayed without refitting. Development tool: dev panels and artificial studies only (never a lockbox).

Examples (from the repository folder):
  python dev_tools/capture_records.py --archive-root <data> --bank <bank> --config configs/models.json \
         --targets Lee2026_EGFR:2,Syn0017_D4_n60_mat:2 --out dev_records/v1 --jobs 2
  python dev_tools/capture_records.py ... --set revision.neighbors_whitening=robust --out dev_records/robust
"""
from __future__ import annotations
import argparse, json, pickle, sys, time, traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _apply_sets(cfg, sets):
    for item in sets or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            node[parts[-1]] = json.loads(val)
        except json.JSONDecodeError:
            node[parts[-1]] = val
    return cfg


def one(job):
    name, k, a = job
    sys.path.insert(0, str(REPO))
    import core_models as cm, data_pipeline as dp
    out = Path(a["out"]) / f"{name}_k{k}.pkl"
    if out.exists():
        return {"target": f"{name}:{k}", "status": "exists"}
    t0 = time.time()
    try:
        if name.startswith("Syn"):
            reg = dp.load_registry(Path(a["bank"]) / "registry.json")
            roots = {"synth": str(Path(a["bank"]) / "data")}
            allow = False
        else:
            reg = dp.load_registry(str(REPO / "configs" / "panels.json"))
            roots = {"archive": a["archive_root"]}
            allow = False
        spec_role = reg["panels"][name]["role"]
        if spec_role.startswith("lockbox"):
            raise PermissionError("refusing: lockbox panel")
        P = dp.load_panel(reg, name, roots, allow_lockbox=allow)
        if k not in P.budgets():
            return {"target": f"{name}:{k}", "status": "skipped (budget not allowed)"}
        cfg = _apply_sets(cm.load_config(a["config"]), a["sets"])
        ep = dp.build_episodes(P, k)
        cap = {}
        r = cm.run_baselines(P, ep, cfg, capture=cap)
        sites = np.array([dp.site_key(s) for s in P.sites], dtype=object)[ep.i]
        rec = {"name": name, "k": k, "config_revision": cfg.get("revision"), "episodes": len(ep), "ids": ep.ids,
               "truth": P.y[ep.i, ep.q], "query_censor": P.censor[ep.i, ep.q], "sites": sites, "folds": P.folds[ep.i],
               "q": ep.q, "n_drugs": P.n_drugs, "n": P.n, "n_sites": P.n_sites, "scale_kind": P.scale_kind,
               "censored_panel": bool((P.censor != 0).any()),
               "preds": {"XGBoost": r.xgb, "SER": r.ser, "CovarianceLocalMedian": r.local, "QueryMedian": r.qmed, "RidgeSupports": r.ridge},
               "decisions": r.decisions, "selected": r.selected, "records": cap, "seconds": round(time.time() - t0, 1)}
        Path(a["out"]).mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(rec, fh)
        return {"target": f"{name}:{k}", "status": "ok", "seconds": rec["seconds"]}
    except Exception as exc:
        return {"target": f"{name}:{k}", "status": "CRASH", "error": repr(exc)[:300], "trace": traceback.format_exc()[-800:]}
    finally:
        try:
            cm.clear_caches()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-root", default=None); ap.add_argument("--bank", default=None)
    ap.add_argument("--config", default=str(REPO / "configs" / "models.json"))
    ap.add_argument("--targets", default=None, help="comma list name:k (default: all dev panels in the registry, k=2,3)")
    ap.add_argument("--set", dest="sets", action="append", default=[], help="override, e.g. revision.ser_censor_mode=no_defer_bi")
    ap.add_argument("--out", required=True); ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    if a.targets:
        targets = [(t.split(":")[0], int(t.split(":")[1])) for t in a.targets.split(",")]
    else:
        reg = json.loads((REPO / "configs" / "panels.json").read_text())
        targets = [(n, k) for n, s in reg["panels"].items() if s["role"] in ("core_dev", "stress_test") for k in (2, 3)]
    args = {"archive_root": a.archive_root, "bank": a.bank, "config": a.config, "sets": a.sets, "out": a.out}
    jobs = [(n, k, args) for n, k in targets]
    print(f"{len(jobs)} jobs", flush=True)
    with Pool(a.jobs) as pool:
        for res in pool.imap_unordered(one, jobs):
            print(res, flush=True)


if __name__ == "__main__":
    main()
