"""Turn an evaluator output folder (results.json + panel_budget_table.csv) into a readable Markdown report.

Usage (from the repository root):
    python dev_tools/make_dev_results_md.py --results results/dev --data data --out results/dev/DEV_RESULTS.md --title "..."

It only formats numbers that the evaluator already wrote. Nothing is recomputed except
the equal-study mean for the secondary methods (mean of per-study gains, each study = mean over its budgets),
which uses the same rule the evaluator uses for SER.
"""
import argparse
import csv
import json
from pathlib import Path

CORE = ["Estevam2025_MET", "DuongLy2016_multi_kinase", "Robichaux2021_EGFR",
        "Nakamura2021_FGFR1_FGFR2_FGFR3", "KiRHub2026_multi_kinase", "OKL2026_multi_kinase"]
STRESS = ["Jung2025_ABL1", "Chakraborty2024_SRC", "Lee2026_EGFR", "Oh2026_ALK"]
ORDER = {"SER": 0, "CovarianceLocalMedian": 1, "NestedAligned": 2, "AlignedAddAudit": 3, "SAER-7g": 4,
         "RidgeSupports": 5, "QueryMedian": 6}


def f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def ci(c, nd=2):
    return "n/a" if not c or c[0] is None else f"{c[0]:.{nd}f} to {c[1]:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Development results")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    res = json.load(open(Path(a.results) / "results.json"))
    rows = list(csv.DictReader(open(Path(a.results) / "panel_budget_table.csv")))
    kinase = {}
    if a.data:
        for s in CORE + STRESS:
            p = Path(a.data) / s / "study.json"
            if p.exists():
                kinase[s] = json.load(open(p)).get("kinase", "")
    prim = res["primary"]
    rob = res.get("robustness", {})
    wt = res.get("win_tie", {})
    L = [f"# {a.title}", ""]
    if a.note:
        L += [a.note, ""]
    L += ["Gain G = 100 x (MAE of XGBoost - MAE of the method) / MAE of XGBoost. Positive means the method beats XGBoost. "
          "Equal weight per study; a study's gain is the mean of its k = 2 and k = 3 gains. 95% intervals come from "
          f"{prim.get('n_draws', 10000)} residue-position bootstrap draws within study.", ""]
    L += ["## Headline: SER versus XGBoost", "",
          "| Quantity | Value |", "| --- | --- |",
          f"| Studies | {len(prim['studies'])} ({len([s for s in prim['studies'] if s in CORE])} core, {len([s for s in prim['studies'] if s in STRESS])} stress test) |",
          f"| Equal-study mean gain | **{f(prim['gain_percent'])}%** (95% CI {ci(prim['ci95'])}) |",
          f"| Median study gain | {f(rob.get('median_study_gain'))}% |"]
    loso = rob.get("leave_one_study_out")
    if loso:
        v = [x["gain_percent"] for x in loso.values()]
        L.append(f"| Leave-one-study-out mean gain | {f(min(v))}% to {f(max(v))}% |")
    if rob.get("site_weighted"):
        L.append(f"| Site-weighted gain | {f(rob['site_weighted']['gain_percent'])}% (95% CI {ci(rob['site_weighted']['ci95'])}) |")
    if rob.get("episode_weighted"):
        L.append(f"| Episode-weighted gain | {f(rob['episode_weighted']['gain_percent'])}% (95% CI {ci(rob['episode_weighted']['ci95'])}) |")
    if wt:
        L.append(f"| Wins / ties / losses (studies) | {wt.get('studies_won')} / {wt.get('studies_tied')} / {wt.get('studies_lost')} |")
    L.append(f"| Target met (mean gain > 2% and CI above 0) | {prim.get('meets_target')} (CI entirely above the 2% threshold: {prim.get('ci_entirely_above_threshold')}) |")
    L.append("")
    L += ["The 2% threshold and the CI rule are the registered H1 rule. These are development numbers on development panels, not the confirmatory result.", ""]

    L += ["## SER gain by panel and budget", "",
          "| Study | Role | Kinase | k | Sites | Episodes | MAE XGBoost | MAE SER | Gain % | 95% CI |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    ser = [r for r in rows if r["method"] == "SER"]
    ser.sort(key=lambda r: (r["study"] not in CORE, r["study"], int(r["budget"])))
    for r in ser:
        role = "core" if r["study"] in CORE else "stress test"
        g = float(r["gain_percent"]) if r["gain_percent"] not in ("", None) else None
        c = json.loads(r["gain_ci95"]) if r["gain_ci95"] else None
        L.append(f"| {r['study']} | {role} | {kinase.get(r['study'], '')} | {r['budget']} | {r['sites']} | {r['episodes']} | "
                 f"{float(r['mae_xgboost']):.4g} | {float(r['mae_method']):.4g} | {f(g)} | {ci(c)} |")
    L += ["", "Per-study gain (mean of its budgets): " + "; ".join(
        f"{s} {f(v)}" for s, v in prim["per_study_gain"].items()), ""]

    L += ["## All methods against XGBoost", "",
          "Equal-study mean gain over the studies where the method produced every prediction (no failed episodes). "
          "A method with fewer than 10 studies could not be run on every panel in that run.", "",
          "| Method | Studies used | Mean gain % | Studies skipped (failed episodes or undefined) |", "| --- | --- | --- | --- |"]
    by = {}
    for r in rows:
        by.setdefault(r["method"], {}).setdefault(r["study"], []).append(r)
    for m in sorted(by, key=lambda x: ORDER.get(x, 99)):
        if m == "XGBoost":
            continue
        gains, skipped = [], []
        for s, rr in by[m].items():
            ok = all(r["gain_defined"] == "True" and int(r["failed_episodes"]) == 0 and r["gain_percent"] not in ("", None) for r in rr)
            if ok:
                gains.append(sum(float(r["gain_percent"]) for r in rr) / len(rr))
            else:
                skipped.append(s)
        mean = sum(gains) / len(gains) if gains else None
        L.append(f"| {m} | {len(gains)} | {f(mean)} | {', '.join(skipped) if skipped else 'none'} |")
    L.append("")

    inter = res.get("intervention") or []
    if inter:
        L += ["## How often SER changes the XGBoost prediction", "",
              "Intervention rate = share of test episodes where SER's prediction differs from XGBoost's.", "",
              "| Study | k | Episodes | Intervention rate |", "| --- | --- | --- | --- |"]
        for r in inter:
            L.append(f"| {r['study']} | {r['budget']} | {r['episodes']} | {r['intervention_rate']:.3f} |")
        L.append("")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
