"""Turn a lockbox evaluator output (results.json) into a readable Markdown report.

Usage (from the repository root):
    python dev_tools/make_lockbox_results_md.py --results results/lockbox_A --lockbox A \
        --study-roots ../SER_Code_and_All_Cleaned_Study_Data/SER/data --out results/lockbox_A/LOCKBOX_RESULTS.md

It only formats numbers the evaluator already wrote. Nothing is recomputed except total site and episode counts.
"""
import argparse
import json
from pathlib import Path

ORDER = ["SER", "CovarianceLocalMedian", "NestedAligned", "AlignedAddAudit", "SAER-7g", "RidgeSupports", "QueryMedian"]
H2 = {"SAER-7g", "AlignedAddAudit", "NestedAligned", "CovarianceLocalMedian"}


def f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def ci(c, nd=2):
    return "n/a" if not c or c[0] is None else f"{c[0]:.{nd}f} to {c[1]:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--lockbox", required=True, choices=["A", "B"])
    ap.add_argument("--study-roots", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    res = json.load(open(Path(a.results) / "results.json"))
    prim, rob, wt = res["primary"], res.get("robustness", {}), res.get("win_tie", {})
    rows = sorted(prim["panel_rows"], key=lambda r: (r["study"], r["budget"]))
    kinase = {}
    for root in [r for r in a.study_roots.split(",") if r]:
        for s in prim["studies"]:
            p = Path(root) / s / "study.json"
            if p.exists():
                kinase[s] = json.load(open(p)).get("kinase", "")
    sites = {}
    for r in rows:
        sites[r["study"]] = max(sites.get(r["study"], 0), r["sites"])
    hyp = "H1" if a.lockbox == "A" else "H1-B"
    title = a.title or f"Lockbox {a.lockbox} results ({hyp})"
    L = [f"# {title}", ""]
    if a.note:
        L += [a.note, ""]
    L += ["Gain G = 100 x (MAE of XGBoost - MAE of the method) / MAE of XGBoost. Positive means the method beats XGBoost. "
          "Equal weight per study; a study's gain is the mean of its budgets (k = 2 and k = 3). 95% intervals come from "
          f"{prim.get('n_draws', 10000)} residue-position bootstrap draws within study. Lockbox A and Lockbox B are never pooled.", ""]
    met = bool(prim.get("meets_target")) and prim["ci95"][0] > 0
    L += [f"## Headline: SER versus XGBoost ({hyp})", "", "| Quantity | Value |", "| --- | --- |",
          f"| Studies | {len(prim['studies'])} |",
          f"| Sites (summed over studies) | {sum(sites.values())} |",
          f"| Episodes (summed over panels and budgets) | {sum(r['episodes'] for r in rows)} |",
          f"| Equal-study mean gain | **{f(prim['gain_percent'])}%** (95% CI {ci(prim['ci95'])}) |",
          f"| Registered criterion (mean gain above 2% and 95% CI above 0) | **{'met' if met else 'not met'}** |",
          f"| 95% CI entirely above 2% | {prim.get('ci_entirely_above_threshold')} (lower end {f(prim['ci95'][0])}%) |",
          f"| Median study gain | {f(rob.get('median_study_gain'))}% |"]
    loso = rob.get("leave_one_study_out")
    if loso:
        v = [x["gain_percent"] for x in loso.values()]
        L.append(f"| Leave-one-study-out mean gain | {f(min(v))}% to {f(max(v))}% |")
    if rob.get("site_weighted"):
        L.append(f"| Site-weighted gain | {f(rob['site_weighted']['gain_percent'])}% (95% CI {ci(rob['site_weighted']['ci95'])}) |")
    if rob.get("episode_weighted"):
        L.append(f"| Episode-weighted gain | {f(rob['episode_weighted']['gain_percent'])}% (95% CI {ci(rob['episode_weighted']['ci95'])}) |")
    if a.lockbox == "A":
        L.append(f"| Studies won / tied / lost | {wt.get('studies_won')} / {wt.get('studies_tied')} / {wt.get('studies_lost')} of {len(prim['studies'])} (ties do not count as wins; chance of at least 5 of 7 by coin flip is 29/128) |")
    else:
        L.append(f"| Studies won / tied / lost | {wt.get('studies_won')} / {wt.get('studies_tied')} / {wt.get('studies_lost')} of {len(prim['studies'])} (the 5-of-7 count is not reported for Lockbox B) |")
    L.append("")
    if a.lockbox == "B":
        L += ["Lockbox B has three studies and 27 sites, so intervals are coarse and one study can dominate. "
              "The episode-weighted gain gives most of the weight to Villa2024_ALK.", ""]

    L += ["## SER gain by panel and budget", "",
          "| Study | Kinase | k | Sites | Episodes | MAE XGBoost | MAE SER | Gain % | 95% CI |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        L.append(f"| {r['study']} | {kinase.get(r['study'], '')} | {r['budget']} | {r['sites']} | {r['episodes']} | "
                 f"{r['mae_xgboost']:.4g} | {r['mae_method']:.4g} | {f(r['gain_percent'])} | {ci(r['gain_ci95'])} |")
    L += ["", "Per-study gain (mean of its budgets): " + "; ".join(f"{s} {f(v)}" for s, v in prim["per_study_gain"].items()), ""]

    summ = dict(res.get("summaries", {}))
    plain = res.get("plain_baselines", {})
    for m, v in plain.items():
        summ.setdefault(m, v)
    L += ["## All methods against XGBoost (equal-study mean gain)", "",
          "| Method | Mean gain % | 95% CI | Role |", "| --- | --- | --- | --- |"]
    for m in ORDER:
        v = summ.get(m)
        if not v:
            continue
        if m == "SER":
            role = f"primary ({hyp})"
        elif m in H2:
            role = "H2 secondary" if a.lockbox == "A" else "secondary (H2 is registered for Lockbox A; descriptive here)"
        else:
            role = "plain baseline (descriptive)"
        L.append(f"| {m} | {f(v['gain_percent'])} | {ci(v['ci95'])} | {role} |")
    L.append("")
    studies = prim["studies"]
    L += ["## Per-study gain by method", "", "| Method | " + " | ".join(studies) + " |", "| --- |" + " --- |" * len(studies)]
    for m in ORDER:
        v = summ.get(m)
        if not v:
            continue
        pg = v.get("per_study_gain", {})
        L.append(f"| {m} | " + " | ".join(f(pg.get(s)) for s in studies) + " |")
    L.append("")

    hs = res.get("secondary_holm")
    if hs:
        label = "H2: Holm-adjusted comparison of the four secondary models with XGBoost" if a.lockbox == "A" else "Holm-adjusted comparison of the four secondary models with XGBoost (descriptive on Lockbox B)"
        L += [f"## {label}", "", f"alpha = {hs['alpha']}. Smallest p is compared with alpha/4, the next with alpha/3, and so on, stopping at the first failure. "
              "The p values are two-sided, so a rejection would mean a difference from XGBoost in either direction; check the sign of the mean gain above.", "",
              "| Method | p | Holm threshold | Rejected |", "| --- | --- | --- | --- |"]
        for m, v in sorted(hs["family"].items(), key=lambda kv: kv[1]["p"]):
            L.append(f"| {m} | {v['p']:.4f} | {v['threshold']:.4f} | {v['reject']} |")
        L.append("")

    inter = res.get("intervention") or []
    if inter:
        L += ["## How often SER changes the XGBoost prediction", "",
              "Intervention rate = share of test episodes where SER's prediction differs from XGBoost's.", "",
              "| Study | k | Episodes | Intervention rate |", "| --- | --- | --- | --- |"]
        for r in inter:
            L.append(f"| {r['study']} | {r['budget']} | {r['episodes']} | {r['intervention_rate']:.3f} |")
        L.append("")
    L += ["## Secondary endpoints", "",
          "T1 to T4 (hidden-resistance screening, inhibitor choice and regret, value of a third measurement, binding-mode strata) are in `results.json` "
          "under their own keys. They are descriptive and are not summarised here.", ""]
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
