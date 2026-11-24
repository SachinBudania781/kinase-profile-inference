"""Summarise synthetic-bank results: failures, harm cases, abstentions and what drives SER's gain."""
import json, sys, collections
import numpy as np
res_path, man_path = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(res_path)]
man = {s["name"]: s for s in json.load(open(man_path))["studies"]}
main = [r for r in rows if r.get("scale", 1.0) == 1.0]
crash = [r for r in main if r.get("status") == "CRASH"]
ok = [r for r in main if r.get("status") == "ok" and r.get("ser_gain_pct") is not None]
print(f"runs: {len(main)}   ok: {len(ok)}   crashed: {len(crash)}")
for r in crash[:8]:
    print("  CRASH", r["study"], "k", r["k"], r.get("error"))
nonfin = [r for r in ok if any(v for m, v in r["nonfinite"].items() if m in ("XGBoost", "SER"))]
print("non-finite XGBoost/SER predictions in", len(nonfin), "runs")
g = np.array([r["ser_gain_pct"] for r in ok])
print(f"SER gain vs XGBoost over {len(ok)} runs: mean {g.mean():.2f}  median {np.median(g):.2f}  "
      f"share >0: {(g>0.05).mean():.2f}  share ~0 (gate off): {(np.abs(g)<=0.05).mean():.2f}  share <0: {(g<-0.05).mean():.2f}")
harm = [r for r in ok if r["ser_gain_pct"] < -1.0 and r.get("ser_gain_ci") and r["ser_gain_ci"][1] < 0]
print(f"HARM cases (gain < -1% and CI entirely below 0): {len(harm)}")
for r in sorted(harm, key=lambda r: r["ser_gain_pct"])[:8]:
    m = man[r["study"]]
    print(f"   {r['study']} k{r['k']} gain {r['ser_gain_pct']:.2f}  CI {np.round(r['ser_gain_ci'],2)}  D{m['D']} n{m['n']} noise{m['noise_mult']} heavy{m['p_heavy_tail']} cens{m['censor_frac']} adv{[k for k,v in m['adversarial'].items() if v]}")
# missed opportunities: SER abstains (gate off) while the local method would have gained clearly
miss = [r for r in ok if abs(r["ser_gain_pct"]) < 0.05 and (r.get("local_gain_pct") or 0) > 3]
print(f"MISSED opportunities (SER = XGBoost but local alternative gains > 3%): {len(miss)} of {len(ok)}")
for r in sorted(miss, key=lambda r: -r["local_gain_pct"])[:6]:
    m = man[r["study"]]
    print(f"   {r['study']} k{r['k']} local gain {r['local_gain_pct']:.2f}  sites {r['sites']}  D{m['D']} n{m['n']}")
# censored interval check
cens = [r for r in ok if r.get("ser_gain_interval_pct") is not None]
if cens:
    d = np.array([r["ser_gain_interval_pct"] - r["ser_gain_pct"] for r in cens])
    print(f"censored panels: {len(cens)}; mean (interval gain - reported gain) {d.mean():.2f}; interval gain < -1%: {sum(r['ser_gain_interval_pct'] < -1 for r in cens)}")
# scale invariance (if present)
by = collections.defaultdict(dict)
for r in rows:
    if r.get("status") == "ok" and r.get("ser_gain_pct") is not None:
        by[(r["study"], r["k"])][r.get("scale", 1.0)] = r["ser_gain_pct"]
bad = [(k, v) for k, v in by.items() if len(v) == 3 and (abs(v[0.001] - v[1.0]) > 0.5 or abs(v[1000.0] - v[1.0]) > 0.5)]
print(f"scale check: {sum(len(v)==3 for v in by.values())} study/k pairs tested, {len(bad)} not scale-invariant (>0.5 pt change)")
# drivers: binned mean gain
def binned(key, edges=None, cats=False):
    out = collections.defaultdict(list)
    for r in ok:
        v = man[r["study"]][key]
        if cats: out[str(v)].append(r["ser_gain_pct"])
        else:
            b = next((e for e in edges if v <= e), edges[-1] + 1)
            out[f"<={b}"].append(r["ser_gain_pct"])
    return {k: (round(float(np.mean(v)), 2), len(v)) for k, v in sorted(out.items(), key=lambda kv: float(kv[0].strip("<=")) if not cats else 0)}
print("mean SER gain by D        :", binned("D", [3, 5, 8, 14, 30, 60, 120]))
print("mean SER gain by n        :", binned("n", [60, 200, 800, 3000]))
print("mean SER gain by rank     :", binned("rank", cats=True))
print("mean SER gain by site_var :", binned("site_var_frac", cats=True))
print("mean SER gain by noise    :", binned("noise_mult", cats=True))
print("mean SER gain by censor   :", binned("censor_frac", cats=True))
print("mean SER gain by fill     :", binned("fill", cats=True))

# ---- what drives the gain: rank correlations and a small linear model (artificial bank only) ----
import math
from scipy.stats import spearmanr
X, y, names = [], [], []
feats = {
    "log10 D": lambda m, r: math.log10(m["D"]),
    "log10 n": lambda m, r: math.log10(m["n"]),
    "log10 (n x fill^3 / mask types)": lambda m, r: math.log10(max(m["n"] * m["fill"] ** 3 / (m["D"] * math.comb(m["D"] - 1, r["k"])), 1e-9)),
    "site variance share": lambda m, r: m["site_var_frac"],
    "noise multiplier": lambda m, r: m["noise_mult"],
    "rank": lambda m, r: m["rank"],
    "common loading": lambda m, r: m["common_loading"],
    "resistant share": lambda m, r: m["p_resistant"],
    "saturating share": lambda m, r: m["p_saturating"],
    "censor fraction": lambda m, r: m["censor_frac"],
    "missing share": lambda m, r: 1 - m["fill"],
    "outlier share": lambda m, r: m["p_outlier"],
}
print("\nSpearman correlation of SER gain with each generator setting (higher = SER helps more):")
for nm, f in feats.items():
    xs = [f(man[r["study"]], r) for r in ok]
    rho = spearmanr(xs, [r["ser_gain_pct"] for r in ok])[0]
    print(f"   {nm:34s} {rho:+.2f}")
Xm = np.array([[feats[k](man[r["study"]], r) for k in ("log10 D", "log10 n", "site variance share", "noise multiplier", "common loading")] for r in ok])
Xm = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-12)
yv = np.array([r["ser_gain_pct"] for r in ok])
A = np.column_stack([np.ones(len(yv)), Xm]); coef = np.linalg.lstsq(A, yv, rcond=None)[0]
print("standardised linear model (gain points per 1 SD):", dict(zip(["intercept", "log10 D", "log10 n", "site share", "noise", "common loading"], np.round(coef, 2))))
