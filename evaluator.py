"""Step 5a - evaluator: scoring join, primary contrast, robustness checks and translational endpoints.

The evaluator is the only place where hidden query responses meet predictions. It reads predictions
by episode ID (``core_models.save_run`` output) and joins truth from the panel inside ``score_runs``.

Primary (pre-registered): equal-study mean of the relative MAE gain of SER over the XGBoost reference,
10,000 paired draws resampling residue sites within each study.
Everything else (win/tie table, robustness checks, plain baselines, T1-T4) is secondary or descriptive.

    python evaluator.py --runs runs/ --data-root <archive> --roles core_dev,stress_test --out results/
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binom

import data_pipeline as dp

SEED = dp.SEED
PRIMARY = "SER"
REFERENCE = "XGBoost"
CHECKPOINTS = ("SAER-7g", "AlignedAddAudit", "NestedAligned", "CovarianceLocalMedian")
DESCRIPTIVE = ("QueryMedian", "RidgeSupports")

EVAL_DEFAULTS = {
    "bootstrap_draws": 10000,
    "bootstrap_seed": SEED,
    "alpha": 0.05,                      # [CONFIRM] familywise level for the Holm family of four
    "practical_threshold": 2.0,         # percent gain; see OSF section 7 for its justification
    "tie_tolerance": 1e-12,
    "win_threshold_studies": 5,         # descriptive: SER better in at least 5 of 7 studies (ties do not count)
    "t1_min_eligible": 20,              # minimum eligible variants for a support/query mask to be reviewed
    "t1_review_fraction": 0.05,
    "t1_ic50_sensitive_below": math.log10(3.0),
    "t1_ic50_resistant_at_least": 1.0,
    "t2_min_remaining": 2,              # at least two remaining inhibitors are needed to choose between
    "t2_unit": "variant",
    "t4_min_episodes": 1,
}


# --------------------------------------------------------------------------------------
# Scoring join
# --------------------------------------------------------------------------------------
@dataclass
class Scored:
    panel: str
    k: int
    ids: np.ndarray
    i: np.ndarray
    q: np.ndarray
    s: np.ndarray
    site: np.ndarray           # site index within the study
    truth: np.ndarray
    censor: np.ndarray
    support_censor: np.ndarray
    support_y: np.ndarray
    preds: dict
    failures: dict
    drugs: tuple
    modes: tuple
    scale_kind: str
    n_sites: int


def score_runs(panel: dp.Panel, ep: dp.EpisodeSet, preds: dict, failures: dict | None = None) -> Scored:
    """Join the hidden truth to the predictions by episode. Every method must have one prediction or an explicit failure."""
    failures = failures or {}
    E = len(ep)
    for m, p in preds.items():
        p = np.asarray(p, dtype=float)
        if len(p) != E:
            raise ValueError(f"{panel.name} k={ep.k} {m}: {len(p)} predictions for {E} episodes.")
        missing = set(np.flatnonzero(~np.isfinite(p)).tolist())
        if not missing <= set((failures.get(m) or {}).keys()):
            raise ValueError(f"{panel.name} k={ep.k} {m}: missing predictions without an explicit failure row.")
    t = ep.truth(panel)
    sites = np.array([dp.site_key(x) for x in panel.sites], dtype=object)
    site_list = sorted(set(sites))
    index = {s: n for n, s in enumerate(site_list)}
    return Scored(panel.name, ep.k, ep.ids, ep.i, ep.q, ep.s, np.array([index[s] for s in sites[ep.i]]), t["y"], t["censor"],
                  panel.censor[ep.i[:, None], ep.s], panel.y[ep.i[:, None], ep.s],
                  {m: np.asarray(p, dtype=float) for m, p in preds.items()}, failures, panel.drugs, panel.modes,
                  panel.scale_kind, len(site_list))


def load_predictions_csv(path: Path) -> tuple[dict, dict, np.ndarray]:
    """Read ``core_models.save_run`` predictions: returns (preds, failures, ids)."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = np.array([r["episode_id"] for r in rows])
    methods = [c for c in rows[0] if c not in ("episode_id", "fold", "failure_reasons")]
    preds = {m: np.array([float(r[m]) if r[m] != "" else np.nan for r in rows]) for m in methods}
    failures = {m: {} for m in methods}
    for e, r in enumerate(rows):
        for part in filter(None, r["failure_reasons"].split(";")):
            m, _, reason = part.partition("=")
            failures.setdefault(m, {})[e] = reason
    return preds, failures, ids


# --------------------------------------------------------------------------------------
# Errors and gains
# --------------------------------------------------------------------------------------
def abs_err(sc: Scored, m: str) -> np.ndarray:
    """Primary loss: distance to the reported number (boundary error for censored hidden responses)."""
    return np.abs(sc.preds[m] - sc.truth)


def interval_err(sc: Scored, m: str) -> np.ndarray:
    r = sc.preds[m] - sc.truth
    return np.where(sc.censor == 1, np.maximum(-r, 0), np.where(sc.censor == -1, np.maximum(r, 0), np.abs(r)))


def gain(mae_ref, mae_m):
    with np.errstate(all="ignore"):
        return np.where(np.asarray(mae_ref) > 0, 100.0 * (np.asarray(mae_ref) - np.asarray(mae_m)) / np.asarray(mae_ref), np.nan)


def valid_pair(sc: Scored, m: str) -> np.ndarray:
    return np.isfinite(sc.preds[m]) & np.isfinite(sc.preds[REFERENCE])


def site_sums(sc: Scored, m: str, ref: str = REFERENCE, kind: str = "abs"):
    """Per-site sums of the method's and reference's loss and episode counts on episodes valid for both."""
    ok = np.isfinite(sc.preds[m]) & np.isfinite(sc.preds[ref])
    f = abs_err if kind == "abs" else interval_err
    e_m, e_r = f(sc, m), f(sc, ref)
    n = np.bincount(sc.site[ok], minlength=sc.n_sites).astype(float)
    return (np.bincount(sc.site[ok], weights=e_m[ok], minlength=sc.n_sites),
            np.bincount(sc.site[ok], weights=e_r[ok], minlength=sc.n_sites), n, int((~ok).sum()))


# --------------------------------------------------------------------------------------
# Primary analysis and robustness
# --------------------------------------------------------------------------------------
def bootstrap_matrices(studies: dict, draws: int, seed: int):
    """Multiplicity matrices (draws x sites) per study; the same sampled sites serve every budget of a study."""
    out = {}
    for n, name in enumerate(sorted(studies)):
        n_sites = next(iter(studies[name].values())).n_sites
        rng = np.random.default_rng([seed, n])
        out[name] = rng.multinomial(n_sites, np.full(n_sites, 1.0 / n_sites), size=draws).astype(float)
    return out


def percentile_ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))] if len(x) else [None, None]


def study_table(studies: dict, method: str, draws: dict | None):
    """Per panel/budget MAEs, gain and bootstrap draws of the gain for ``method`` against XGBoost."""
    rows, boot = [], {}
    for name in sorted(studies):
        for k in sorted(studies[name]):
            sc = studies[name][k]
            if method not in sc.preds:
                continue
            em, er, n, failed = site_sums(sc, method)
            mae_m, mae_r = em.sum() / max(n.sum(), 1), er.sum() / max(n.sum(), 1)
            row = {"study": name, "budget": k, "method": method, "episodes": int(n.sum()), "sites": int((n > 0).sum()),
                   "failed_episodes": failed, "mae_method": float(mae_m), "mae_xgboost": float(mae_r),
                   "gain_percent": None if mae_r == 0 else float(100 * (mae_r - mae_m) / mae_r),
                   "gain_defined": bool(mae_r > 0), "absolute_error_difference": float(mae_r - mae_m)}
            if draws is not None:
                M = draws[name]
                den = M @ n
                with np.errstate(all="ignore"):
                    mm, rr = (M @ em) / den, (M @ er) / den
                g = gain(rr, mm)
                boot[(name, k)] = g
                row["gain_ci95"] = percentile_ci(g)
            rows.append(row)
    return rows, boot


def study_gains(rows: list, boot: dict, weights: dict | None = None):
    """Average gains over budgets within a study, then over studies with equal weight."""
    studies = sorted({r["study"] for r in rows})
    per_study, per_study_boot = {}, {}
    for s in studies:
        rs = [r for r in rows if r["study"] == s]
        per_study[s] = float(np.mean([r["gain_percent"] for r in rs])) if all(r["gain_defined"] for r in rs) else None
        bs = [boot[(s, r["budget"])] for r in rs if (s, r["budget"]) in boot]
        per_study_boot[s] = np.mean(bs, axis=0) if bs else None
    return per_study, per_study_boot


def summarise_primary(per_study: dict, per_study_boot: dict, cfg: dict, expected_studies: int | None = None):
    names = sorted(per_study)
    complete = all(v is not None for v in per_study.values()) and (expected_studies is None or len(names) == expected_studies)
    out = {"studies": names, "complete": bool(complete), "per_study_gain": per_study}
    if not complete:
        out["reason"] = "a required configuration is missing or its ratio is undefined"
        return out
    est = float(np.mean([per_study[s] for s in names]))
    boot = np.mean([per_study_boot[s] for s in names], axis=0)
    ci = percentile_ci(boot)
    thr = cfg["practical_threshold"]
    out.update({"gain_percent": est, "ci95": ci, "n_draws": int(np.isfinite(boot).sum()),
                "meets_target": bool(est > thr and ci[0] is not None and ci[0] > 0),
                "ci_entirely_above_threshold": bool(ci[0] is not None and ci[0] > thr),
                "bootstrap_p_two_sided": bootstrap_p(boot)})
    return out


def bootstrap_p(boot) -> float:
    b = np.asarray(boot)
    b = b[np.isfinite(b)]
    if not len(b):
        return float("nan")
    return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()) + 1.0 / (len(b) + 1)))


def holm(pvalues: dict, alpha: float) -> dict:
    """Holm step-down over the family: smallest p vs alpha/m, next vs alpha/(m-1), stop at the first failure."""
    order = sorted(pvalues, key=lambda k: (pvalues[k], k))
    m = len(order)
    out, stop = {}, False
    for rank, name in enumerate(order):
        thr = alpha / (m - rank)
        reject = (not stop) and pvalues[name] <= thr
        stop = stop or not reject
        out[name] = {"p": pvalues[name], "threshold": thr, "reject": bool(reject)}
    return out


def win_tie_table(rows: list, per_study: dict, cfg: dict):
    """Edit 1: strictly lower MAE is a win; identical MAE is a tie. Reported per panel-budget and per study."""
    tol = cfg["tie_tolerance"]

    def tag(diff):
        return "win" if diff > tol else "loss" if diff < -tol else "tie"
    per_panel = [{"study": r["study"], "budget": r["budget"], "outcome": tag(r["mae_xgboost"] - r["mae_method"])} for r in rows]
    study_out = {}
    for s in sorted(per_study):
        d = [r for r in rows if r["study"] == s]
        mean_diff = float(np.mean([r["mae_xgboost"] - r["mae_method"] for r in d]))
        study_out[s] = tag(mean_diff)
    wins = sum(v == "win" for v in study_out.values())
    n = len(study_out)
    return {"per_panel_budget": per_panel, "per_study": study_out,
            "studies_won": wins, "studies_tied": sum(v == "tie" for v in study_out.values()),
            "studies_lost": sum(v == "loss" for v in study_out.values()),
            "meets_descriptive_5_of_7": bool(wins >= cfg["win_threshold_studies"]),
            "sign_test_upper_tail": float(binom.sf(wins - 1, n, 0.5)) if n else None,
            "note": "descriptive: ties are not wins; chance of at least 5 of 7 by luck is 29/128 (22.7%)"}


def robustness(per_study: dict, per_study_boot: dict, studies: dict, cfg: dict):
    """Edit 2: checks that never replace the equal-study primary estimate."""
    names = sorted(per_study)
    if any(per_study[s] is None for s in names):
        return {"available": False}
    g = np.array([per_study[s] for s in names])
    boot = {s: per_study_boot[s] for s in names}
    out = {"available": True, "median_study_gain": float(np.median(g))}
    loso = {}
    for left in names:
        keep = [s for s in names if s != left]
        if not keep:                                    # a single study has no leave-one-out estimate
            continue
        b = np.mean([boot[s] for s in keep], axis=0)
        loso[left] = {"gain_percent": float(np.mean([per_study[s] for s in keep])), "ci95": percentile_ci(b)}
    out["leave_one_study_out"] = loso
    n_ep = np.array([sum(sc.truth.size for sc in studies[s].values()) for s in names], dtype=float)
    n_site = np.array([next(iter(studies[s].values())).n_sites for s in names], dtype=float)
    for label, w in (("episode_weighted", n_ep), ("site_weighted", n_site)):
        w = w / w.sum()
        b = np.sum([wi * boot[s] for wi, s in zip(w, names)], axis=0)
        out[label] = {"gain_percent": float((w * g).sum()), "ci95": percentile_ci(b),
                      "weights": {s: float(x) for s, x in zip(names, w)}}
    return out


def plain_baseline_rows(studies: dict, draws: dict | None):
    """Edit 3: query median and ridge-on-supports gains against XGBoost, descriptive."""
    out = {}
    for m in DESCRIPTIVE:
        rows, boot = study_table(studies, m, draws)
        if rows:
            per, pb = study_gains(rows, boot)
            out[m] = {"panel_rows": rows, **summarise_primary(per, pb, EVAL_DEFAULTS)}
    return out


def intervention_report(studies: dict, interventions: dict):
    """Edit 1: SER intervention rate per panel/budget and per fold (episodes where SER differs from XGBoost)."""
    out = []
    for name in sorted(studies):
        for k in sorted(studies[name]):
            sc = studies[name][k]
            diff = np.isfinite(sc.preds[PRIMARY]) & np.isfinite(sc.preds[REFERENCE]) & (sc.preds[PRIMARY] != sc.preds[REFERENCE])
            rec = {"study": name, "budget": k, "episodes": int(len(diff)), "intervention_rate": float(diff.mean())}
            fold = interventions.get((name, k))
            if fold is not None:
                rec["by_fold"] = fold
            out.append(rec)
    return out


def exact_and_interval_tables(studies: dict, method: str):
    """Sensitivity: interval-error MAE and the fully exact-profile subset (censored panels only)."""
    rows = []
    for name in sorted(studies):
        for k in sorted(studies[name]):
            sc = studies[name][k]
            if method not in sc.preds or not ((sc.censor != 0).any() or (sc.support_censor != 0).any()):
                continue
            ok = valid_pair(sc, method)
            exact_profile = ok & (sc.censor == 0) & (sc.support_censor == 0).all(axis=1)
            rec = {"study": name, "budget": k}
            for label, mask, f in (("interval", ok, interval_err), ("exact_profile_subset", exact_profile, abs_err)):
                if mask.sum():
                    a, b = f(sc, method)[mask].mean(), f(sc, REFERENCE)[mask].mean()
                    rec[label] = {"denominator": int(mask.sum()), "mae_method": float(a), "mae_xgboost": float(b),
                                  "gain_percent": None if b == 0 else float(100 * (b - a) / b)}
            rows.append(rec)
    return rows


# --------------------------------------------------------------------------------------
# T1 - hidden-resistance screening
# --------------------------------------------------------------------------------------
def _below(y, c, thr):
    """(determined, value) for the statement 'response < thr' given a censor flag."""
    determined = np.where(c == 0, True, np.where(c == -1, y < thr, y >= thr))
    value = np.where(c == 0, y < thr, np.where(c == -1, True, False))
    return determined, value


def _at_least(y, c, thr, strict=False):
    cmp = (y > thr) if strict else (y >= thr)
    determined = np.where(c == 0, True, np.where(c == 1, cmp, y < thr))
    value = np.where(c == 0, cmp, np.where(c == 1, True, False))
    return determined, value


def screening_labels(sc: Scored, cfg: dict):
    """Eligibility (supports satisfy the sensitivity rule) and resistance label, with ambiguous cases counted."""
    if sc.scale_kind == "enrichment":
        sup_det, sup_val = _below(sc.support_y, sc.support_censor, 0.0)
        hid_det, hid_val = _at_least(sc.truth, sc.censor, 0.0, strict=True)
    elif sc.scale_kind == "ic50_log10_ratio":
        sup_det, sup_val = _below(sc.support_y, sc.support_censor, cfg["t1_ic50_sensitive_below"])
        hid_det, hid_val = _at_least(sc.truth, sc.censor, cfg["t1_ic50_resistant_at_least"])
    else:
        return None
    sup_ok = sup_det.all(axis=1)
    sup_sens = (sup_det & sup_val).all(axis=1)
    ambiguous_support = ~sup_ok & ~(sup_det & ~sup_val).any(axis=1)   # cannot tell if every support is sensitive
    eligible = sup_sens & hid_det
    ambiguous = (sup_ok & sup_sens & ~hid_det) | ambiguous_support
    return {"eligible": eligible, "positive": eligible & hid_val, "ambiguous": ambiguous}


def average_precision(order_labels: np.ndarray) -> float:
    pos = order_labels.sum()
    if pos == 0:
        return float("nan")
    hits = np.cumsum(order_labels)
    return float((hits / np.arange(1, len(order_labels) + 1))[order_labels.astype(bool)].mean())


def t1_screening(sc: Scored, methods: list, cfg: dict):
    lab = screening_labels(sc, cfg)
    if lab is None:
        return {"defined": False, "reason": f"no sensitivity rule for scale '{sc.scale_kind}'"}
    masks = np.column_stack([sc.q, sc.s])
    uniq, inv = np.unique(masks, axis=0, return_inverse=True)
    inv = inv.ravel()
    out = {"defined": True, "ambiguous_excluded": int(lab["ambiguous"].sum()), "groups_total": 0, "groups_used": 0,
           "groups_below_minimum": 0, "methods": {}}
    acc = {m: {"hits": 0, "reviewed": 0, "ap": [], "pairs": set()} for m in methods}
    pos_total = elig_total = 0
    for g in range(len(uniq)):
        idx = np.flatnonzero((inv == g) & lab["eligible"])
        if not len(idx):
            continue
        out["groups_total"] += 1
        if len(idx) < cfg["t1_min_eligible"]:
            out["groups_below_minimum"] += 1
            continue
        out["groups_used"] += 1
        review = int(math.ceil(cfg["t1_review_fraction"] * len(idx)))
        pos = lab["positive"][idx]
        pos_total += int(pos.sum())
        elig_total += len(idx)
        for m in methods:
            p = sc.preds[m][idx]
            order = np.lexsort((sc.ids[idx], -np.where(np.isfinite(p), p, -np.inf)))     # stable episode-id tie break
            top = order[:review]
            acc[m]["hits"] += int(pos[top].sum())
            acc[m]["reviewed"] += review
            ap = average_precision(pos[order])
            if np.isfinite(ap):
                acc[m]["ap"].append(ap)
            acc[m]["pairs"] |= {(int(sc.i[idx][t]), int(sc.q[idx][t])) for t in top if pos[t]}
    out["eligible_episodes_used"], out["positive_episodes_used"] = elig_total, pos_total
    base = pos_total / elig_total if elig_total else float("nan")
    out["base_rate"] = base
    for m in methods:
        a = acc[m]
        prec = a["hits"] / a["reviewed"] if a["reviewed"] else float("nan")
        out["methods"][m] = {"hits": a["hits"], "reviewed": a["reviewed"], "precision": prec,
                             "recall": a["hits"] / pos_total if pos_total else float("nan"),
                             "enrichment_factor": prec / base if base and np.isfinite(prec) else float("nan"),
                             "mean_average_precision": float(np.mean(a["ap"])) if a["ap"] else float("nan"),
                             "groups_with_positives": len(a["ap"]), "distinct_hit_pairs": len(a["pairs"])}
    return out


# --------------------------------------------------------------------------------------
# T2 - inhibitor choice and regret
# --------------------------------------------------------------------------------------
def t2_choice(sc: Scored, methods: list, cfg: dict):
    D = len(sc.drugs)
    remaining = D - sc.k
    if remaining < cfg["t2_min_remaining"]:
        return {"defined": False, "reason": f"only {remaining} remaining inhibitor(s) at k={sc.k}; at least "
                                            f"{cfg['t2_min_remaining']} are needed"}
    key = {}
    for e in range(len(sc.ids)):
        key.setdefault((int(sc.i[e]), tuple(sc.s[e])), {})[int(sc.q[e])] = e
    per_unit = {m: {} for m in methods}
    n_sets = excluded_incomplete = excluded_censored = ties = 0
    best_hits = {m: 0 for m in methods}
    for (i, S), by_q in key.items():
        rest = [j for j in range(D) if j not in S]
        if set(by_q) != set(rest):
            excluded_incomplete += 1
            continue
        eps = [by_q[j] for j in rest]
        if (sc.censor[eps] != 0).any():
            excluded_censored += 1
            continue
        n_sets += 1
        y = sc.truth[eps]
        best = y.min()
        for m in methods:
            p = sc.preds[m][eps]
            if not np.isfinite(p).all():
                continue
            pick = int(np.argmin(p))                                # ties: lowest drug index
            if np.sum(np.abs(p - p.min()) <= cfg["tie_tolerance"]) > 1 and m == methods[0]:
                ties += 1
            per_unit[m].setdefault(i, []).append(y[pick] - best)
            best_hits[m] += int(abs(y[pick] - best) <= cfg["tie_tolerance"])
    out = {"defined": True, "support_sets": n_sets, "excluded_incomplete": excluded_incomplete,
           "excluded_censored": excluded_censored, "prediction_ties_first_method": ties, "unit": cfg["t2_unit"],
           "methods": {}}
    for m in methods:
        regrets = [float(np.mean(v)) for v in per_unit[m].values()]
        out["methods"][m] = {"eligible_variants": len(regrets), "mean_regret": float(np.mean(regrets)) if regrets else None,
                             "chosen_is_best_fraction": best_hits[m] / n_sets if n_sets else None}
    return out


# --------------------------------------------------------------------------------------
# T3 - value of a third measurement
# --------------------------------------------------------------------------------------
def t3_third_measurement(sc2: Scored, sc3: Scored, methods: list):
    """Paired change in absolute error: base two-drug episode vs the mean over its three-drug supersets."""
    D = len(sc2.drugs)
    idx3 = {(int(i), int(q), tuple(s)): e for e, (i, q, s) in enumerate(zip(sc3.i, sc3.q, sc3.s))}
    out = {"defined": True, "base_episodes": 0, "methods": {}}
    deltas = {m: [] for m in methods}
    sites = []
    for e in range(len(sc2.ids)):
        i, q, S = int(sc2.i[e]), int(sc2.q[e]), tuple(int(x) for x in sc2.s[e])
        sup = [idx3.get((i, q, tuple(sorted(S + (j,))))) for j in range(D) if j != q and j not in S]
        sup = [x for x in sup if x is not None]
        if not sup:
            continue
        sites.append(sc2.site[e])
        for m in methods:
            e2 = abs(sc2.preds[m][e] - sc2.truth[e])
            e3 = np.abs(sc3.preds[m][sup] - sc3.truth[sup]).mean()
            deltas[m].append(e2 - e3)                                  # positive: the third measurement helped
    out["base_episodes"] = len(sites)
    if not sites:
        return {"defined": False, "reason": "no paired two- and three-drug episodes"}
    sites = np.array(sites)
    for m in methods:
        d = np.array(deltas[m])
        site_means = np.array([d[sites == s].mean() for s in np.unique(sites)])
        out["methods"][m] = {"mean_error_reduction": float(np.nanmean(d)), "site_mean_error_reduction": float(np.nanmean(site_means)),
                             "fraction_helped": float(np.nanmean(d > 0))}
    return out


# --------------------------------------------------------------------------------------
# T4 - binding-mode strata (assay associations, not causal claims)
# --------------------------------------------------------------------------------------
def t4_modes(sc: Scored, methods: list, cfg: dict):
    mode = np.array(sc.modes)
    if len(set(mode.tolist())) <= 1 and (mode == "unspecified").all():
        return {"defined": False, "reason": "no inhibitor mode annotation for this panel"}
    match = (mode[sc.s] == mode[sc.q][:, None]).any(axis=1)
    out = {"defined": True, "strata": {}}
    for label, sel in (("support_matches_query_mode", match), ("no_matching_support", ~match)):
        if not sel.any():
            out["strata"][label] = {"observations": 0}
            continue
        entry = {"observations": int(sel.sum()), "sites": int(len(np.unique(sc.site[sel]))), "by_query": {}, "methods": {}}
        for m in methods:
            ok = sel & np.isfinite(sc.preds[m]) & np.isfinite(sc.preds[REFERENCE])
            if not ok.any():
                continue
            per_q = []
            for q in np.unique(sc.q[ok]):                              # compare within the same query inhibitor
                qs = ok & (sc.q == q)
                if qs.sum() < cfg["t4_min_episodes"]:
                    continue
                a, b = abs_err(sc, m)[qs].mean(), abs_err(sc, REFERENCE)[qs].mean()
                per_q.append({"query": sc.drugs[int(q)], "observations": int(qs.sum()), "mae_method": float(a),
                              "mae_xgboost": float(b), "gain_percent": None if b == 0 else float(100 * (b - a) / b)})
            gains = [r["gain_percent"] for r in per_q if r["gain_percent"] is not None]
            entry["by_query"][m] = per_q
            entry["methods"][m] = {"mean_gain_over_xgboost_across_queries": float(np.mean(gains)) if gains else None}
        out["strata"][label] = entry
    return out


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def evaluate(studies: dict, cfg: dict | None = None, interventions: dict | None = None, expected_studies: int | None = None):
    """``studies``: {panel: {k: Scored}}. Returns the full results dictionary."""
    cfg = {**EVAL_DEFAULTS, **(cfg or {})}
    draws = bootstrap_matrices(studies, cfg["bootstrap_draws"], cfg["bootstrap_seed"])
    methods = sorted({m for name in studies for sc in studies[name].values() for m in sc.preds})
    res = {"config": {k: v for k, v in cfg.items()}, "studies": sorted(studies), "methods": methods, "tables": {}}
    summaries, per_method = {}, {}
    for m in methods:
        if m == REFERENCE:
            continue
        rows, boot = study_table(studies, m, draws)
        per, pb = study_gains(rows, boot)
        summ = summarise_primary(per, pb, cfg, expected_studies)
        per_method[m] = (rows, per, pb)
        summaries[m] = {"panel_rows": rows, **summ}
    res["summaries"] = summaries
    if PRIMARY in per_method:
        rows, per, pb = per_method[PRIMARY]
        res["primary"] = {"method": PRIMARY, **summaries[PRIMARY]}
        res["win_tie"] = win_tie_table(rows, per, cfg)
        res["robustness"] = robustness(per, pb, studies, cfg)
        res["sensitivity"] = exact_and_interval_tables(studies, PRIMARY)
    fam = {m: summaries[m]["bootstrap_p_two_sided"] for m in CHECKPOINTS if m in summaries and summaries[m]["complete"]}
    if fam:
        res["secondary_holm"] = {"alpha": cfg["alpha"], "family": holm(fam, cfg["alpha"]),
                                 "note": "formal claims only if made; otherwise descriptive"}
    res["plain_baselines"] = plain_baseline_rows(studies, draws) if any(m in methods for m in DESCRIPTIVE) else {}
    res["intervention"] = intervention_report(studies, interventions or {}) if PRIMARY in methods else []
    eval_methods = [m for m in methods if m not in DESCRIPTIVE]
    t1, t2, t3, t4 = {}, {}, {}, {}
    for name in sorted(studies):
        for k in sorted(studies[name]):
            sc = studies[name][k]
            ms = [m for m in eval_methods if m in sc.preds]
            t1[f"{name}|k{k}"] = t1_screening(sc, ms, cfg)
            t2[f"{name}|k{k}"] = t2_choice(sc, ms, cfg)
            t4[f"{name}|k{k}"] = t4_modes(sc, ms, cfg)
        if 2 in studies[name] and 3 in studies[name]:
            t3[name] = t3_third_measurement(studies[name][2], studies[name][3], [m for m in eval_methods if m in studies[name][2].preds])
    res["T1_hidden_resistance"], res["T2_inhibitor_choice"], res["T3_third_measurement"], res["T4_binding_mode"] = t1, t2, t3, t4
    return res


def write_results(res: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=_default) + "\n", encoding="utf-8")
    rows = []
    for m, s in res["summaries"].items():
        for r in s["panel_rows"]:
            rows.append({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    if rows:
        with (out / "panel_budget_table.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def studies_from_runs(panels: dict, runs: list) -> tuple[dict, dict]:
    """Build Scored objects (scoring join) and per-fold intervention records from ``core_models.PanelRun`` objects."""
    studies, interventions = {}, {}
    for run in runs:
        panel = panels[run.panel]
        ep = dp.build_episodes(panel, run.k)
        if not np.array_equal(ep.ids, run.ids):
            raise ValueError(f"{run.panel} k={run.k}: predictions were not made on the registered episode manifest.")
        studies.setdefault(run.panel, {})[run.k] = score_runs(panel, ep, run.preds, run.failures)
        by_fold = {}
        for f in np.unique(run.folds):
            sel = run.folds == f
            by_fold[int(f)] = float(run.ser_intervened[sel].mean())
        interventions[(run.panel, run.k)] = by_fold
    return studies, interventions


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="folder with <panel>_k<k>_predictions.csv.gz files")
    p.add_argument("--archive-root", required=True)
    p.add_argument("--lockbox-b-root", default=None)
    p.add_argument("--roles", default="core_dev,stress_test")
    p.add_argument("--score-lockbox", action="store_true", help="required to join lockbox truth (curator step only)")
    p.add_argument("--out", default="results")
    p.add_argument("--draws", type=int, default=EVAL_DEFAULTS["bootstrap_draws"])
    p.add_argument("--registry", default=None)
    args = p.parse_args(argv)
    roles = args.roles.split(",")
    if set(roles) & set(dp.LOCKBOX_ROLES) and not args.score_lockbox:
        p.error("lockbox roles need --score-lockbox (scoring join is a curator step after the release freeze)")
    registry = dp.load_registry(args.registry)
    roots = {"archive": args.archive_root}
    if args.lockbox_b_root:
        roots["lockbox_b"] = args.lockbox_b_root
    studies = {}
    for name in dp.panels_by_role(registry, roles):
        panel = dp.load_panel(registry, name, roots, allow_lockbox=args.score_lockbox)
        for k in panel.budgets():
            f = Path(args.runs) / f"{name}_k{k}_predictions.csv.gz"
            if not f.exists():
                continue
            preds, failures, ids = load_predictions_csv(f)
            ep = dp.build_episodes(panel, k)
            if not np.array_equal(ids, ep.ids):
                raise SystemExit(f"{f.name}: episode ids do not match the registered manifest")
            studies.setdefault(name, {})[k] = score_runs(panel, ep, preds, failures)
    if not studies:
        raise SystemExit("no predictions found")
    res = evaluate(studies, {"bootstrap_draws": args.draws})
    write_results(res, Path(args.out))
    prim = res.get("primary", {})
    print("primary SER vs XGBoost:", {k: prim.get(k) for k in ("gain_percent", "ci95", "meets_target", "complete")})


if __name__ == "__main__":
    main()
