"""Step 3a - model roster: XGBoost reference, CovarianceLocalMedian and SER (baseline engine),
plus priors and the seven SAER experts. Routed pipelines (SAER-7g, AlignedAddAudit, NestedAligned)
live in ``core_models_routed`` section below and use ``routers.py``.

Conventions
-----------
* A *panel* is a ``data_pipeline.Panel``; an *episode set* is a ``data_pipeline.EpisodeSet``.
* Every method receives the same episodes and returns one prediction per episode (NaN plus a failure
  reason where the configuration is unsupported).
* Folds: outer test fold ``o``; inner validation folds are the other folds in turn.
* Hidden query responses are touched only through ``panel.y`` of *training or validation* variants.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import ndtr
from scipy.stats import t as student_t
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from xgboost import XGBRegressor

import data_pipeline as dp
import routers as rt

SEED = dp.SEED
METHODS = ("XGBoost", "SAER-7g", "AlignedAddAudit", "NestedAligned", "CovarianceLocalMedian", "SER")
CHECKPOINTS = ("SAER-7g", "AlignedAddAudit", "NestedAligned", "CovarianceLocalMedian")
DESCRIPTIVE = ("QueryMedian", "RidgeSupports")        # plain references reported descriptively (never primary)
ALL_METHODS = METHODS + DESCRIPTIVE
EXPERT_NAMES = ("prior_residual_median", "gaussian_transfer", "gaussian_mixture", "donor_transport",
                "pair_tree_median", "pair_tree_correlation", "joint_tree")
N_EXPERTS = 7

DEFAULT_CONFIG = {
    "seed": SEED,
    "xgb": {"eta": 0.055, "l2": 4.0, "seed": 20261112, "n_jobs": 1, "max_train_episodes": 36000,
            "grid": [[3, 200, 15], [5, 260, 15], [7, 300, 15]]},
    "neighbors": [16, 48, 96],
    "ridge_kappa": [0.1, 1.0, 10.0],
    "ser": {"weights": [0.25, 0.5, 0.75, 1.0], "confidence": 0.90, "min_validation_sites": 3,
            "experts": ["bounded_local", "shrunk_local", "bounded_ridge", "query_median", "pooled_median"]},
    "saer": {"joint_min_profiles": 1000, "joint_min_cells": 10000, "router_max_iter": 300,
             "gaussian_centers": ["zero", "mean", "median"], "gaussian_shrink": [0.01, 0.05, 0.25, 0.5, 0.9],
             "gaussian_mult": [0.25, 0.5, 0.75, 1.0], "mixture_components": [1, 2, 4, 8],
             "mixture_cov": ["tied", "full"], "mixture_reg": [0.05, 0.2],
             "donor_bandwidth": [0.1, 0.25, 0.5, 1, 2, 4], "donor_companion_weight": [0, 0.25, 1],
             "donor_pool": 256, "pair_tree_max_pairs": 150000, "pair_tree_capacity": [[100, 15], [300, 31], [600, 63]],
             "pair_tree_thresholds": [30000, 150000], "pair_lr": 0.06, "pair_min_leaf": 40, "pair_l2": 1.0,
             "joint_samples": [40000, 120000, 240000], "joint_capacity": [[150, 15], [250, 31], [400, 63]],
             "joint_cell_thresholds": [20000, 200000], "joint_lr": 0.06, "joint_min_leaf": 40, "joint_l2": 1.0,
             "context_model": {"max_iter": 150, "max_leaf_nodes": 15, "learning_rate": 0.07,
                               "l2_regularization": 10.0, "min_samples_leaf": 30},
             "select_sample": 1500, "aux_max_episodes": 20000, "aux_seed": 20261113},
    "aligned": {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 40,
                "l2_regularization": 1.0, "seed": 20261111, "nested_seed": 20261116,
                "multipliers": [0.25, 1.0, 4.0], "tau": [0.0, 0.25, 0.5, 1.0], "lambda": [0.0, 0.25, 0.5, 0.75, 1.0],
                "fallback_head": {"max_iter": 100, "max_leaf_nodes": 7, "min_samples_leaf": 120}},
    "nested_fixed": {"gaussian": {"center": "median", "shrink": 0.25, "mult": 0.75},
                     "mixture": {"n": 2, "cov": "tied", "reg": 0.2, "median": True},
                     "donor": {"bandwidth": 1.0, "companion_weight": 0.25, "median": True}},
}


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path and Path(path).exists():
        _deep_update(cfg, json.loads(Path(path).read_text(encoding="utf-8")))
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def _deep_update(base: dict, new: dict):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def rows_hash(rows) -> str:
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()[:16]


# ======================================================================================
# Generic helpers
# ======================================================================================
def weighted_median(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Smallest sorted value whose cumulative weight reaches 0.5 (weights sum to one per row)."""
    order = np.argsort(x, axis=1, kind="stable")
    cum = np.cumsum(np.take_along_axis(w, order, axis=1), axis=1)
    idx = np.argmax(cum >= 0.5 - 1e-12, axis=1)
    rows = np.arange(len(x))
    return x[rows, order[rows, idx]]


def interval_error(pred, truth, censor):
    """Distance from the prediction to the set of compatible responses (0 exact, +1 >= L, -1 <= U)."""
    resid = pred - truth
    return np.where(censor == 1, np.maximum(-resid, 0.0), np.where(censor == -1, np.maximum(resid, 0.0), np.abs(resid)))


def subsample_indices(idx: np.ndarray, cap: int, seed: int = SEED) -> np.ndarray:
    if len(idx) <= cap:
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))


# ======================================================================================
# XGBoost reference
# ======================================================================================
def xgb_features(panel: dp.Panel, ep: dp.EpisodeSet) -> np.ndarray:
    """Query id, sparse response vector (absent drugs missing), support ids, support responses in declared
    order, sorted values, mean, range, mode-compatible mean, compatibility count and flags."""
    i, q, s = ep.i, ep.q, ep.s
    E, k = s.shape
    obs = panel.y[i[:, None], s]
    layout = np.full((E, panel.n_drugs), np.nan)
    layout[np.arange(E)[:, None], s] = obs
    mode_names = {m: n for n, m in enumerate(dict.fromkeys(panel.modes))}
    mode = np.array([mode_names[m] for m in panel.modes])
    compat = mode[s] == mode[q][:, None]
    same_mode = (obs * compat).sum(1) / np.maximum(compat.sum(1), 1)
    raw = np.column_stack([q, layout, s, obs, np.sort(obs, axis=1), obs.mean(1), np.ptp(obs, axis=1), same_mode,
                           compat.sum(1), compat])
    return raw.astype(np.float32)


def make_xgb(row, cfg) -> XGBRegressor:
    depth, trees, leaf = row
    x = cfg["xgb"]
    return XGBRegressor(objective="reg:absoluteerror", max_depth=int(depth), n_estimators=int(trees),
                        learning_rate=x["eta"], min_child_weight=leaf, reg_lambda=x["l2"], n_jobs=x["n_jobs"],
                        random_state=x["seed"], tree_method="hist")


def fit_xgb(X, y, fit_idx, row, cfg) -> XGBRegressor:
    ix = subsample_indices(fit_idx, cfg["xgb"]["max_train_episodes"], cfg["xgb"]["seed"])
    return make_xgb(row, cfg).fit(X[ix], y[ix])


# ======================================================================================
# Covariance-adjusted local neighbourhoods (CovarianceLocalMedian and SER alternatives)
# ======================================================================================
def whiten(fit: np.ndarray, new: np.ndarray):
    """Training-only median/SD standardisation, then 0.85*Cov + 0.15*I whitening (Cholesky)."""
    center = np.median(fit, axis=0)
    scale = np.maximum(np.std(fit, axis=0), 1e-6)
    z, zn = (fit - center) / scale, (new - center) / scale
    cov = np.atleast_2d(np.cov(z.T))
    chol = np.linalg.cholesky(0.85 * cov + 0.15 * np.eye(z.shape[1]))
    rot = np.linalg.inv(chol.T)
    return z @ rot, zn @ rot


@dataclass
class Neighborhood:
    original: np.ndarray        # (E, nK) detrended weighted median
    shrunk: np.ndarray          # (E, nK) site-ESS shrunk slope (clipped)
    nodetrend: np.ndarray       # (E, nK) undetrended weighted median
    site_ess: np.ndarray        # (E, nK)
    eligible: np.ndarray        # (E,) eligible training profiles for the mask
    ok: np.ndarray              # (E,) bool - prediction available


def neighborhood(y: np.ndarray, train_rows: np.ndarray, train_sites: np.ndarray, i, q, s, Ks,
                 lo_hi: tuple | None = None) -> Neighborhood:
    """Covariance-adjusted neighbourhood predictions for episodes (i, q, S) using only ``train_rows``.

    Training profiles are those with finite responses on the mask's supports and query. The requested
    neighbourhood size is capped at the number of eligible training profiles.
    """
    E, k = s.shape
    nK = len(Ks)
    out = {key: np.full((E, nK), np.nan) for key in ("original", "shrunk", "nodetrend", "site_ess")}
    eligible = np.zeros(E, dtype=int)
    masks = np.column_stack([q, s])
    uniq, inv = np.unique(masks, axis=0, return_inverse=True)
    inv = inv.ravel()
    site_codes = np.unique(train_sites, return_inverse=True)[1]
    for g, mask in enumerate(uniq):
        take = np.flatnonzero(inv == g)
        query, sup = int(mask[0]), [int(v) for v in mask[1:]]
        block = y[np.ix_(train_rows, sup + [query])]
        ok = np.isfinite(block).all(axis=1)
        n_el = int(ok.sum())
        eligible[take] = n_el
        if n_el < k + 3:                                        # too few profiles for a covariance fit
            continue
        fit, target, sites = block[ok][:, :-1], block[ok][:, -1], site_codes[ok]
        point = y[np.ix_(i[take], sup)]
        a, b = whiten(fit, point)
        dist = cdist(b, a)
        order = np.argsort(dist, axis=1, kind="stable")[:, :min(max(Ks), n_el)]
        lo, hi = (target.min(), target.max()) if lo_hi is None else (lo_hi[0][query], lo_hi[1][query])
        for j, K in enumerate(Ks):
            near = order[:, :min(K, n_el)]
            d = np.take_along_axis(dist, near, axis=1)
            w = np.exp(-2.0 * (d / np.maximum(d[:, -1:], 1e-8)) ** 2)
            w /= w.sum(axis=1, keepdims=True)
            resp = target[near]
            delta = a[near] - b[:, None]
            dx = np.einsum("nk,nkf->nf", w, delta)
            ybar = np.einsum("nk,nk->n", w, resp)
            xc = delta - dx[:, None]
            gram = np.einsum("nk,nkf,nkg->nfg", w, xc, xc)
            gram += (0.03 * np.trace(gram, axis1=1, axis2=2) + 0.002)[:, None, None] * np.eye(a.shape[1])
            rhs = np.einsum("nk,nkf,nk->nf", w, xc, resp - ybar[:, None])
            beta = np.linalg.solve(gram, rhs[..., None])[..., 0]
            trend = np.einsum("nkf,nf->nk", delta, beta)
            site_mat = np.eye(int(sites.max()) + 1)[sites]
            Wg = np.einsum("nk,nkp->np", w, site_mat[near])
            neff = 1.0 / np.square(Wg).sum(axis=1)
            gamma = np.maximum(0.0, 1.0 - (k + 1) / neff)
            out["original"][take, j] = weighted_median(resp - trend, w)
            out["shrunk"][take, j] = np.clip(weighted_median(resp - gamma[:, None] * trend, w), lo, hi)
            out["nodetrend"][take, j] = weighted_median(resp, w)
            out["site_ess"][take, j] = neff
    ok_any = np.isfinite(out["original"]).all(axis=1)
    return Neighborhood(out["original"], out["shrunk"], out["nodetrend"], out["site_ess"], eligible, ok_any)


def ridge_predictions(y, train_rows, i, q, s, kappas) -> np.ndarray:
    """Ridge on support columns scaled by fitting SDs with an unpenalised intercept (unclipped).

    Penalty selection uses these raw predictions; the SER alternative clips them to the fitting query range."""
    E, k = s.shape
    out = np.full((E, len(kappas)), np.nan)
    masks = np.column_stack([q, s])
    uniq, inv = np.unique(masks, axis=0, return_inverse=True)
    inv = inv.ravel()
    for g, mask in enumerate(uniq):
        take = np.flatnonzero(inv == g)
        query, sup = int(mask[0]), [int(v) for v in mask[1:]]
        block = y[np.ix_(train_rows, sup + [query])]
        ok = np.isfinite(block).all(axis=1)
        if ok.sum() < k + 2:
            continue
        fit, target = block[ok][:, :-1], block[ok][:, -1]
        scale = np.maximum(fit.std(axis=0), 1e-6)
        point = y[np.ix_(i[take], sup)]
        for j, kappa in enumerate(kappas):
            model = Ridge(alpha=kappa).fit(fit / scale, target)
            out[take, j] = model.predict(point / scale)
    return out


def train_range(y, train_rows):
    block = y[train_rows]
    return np.nanmin(block, axis=0), np.nanmax(block, axis=0)


def simple_locations(y, train_rows, q):
    block = y[train_rows]
    with np.errstate(all="ignore"):
        per_drug = np.nanmedian(block, axis=0)
    pooled = np.nanmedian(block)
    per_drug = np.where(np.isfinite(per_drug), per_drug, pooled)
    return per_drug[q], np.full(len(q), pooled)


# ======================================================================================
# SER gate
# ======================================================================================
def site_lower_score(gain: np.ndarray, sites: np.ndarray, confidence=0.90, min_sites=3) -> float:
    """Q = mean gain - t_{0.90,G-1} * clustered SE; -inf if fewer than 3 sites or leave-one-site-out fails."""
    groups, inv = np.unique(sites, return_inverse=True)
    if len(groups) < min_sites or len(gain) == 0:
        return float("-inf")
    counts = np.bincount(inv)
    sums = np.bincount(inv, weights=gain)
    total = sums.sum()
    if (total - sums <= 0).any():
        return float("-inf")
    mean = total / len(gain)
    resid = sums - mean * counts
    se = np.linalg.norm(resid) * math.sqrt(len(groups) / (len(groups) - 1)) / len(gain)
    return float(mean - student_t.ppf(confidence, len(groups) - 1) * se)


def ser_gate(truth, anchor, alternatives: dict, sites, weights, confidence, min_sites, observed_limit=None,
             query_censor=None, censored_panel=False) -> dict:
    """Largest positive Q over (alternative, weight) in fixed order; ties keep the earlier entry."""
    best = {"expert": "XGBoost", "weight": 0.0, "selection_score": 0.0, "validation_sites": int(len(np.unique(sites)))}
    ref_abs = np.abs(anchor - truth)
    if censored_panel:
        exact = ~observed_limit & (query_censor == 0)
        ref_int = interval_error(anchor, truth, query_censor)
    for name, alt in alternatives.items():
        alt = np.where(np.isfinite(alt), alt, anchor)
        for w in weights:
            pred = anchor + w * (alt - anchor)
            if censored_panel:
                pred = np.where(observed_limit, anchor, pred)
                reported = ref_abs - np.abs(pred - truth)
                interval = ref_int - interval_error(pred, truth, query_censor)
                scores = [site_lower_score(reported, sites, confidence, min_sites),
                          site_lower_score(interval, sites, confidence, min_sites),
                          site_lower_score(reported[exact], sites[exact], confidence, min_sites)]
                score = min(scores)
            else:
                score = site_lower_score(ref_abs - np.abs(pred - truth), sites, confidence, min_sites)
            if score > best["selection_score"]:
                best = {"expert": name, "weight": float(w), "selection_score": float(score),
                        "validation_sites": int(len(np.unique(sites)))}
    return best


# ======================================================================================
# Baseline engine: XGBoost reference, CovarianceLocalMedian and SER on the same nested folds
# ======================================================================================
@dataclass
class BaselineOutput:
    xgb: np.ndarray
    local: np.ndarray
    ser: np.ndarray
    ser_intervened: np.ndarray            # bool per episode: SER prediction differs from XGBoost
    qmed: np.ndarray                      # descriptive baseline: training query median
    ridge: np.ndarray                     # descriptive baseline: ridge on supports (clipped)
    decisions: list
    selected: list
    failures: dict = field(default_factory=dict)
    inner: dict = field(default_factory=dict)      # outer fold -> inner validation records (for diagnostics)


def run_baselines(panel: dp.Panel, ep: dp.EpisodeSet, cfg: dict, log=None) -> BaselineOutput:
    """Nested residue-group evaluation of the XGBoost reference, CovarianceLocalMedian and SER.

    For every outer fold: each other fold is held out in turn (inner validation) and the remaining
    groups fit XGBoost (3 grid rows), local neighbourhoods (3 sizes) and ridge (3 penalties). Settings are
    selected on aggregated inner total absolute error, the selected XGBoost row generates the gate's inner
    predictions, and everything is refit on the outer training groups before predicting the outer fold.
    """
    y = panel.y
    folds_v = panel.folds
    ef = folds_v[ep.i]
    X = xgb_features(panel, ep)
    truth = y[ep.i, ep.q]
    censor_q = panel.censor[ep.i, ep.q]
    obs_limit = (panel.censor[ep.i[:, None], ep.s] != 0).any(axis=1)
    censored_panel = bool((panel.censor != 0).any())
    sites_v = np.array([dp.site_key(s) for s in panel.sites], dtype=object)
    sites_e = sites_v[ep.i]
    xgb_pred = np.full(len(ep), np.nan)
    local_pred = np.full(len(ep), np.nan)
    ser_pred = np.full(len(ep), np.nan)
    qmed_pred = np.full(len(ep), np.nan)
    ridge_pred = np.full(len(ep), np.nan)
    intervened = np.zeros(len(ep), dtype=bool)
    decisions, selected_log, failures = [], [], {"CovarianceLocalMedian": {}, "SER": {}}
    Ks, kappas = cfg["neighbors"], cfg["ridge_kappa"]
    grid = cfg["xgb"]["grid"]
    min_sites = cfg["ser"]["min_validation_sites"]
    for outer in np.unique(folds_v):
        outer_test = np.flatnonzero(ef == outer)
        assert not (set(sites_v[folds_v == outer]) & set(sites_v[folds_v != outer])), "residue separation violated"
        inner_folds = [f for f in np.unique(folds_v) if f != outer]
        store = []                                                  # per inner fold records
        for inner in inner_folds:
            train_rows = np.flatnonzero((folds_v != outer) & (folds_v != inner))
            val = np.flatnonzero(ef == inner)
            if len(val) == 0 or len(train_rows) < 3:
                continue
            train_ep = np.flatnonzero(np.isin(ep.i, train_rows))
            xgb_all = np.column_stack([fit_xgb(X, truth, train_ep, row, cfg).predict(X[val]) for row in grid])
            lo_hi = train_range(y, train_rows)
            nb = neighborhood(y, train_rows, sites_v[train_rows], ep.i[val], ep.q[val], ep.s[val], Ks, lo_hi)
            rid = ridge_predictions(y, train_rows, ep.i[val], ep.q[val], ep.s[val], kappas)
            qmed, pmed = simple_locations(y, train_rows, ep.q[val])
            store.append({"val": val, "xgb": xgb_all, "nb": nb, "ridge": rid, "qmed": qmed, "pmed": pmed,
                          "lo_hi": lo_hi})
        if not store:
            failures["SER"][int(outer)] = "no_inner_validation"
            continue
        val_all = np.concatenate([r["val"] for r in store])
        t_val = truth[val_all]
        stack = lambda key: np.concatenate([r[key] for r in store])
        xgb_stack = stack("xgb")
        best_x = int(np.argmin(np.abs(xgb_stack - t_val[:, None]).sum(axis=0)))
        orig_stack = np.concatenate([r["nb"].original for r in store])
        loc_loss = np.nansum(np.abs(orig_stack - t_val[:, None]), axis=0) + 1e9 * (~np.isfinite(orig_stack)).sum(axis=0)
        best_k = int(np.argmin(loc_loss))
        rid_stack = stack("ridge")
        rid_loss = np.nansum(np.abs(rid_stack - t_val[:, None]), axis=0) + 1e9 * (~np.isfinite(rid_stack)).sum(axis=0)
        best_r = int(np.argmin(rid_loss))
        selected_log.append({"outer_fold": int(outer), "xgb_row": grid[best_x], "neighbors": Ks[best_k],
                             "ridge_kappa": kappas[best_r]})
        # ---------------- SER gate on aggregated inner validation ----------------
        anchor = xgb_stack[:, best_x]
        nb_orig = orig_stack[:, best_k]
        nb_shr = np.concatenate([r["nb"].shrunk for r in store])[:, best_k]
        lo = np.concatenate([r["lo_hi"][0][ep.q[r["val"]]] for r in store])
        hi = np.concatenate([r["lo_hi"][1][ep.q[r["val"]]] for r in store])
        alts = {"bounded_local": np.clip(nb_orig, lo, hi), "shrunk_local": nb_shr,
                "bounded_ridge": np.clip(rid_stack[:, best_r], lo, hi), "query_median": stack("qmed"), "pooled_median": stack("pmed")}
        alts = {n: alts[n] for n in cfg["ser"]["experts"]}
        choice = ser_gate(t_val, anchor, alts, sites_e[val_all], cfg["ser"]["weights"], cfg["ser"]["confidence"],
                          min_sites, obs_limit[val_all], censor_q[val_all], censored_panel)
        decisions.append({"fold": int(outer), **choice, "inner_episodes": int(len(val_all))})
        # ---------------- outer refit and prediction ----------------
        train_rows = np.flatnonzero(folds_v != outer)
        if len(outer_test) == 0 or len(train_rows) < 3:
            continue
        train_ep = np.flatnonzero(np.isin(ep.i, train_rows))
        xgb_pred[outer_test] = fit_xgb(X, truth, train_ep, grid[best_x], cfg).predict(X[outer_test])
        lo_hi = train_range(y, train_rows)
        nb = neighborhood(y, train_rows, sites_v[train_rows], ep.i[outer_test], ep.q[outer_test], ep.s[outer_test],
                          [Ks[best_k]], lo_hi)
        local_pred[outer_test] = nb.original[:, 0]
        qmed_pred[outer_test] = simple_locations(y, train_rows, ep.q[outer_test])[0]
        ridge_pred[outer_test] = np.clip(
            ridge_predictions(y, train_rows, ep.i[outer_test], ep.q[outer_test], ep.s[outer_test], [kappas[best_r]])[:, 0],
            lo_hi[0][ep.q[outer_test]], lo_hi[1][ep.q[outer_test]])
        for e in np.flatnonzero(~nb.ok):
            failures["CovarianceLocalMedian"][int(outer_test[e])] = "insufficient_eligible_training_profiles"
        ser_pred[outer_test] = xgb_pred[outer_test]
        if choice["expert"] != "XGBoost":
            test = outer_test[~obs_limit[outer_test]]               # censored supports keep XGBoost
            if len(test):
                name = choice["expert"]
                if name in ("bounded_local", "shrunk_local"):
                    nb_t = neighborhood(y, train_rows, sites_v[train_rows], ep.i[test], ep.q[test], ep.s[test],
                                        [Ks[best_k]], lo_hi)
                    alt = np.clip(nb_t.original[:, 0], lo_hi[0][ep.q[test]], lo_hi[1][ep.q[test]]) \
                        if name == "bounded_local" else nb_t.shrunk[:, 0]
                elif name == "bounded_ridge":
                    alt = np.clip(ridge_predictions(y, train_rows, ep.i[test], ep.q[test], ep.s[test], [kappas[best_r]])[:, 0],
                                  lo_hi[0][ep.q[test]], lo_hi[1][ep.q[test]])
                elif name == "query_median":
                    alt = simple_locations(y, train_rows, ep.q[test])[0]
                else:
                    alt = simple_locations(y, train_rows, ep.q[test])[1]
                base = xgb_pred[test]
                alt = np.where(np.isfinite(alt), alt, base)
                ser_pred[test] = base + choice["weight"] * (alt - base)
    intervened = np.isfinite(ser_pred) & np.isfinite(xgb_pred) & (ser_pred != xgb_pred)
    return BaselineOutput(xgb_pred, local_pred, ser_pred, intervened, qmed_pred, ridge_pred, decisions, selected_log, failures)


# ======================================================================================
# Priors (permitted per-panel recipe)
# ======================================================================================
_PRIOR_CACHE: dict = {}
_TREE_CACHE: dict = {}


def clear_caches():
    _PRIOR_CACHE.clear()
    _TREE_CACHE.clear()


def _median_prior(panel: dp.Panel, fit_rows) -> np.ndarray:
    """Training-only per-drug medians with a pooled-median fallback."""
    block = panel.y[fit_rows]
    with np.errstate(all="ignore"):
        per_drug = np.nanmedian(block, axis=0)
    pooled = np.nanmedian(block)
    per_drug = np.where(np.isfinite(per_drug), per_drug, pooled)
    return np.broadcast_to(per_drug[None, :], panel.y.shape).copy()


def _context_prior(panel: dp.Panel, fit_rows, cfg) -> np.ndarray:
    """pi_ij = W_ij + H_j(context_i); H_j is fitted by query inhibitor on training groups and
    cross-fitted (grouped folds) for the training rows themselves."""
    c = cfg["saer"]["context_model"]
    X, W, Y = panel.context, panel.companion, panel.y
    sites = np.array([dp.site_key(s) for s in panel.sites], dtype=object)
    cv = None
    if len(set(sites[fit_rows])) >= 2:
        cv = dp.fold_assignment(sites[fit_rows], max_folds=5, min_groups=2)

    def model():
        return HistGradientBoostingRegressor(max_iter=c["max_iter"], max_leaf_nodes=c["max_leaf_nodes"],
                                             learning_rate=c["learning_rate"], l2_regularization=c["l2_regularization"],
                                             min_samples_leaf=c["min_samples_leaf"], early_stopping=False,
                                             random_state=SEED)
    pi = np.empty_like(Y)
    for j in range(panel.n_drugs):
        target = Y[:, j] - W[:, j]
        okf = np.isfinite(target[fit_rows])
        full = model().fit(X[fit_rows][okf], target[fit_rows][okf])
        pi[:, j] = W[:, j] + full.predict(X)
        if cv is not None:
            for f in np.unique(cv):
                tr = fit_rows[(cv != f) & okf]
                te = fit_rows[cv == f]
                if len(tr) >= 2:
                    pi[te, j] = W[te, j] + model().fit(X[tr], target[tr]).predict(X[te])
    return pi


def compute_prior(panel: dp.Panel, fit_rows, cfg, allow_companion=True) -> np.ndarray:
    fit_rows = np.asarray(fit_rows)
    use_ctx = (allow_companion and panel.spec.get("prior") == "companion_context"
               and panel.companion is not None and panel.context is not None)
    key = (panel.name, rows_hash(fit_rows), use_ctx)
    if key not in _PRIOR_CACHE:
        _PRIOR_CACHE[key] = _context_prior(panel, fit_rows, cfg) if use_ctx else _median_prior(panel, fit_rows)
    return _PRIOR_CACHE[key]


# ======================================================================================
# SAER experts
# ======================================================================================
def _solve(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        jitter = 1e-6 * np.trace(A, axis1=-2, axis2=-1)[..., None, None] * np.eye(A.shape[-1])
        return np.linalg.solve(A + jitter, b)


class SafeHGB:
    """HistGradientBoostingRegressor that drops all-missing feature columns (fit and predict consistently)."""

    def __init__(self, **params):
        self.params = params
        self.keep = None
        self.model = None
        self.categorical_index = params.pop("categorical_index", None)

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        self.keep = np.isfinite(X).any(axis=0)
        params = dict(self.params)
        if self.categorical_index is not None and self.keep[self.categorical_index]:
            params["categorical_features"] = [int(self.keep[: self.categorical_index].sum())]
        self.model = HistGradientBoostingRegressor(**params).fit(X[:, self.keep], y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(np.asarray(X, dtype=float)[:, self.keep])


def masked_corr(Rt: np.ndarray) -> np.ndarray:
    """Pairwise-complete correlation of residual columns (NaN -> 0, unit diagonal)."""
    M = np.isfinite(Rt).astype(float)
    Z = np.where(M > 0, Rt - 0.0, 0.0)
    with np.errstate(all="ignore"):
        mean = (Z * M).sum(0) / np.maximum(M.sum(0), 1)
    Z = np.where(M > 0, Rt - mean, 0.0)
    num = Z.T @ Z
    ss = (Z ** 2).T @ M
    den = np.sqrt(ss * ss.T)
    with np.errstate(all="ignore"):
        rho = np.where(den > 1e-12, num / den, 0.0)
    np.fill_diagonal(rho, 1.0)
    return np.clip(rho, -1.0, 1.0)


class ExpertBank:
    """Seven experts fitted on one fitting partition (``rows``) of a panel.

    ``settings`` fixes the tuned experts' configuration (Gaussian, mixture, donor); ``None`` uses the
    reference defaults until ``select_settings`` is called. Trees are budget-agnostic (they learn from
    zero to three observed supports) and are cached per fitting partition.
    """

    def __init__(self, panel: dp.Panel, rows, cfg: dict, settings: dict | None = None, allow_companion: bool = True):
        self.panel, self.cfg = panel, cfg
        self.rows = np.asarray(rows)
        self.D = panel.n_drugs
        sc = cfg["saer"]
        cells = int(np.isfinite(panel.y).sum())
        self.use_joint = panel.n >= sc["joint_min_profiles"] or cells >= sc["joint_min_cells"]
        self.cells = cells
        self.prior = compute_prior(panel, self.rows, cfg, allow_companion)
        self.companion = panel.companion if (allow_companion and panel.spec.get("prior") == "companion_context") else None
        self.R = panel.y - self.prior
        Rt = self.R[self.rows]
        with np.errstate(all="ignore"):
            pooled_med = np.nanmedian(Rt)
            self.mu_med = np.nanmedian(Rt, axis=0)
            self.mu_mean = np.nanmean(Rt, axis=0)
            sd = np.nanstd(Rt, axis=0)
        self.pooled_med = float(pooled_med)
        self.mu_med = np.where(np.isfinite(self.mu_med), self.mu_med, pooled_med)
        self.mu_mean = np.where(np.isfinite(self.mu_mean), self.mu_mean, 0.0)
        pooled_sd = float(np.nanstd(Rt)) or 1.0
        self.sd_col = np.maximum(np.where(np.isfinite(sd) & (sd > 0), sd, pooled_sd), 1e-6)
        self.s = max(float(np.nanstd(Rt)), rt.SCALE_FLOOR)              # fold training residual SD (scalar)
        self.rho = masked_corr(Rt)
        self.settings = settings or {
            "gaussian": {"center": "median", "shrink": 0.25, "mult": 0.75},
            "mixture": {"n": 2, "cov": "tied", "reg": 0.2, "median": True},
            "donor": {"bandwidth": 1.0, "companion_weight": 0.25, "median": True}}
        self._sigma0, self._gm, self._tree = {}, {}, {}
        self._pool = None

    # ---- expert 1 -----------------------------------------------------------------
    def base(self, i, q):
        return self.prior[i, q] + self.mu_med[q]

    # ---- expert 2 -----------------------------------------------------------------
    def _center(self, name):
        return {"zero": np.zeros(self.D), "mean": self.mu_mean, "median": self.mu_med}[name]

    def _gauss_params(self, center, shrink):
        key = center
        if key not in self._sigma0:
            Rt = self.R[self.rows]
            Z = np.where(np.isfinite(Rt), Rt - self._center(center), 0.0)
            self._sigma0[key] = Z.T @ Z / max(len(self.rows) - 1, 1)
        S0 = self._sigma0[key]
        diag = np.maximum(np.diag(S0), 1e-8)
        Sig = (1 - shrink) * S0 + shrink * np.diag(diag)
        Sig[np.diag_indices(self.D)] = np.maximum(np.diag(Sig), 1e-8)
        return self._center(center), Sig

    def gaussian(self, i, q, S, st=None):
        st = st or self.settings["gaussian"]
        mu, Sig = self._gauss_params(st["center"], st["shrink"])
        RS = self.R[i[:, None], S] - mu[S]
        coef = _solve(Sig[S[:, :, None], S[:, None, :]], Sig[q[:, None], S][..., None])[..., 0]
        return self.prior[i, q] + mu[q] + st["mult"] * (coef * RS).sum(1)

    # ---- expert 3 -----------------------------------------------------------------
    def _mixture(self, st):
        key = (st["n"], st["cov"], st["reg"])
        if key not in self._gm:
            Rt = self.R[self.rows]
            mean = self.mu_mean
            Z = np.where(np.isfinite(Rt), (Rt - mean) / self.sd_col, 0.0)
            n = int(min(st["n"], max(1, len(self.rows) // 4)))
            gm = GaussianMixture(n_components=n, covariance_type=st["cov"], reg_covar=st["reg"],
                                 random_state=SEED, max_iter=100).fit(Z)
            covs = np.broadcast_to(gm.covariances_, (n,) + gm.covariances_.shape) if st["cov"] == "tied" else gm.covariances_
            self._gm[key] = (gm.weights_, gm.means_, covs)
        return self._gm[key]

    def mixture(self, i, q, S, st=None):
        st = st or self.settings["mixture"]
        w0, means, covs = self._mixture(st)
        zS = (self.R[i[:, None], S] - self.mu_mean[S]) / self.sd_col[S]
        E, m = S.shape
        logw, mc, vc = [], [], []
        for c in range(len(w0)):
            Sig, mu = covs[c], means[c]
            SSS, SqS = Sig[S[:, :, None], S[:, None, :]], Sig[q[:, None], S]
            coef = _solve(SSS, SqS[..., None])[..., 0]
            dev = zS - mu[S]
            sol = _solve(SSS, dev[..., None])[..., 0]
            quad = (dev * sol).sum(1)
            logdet = np.linalg.slogdet(SSS)[1]
            logw.append(np.log(w0[c]) - 0.5 * (m * math.log(2 * math.pi) + logdet + quad))
            mc.append(mu[q] + (coef * dev).sum(1))
            vc.append(np.maximum(Sig[q, q] - (coef * SqS).sum(1), 1e-9))
        logw, mc, vc = np.stack(logw, 1), np.stack(mc, 1), np.stack(vc, 1)
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        if not st["median"]:
            z = (w * mc).sum(1)
        else:
            sd = np.sqrt(vc)
            lo, hi = (mc - 8 * sd).min(1), (mc + 8 * sd).max(1)
            for _ in range(rt.LAPLACE_ITERS):
                mid = 0.5 * (lo + hi)
                cdf = (w * ndtr((mid[:, None] - mc) / sd)).sum(1)
                lo, hi = np.where(cdf < 0.5, mid, lo), np.where(cdf < 0.5, hi, mid)
            z = 0.5 * (lo + hi)
        return self.prior[i, q] + self.mean_col(q) + self.sd_col[q] * z

    def mean_col(self, q):
        return self.mu_mean[q]

    # ---- expert 4 -----------------------------------------------------------------
    def _donor_pool(self):
        if self._pool is None:
            labels = [self.panel.labels[r] for r in self.rows]
            order = sorted(range(len(self.rows)), key=lambda t: hashlib.sha256(labels[t].encode()).digest())
            take = self.rows[order[: self.cfg["saer"]["donor_pool"]]]
            Wz = None
            if self.companion is not None:
                Wt = self.companion[self.rows]
                Wz = np.nan_to_num((self.companion - Wt.mean(0)) / np.maximum(Wt.std(0), 1e-6))
            self._pool = (take, self.R[take], Wz)
        return self._pool

    def donor(self, i, q, S, st=None):
        st = st or self.settings["donor"]
        take, Rp, Wz = self._donor_pool()
        mu, Sig = self._gauss_params("median", 0.25)
        h, lam_ref = st["bandwidth"], st["companion_weight"]
        fallback = self.gaussian(i, q, S, {"center": "median", "shrink": 0.25, "mult": 1.0})
        out = fallback.copy()
        E, m = S.shape
        for lo in range(0, E, 1500):
            sl = slice(lo, min(E, lo + 1500))
            ii, qq, SS = i[sl], q[sl], S[sl]
            RS_i = self.R[ii[:, None], SS]                                  # (e, m)
            RS_j = np.transpose(Rp[:, SS], (1, 0, 2))                        # (e, P, m)
            Rq_j = Rp[:, qq].T                                               # (e, P)
            valid = np.isfinite(Rq_j) & np.isfinite(RS_j).all(axis=2)
            diff = RS_i[:, None, :] - np.where(np.isfinite(RS_j), RS_j, 0.0)
            SSS = Sig[SS[:, :, None], SS[:, None, :]]
            inv = np.linalg.inv(SSS + 1e-12 * np.eye(m))
            d = np.einsum("epm,emn,epn->ep", diff, inv, diff) / m
            beta = np.einsum("em,emn->en", Sig[qq[:, None], SS], inv)
            tj = (self.prior[ii, qq][:, None] + np.where(np.isfinite(Rq_j), Rq_j, 0.0)
                  + np.einsum("en,epn->ep", beta, diff))
            logit = -0.5 * d / h
            if lam_ref and Wz is not None:
                dref = (np.square(Wz[ii]).sum(1)[:, None] + np.square(Wz[take]).sum(1)[None, :]
                        - 2 * Wz[ii] @ Wz[take].T) / self.D
                logit = logit - lam_ref * dref
            logit = np.where(valid, logit, -np.inf)
            has = valid.any(axis=1)
            logit = np.where(has[:, None], logit, 0.0)
            w = np.exp(logit - logit.max(axis=1, keepdims=True)) * np.where(has[:, None], valid, 1.0)
            w /= w.sum(axis=1, keepdims=True)
            est = weighted_median(tj, w) if st["median"] else (w * tj).sum(1)
            out[sl] = np.where(has, est, fallback[sl])
        return out

    # ---- experts 5 and 6: shared pairwise tree --------------------------------------
    def _pair_features(self, i, q, j):
        z = (self.R[i, j] - self.mu_med[j]) / self.sd_col[j]
        if self.companion is not None:
            wq, wj = self.companion[i, q], self.companion[i, j]
        else:
            wq = wj = np.full(len(i), np.nan)
        pq, pj = self.prior[i, q], self.prior[i, j]
        rho = self.rho[q, j]
        return np.column_stack([z, wq, wj, wq - wj, pq, pj, pq - pj, self.mu_med[q], self.mu_med[j], self.sd_col[q],
                                self.sd_col[j], rho, rho * z])

    def _pair_tree(self):
        key = ("pair", self.panel.name, rows_hash(self.rows), self.D)
        if key not in _TREE_CACHE:
            sc = self.cfg["saer"]
            Rt = self.R[self.rows]
            fin = np.isfinite(Rt)
            r, a, b = np.nonzero(fin[:, :, None] & fin[:, None, :] & ~np.eye(self.D, dtype=bool)[None])
            rng = np.random.default_rng(SEED + 1)
            if len(r) > sc["pair_tree_max_pairs"]:
                pick = np.sort(rng.choice(len(r), sc["pair_tree_max_pairs"], replace=False))
                r, a, b = r[pick], a[pick], b[pick]
            rows = self.rows[r]
            X = self._pair_features(rows, a, b)
            target = (self.R[rows, a] - self.mu_med[a]) / self.sd_col[a]
            lo, hi = sc["pair_tree_thresholds"]
            it, leaves = sc["pair_tree_capacity"][0 if len(r) < lo else 1 if len(r) < hi else 2]
            _TREE_CACHE[key] = SafeHGB(
                loss="absolute_error", max_iter=it, max_leaf_nodes=leaves, learning_rate=sc["pair_lr"],
                min_samples_leaf=sc["pair_min_leaf"], l2_regularization=sc["pair_l2"], early_stopping=False,
                random_state=SEED).fit(X, target)
        return _TREE_CACHE[key]

    def pair_trees(self, i, q, S):
        """Return (expert 5 median aggregate, expert 6 correlation-weighted aggregate)."""
        E, m = S.shape
        tree = self._pair_tree()
        ii, qq = np.repeat(i, m), np.repeat(q, m)
        jj = S.reshape(-1)
        h = tree.predict(self._pair_features(ii, qq, jj)).reshape(E, m)
        u = self.mu_med[q][:, None] + self.sd_col[q][:, None] * h                # residual proposals per support
        w = np.abs(self.rho[q[:, None], S]) + 0.001
        p5 = self.prior[i, q] + np.median(u, axis=1)
        p6 = self.prior[i, q] + (w * u).sum(1) / w.sum(1)
        return p5, p6

    # ---- expert 7: joint-support tree ------------------------------------------------
    def _joint_features(self, i, q, S, valid):
        """S (E, 3) support indices padded with 0, valid (E, 3) bool; supports ordered by decreasing |rho_q,j|."""
        E = len(i)
        rho = np.where(valid, self.rho[q[:, None], S], 0.0)
        key = np.where(valid, -np.abs(rho), 1.0)
        order = np.argsort(key, axis=1, kind="stable")
        S, valid, rho = (np.take_along_axis(a, order, axis=1) for a in (S, valid, rho))
        z = np.where(valid, (self.R[i[:, None], S] - self.mu_med[S]) / self.sd_col[S], np.nan)
        pr = np.where(valid, self.prior[i[:, None], S], np.nan)
        rho_n = np.where(valid, rho, np.nan)
        wabs = np.where(valid, np.abs(rho) + 0.001, 0.0)
        with np.errstate(all="ignore"):
            summary = np.where(valid.any(1), (np.nan_to_num(z) * wabs).sum(1) / np.maximum(wabs.sum(1), 1e-12), 0.0)
        blocks = np.stack([z, rho_n, pr], axis=2).reshape(E, -1)
        return np.column_stack([q, self.prior[i, q], self.mu_med[q], self.sd_col[q], valid.sum(1), summary, blocks])

    def _joint_tree(self):
        key = ("joint", self.panel.name, rows_hash(self.rows), self.D)
        if key not in _TREE_CACHE:
            sc = self.cfg["saer"]
            lo, hi = sc["joint_cell_thresholds"]
            cap = 0 if self.cells < lo else 1 if self.cells < hi else 2
            n_samples = sc["joint_samples"][cap]
            it, leaves = sc["joint_capacity"][cap]
            Rt = self.R[self.rows]
            fin = np.isfinite(Rt)
            rng = np.random.default_rng(SEED + 2)
            Xs, ys = [], []
            done = 0
            while done < n_samples:
                n = min(20000, n_samples - done)
                r = rng.integers(len(self.rows), size=n)
                keys = np.where(fin[r], rng.random((n, self.D)), 2.0)
                order = np.argsort(keys, axis=1)[:, :4]
                nf = fin[r].sum(1)
                m = np.minimum(rng.integers(0, 4, size=n), nf - 1)
                ok = m >= 0
                r, order, m = r[ok], order[ok], m[ok]
                q = order[:, 0]
                S = np.where(np.arange(3)[None, :] < m[:, None], order[:, 1:4], 0)
                valid = np.arange(3)[None, :] < m[:, None]
                rows = self.rows[r]
                Xs.append(self._joint_features(rows, q, S, valid))
                ys.append(self.R[rows, q])
                done += n
            X, y = np.concatenate(Xs), np.concatenate(ys)
            _TREE_CACHE[key] = SafeHGB(
                loss="absolute_error", max_iter=it, max_leaf_nodes=leaves, learning_rate=sc["joint_lr"],
                min_samples_leaf=sc["joint_min_leaf"], l2_regularization=sc["joint_l2"], early_stopping=False,
                categorical_index=0 if self.D <= 255 else None, random_state=SEED).fit(X, y)
        return _TREE_CACHE[key]

    def joint(self, i, q, S):
        E, m = S.shape
        Sp = np.zeros((E, 3), dtype=int)
        Sp[:, :m] = S
        valid = np.zeros((E, 3), dtype=bool)
        valid[:, :m] = True
        return self.prior[i, q] + self._joint_tree().predict(self._joint_features(i, q, Sp, valid))

    # ---- all experts -------------------------------------------------------------------
    def predict(self, i, q, S, settings: dict | None = None) -> np.ndarray:
        """Proposals (E, 7) for episodes (i, q, S) of any support size 1..3; column 7 is NaN if unused."""
        st = settings or self.settings
        i, q, S = np.asarray(i), np.asarray(q), np.asarray(S)
        out = np.full((len(i), N_EXPERTS), np.nan)
        out[:, 0] = self.base(i, q)
        out[:, 1] = self.gaussian(i, q, S, st["gaussian"])
        out[:, 2] = self.mixture(i, q, S, st["mixture"])
        out[:, 3] = self.donor(i, q, S, st["donor"])
        out[:, 4], out[:, 5] = self.pair_trees(i, q, S)
        if self.use_joint:
            out[:, 6] = self.joint(i, q, S)
        return out

    @property
    def avail(self) -> np.ndarray:
        a = np.ones(N_EXPERTS, dtype=bool)
        a[6] = self.use_joint
        return a


# ======================================================================================
# Routed pipelines: SAER-7g, AlignedAddAudit and NestedAligned
# ======================================================================================
def site_keys(panel: dp.Panel) -> np.ndarray:
    return np.array([dp.site_key(s) for s in panel.sites], dtype=object)


def half_split(panel: dp.Panel, rows) -> np.ndarray | None:
    """Deterministic group-disjoint halves (0/1) of the fitting rows, or None if fewer than two groups."""
    sites = site_keys(panel)[rows]
    if len(set(sites)) < 2:
        return None
    return dp.fold_assignment(sites, max_folds=2, min_groups=2)


def sample_episodes(ep: dp.EpisodeSet, variant_rows, cap: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(np.isin(ep.i, variant_rows))
    if len(idx) > cap:
        idx = np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))
    return idx


def featurize(bank: ExpertBank, i, q, S) -> dict:
    """Expert proposals, signed support audits, correlations and normalised support residuals.

    Nothing here reads the hidden query response: audits compare each expert's prediction of a *support*
    inhibitor (from the remaining supports) with that support's observed response.
    """
    panel = bank.panel
    E, m = S.shape
    raw = bank.predict(i, q, S)
    b = raw[:, 0]
    Araw = np.empty((E, N_EXPERTS, m))
    for r in range(m):
        pr = bank.predict(i, S[:, r], np.delete(S, r, axis=1))
        Araw[:, :, r] = pr - panel.y[i, S[:, r]][:, None]
    s = bank.s
    P = (raw - b[:, None]) / s
    A = Araw / s
    avail = bank.avail
    return {"raw": raw, "b": b, "s": s, "P": np.where(avail, P, np.nan), "A": np.where(avail[None, :, None], A, np.nan),
            "rho": bank.rho[q[:, None], S], "r": (panel.y[i[:, None], S] - bank.prior[i[:, None], S]) / s,
            "prior_q": bank.prior[i, q], "q": q, "i": i, "avail": avail}


def feature_arrays(f: dict, skill: np.ndarray):
    P0, A0 = np.nan_to_num(f["P"]), np.nan_to_num(f["A"])
    Phi, C = rt.audit_features(P0, A0, f["rho"], skill)
    return Phi, C, P0, A0


def expert_mae(raw: np.ndarray, y: np.ndarray, w: np.ndarray | None, avail: np.ndarray) -> np.ndarray:
    err = np.abs(raw - y[:, None])
    w = np.ones(len(y)) if w is None else w
    mae = np.where(avail, (np.nan_to_num(err) * w[:, None]).sum(0) / w.sum(), np.nan)
    return np.where(avail, mae, mae[0])


def episode_weights(panel: dp.Panel, i: np.ndarray) -> np.ndarray:
    """Equal mass per gene on multi-kinase panels, uniform otherwise (normalised to sum to one)."""
    if panel.multi_gene:
        return rt.group_equal_mass(np.asarray(panel.genes)[i])
    return np.full(len(i), 1.0 / max(len(i), 1))


def select_settings(panel, cfg, halves: dict, eps_by_half: dict) -> dict:
    """Choose the Gaussian, mixture and donor settings on cross-half auxiliary predictions.

    ``halves`` maps direction -> ExpertBank fitted on one half; ``eps_by_half`` maps direction -> (i, q, S)
    episodes from the other half. The summed absolute error over both directions is minimised.
    """
    sc = cfg["saer"]
    grids = {
        "gaussian": [{"center": c, "shrink": h, "mult": a} for c, h, a in
                     product(sc["gaussian_centers"], sc["gaussian_shrink"], sc["gaussian_mult"])],
        "mixture": [{"n": n, "cov": cv, "reg": rg, "median": med} for n, cv, rg, med in
                    product(sc["mixture_components"], sc["mixture_cov"], sc["mixture_reg"], (False, True))],
        "donor": [{"bandwidth": h, "companion_weight": lam, "median": med} for h, lam, med in
                  product(sc["donor_bandwidth"], sc["donor_companion_weight"], (False, True))],
    }
    chosen = {}
    for fam, fn in (("gaussian", "gaussian"), ("mixture", "mixture"), ("donor", "donor")):
        losses = np.zeros(len(grids[fam]))
        for direction, bank in halves.items():
            i, q, S = eps_by_half[direction]
            y = bank.panel.y[i, q]
            for g, st in enumerate(grids[fam]):
                if fam == "donor" and st["companion_weight"] and bank.companion is None:
                    losses[g] = np.inf
                    continue
                losses[g] += np.abs(getattr(bank, fn)(i, q, S, st) - y).sum()
        chosen[fam] = grids[fam][int(np.argmin(losses))]
    return chosen


@dataclass
class AuxSet:
    feats: dict               # concatenated featurize() outputs over both directions
    y: np.ndarray
    i: np.ndarray
    w: np.ndarray


def build_aux(panel, ep, cfg, fit_rows, mode: str, k: int) -> tuple[AuxSet | None, dict | None]:
    """Cross-half auxiliary predictions on the fitting partition (experts fit on one half predict the other).

    ``mode='tuned'`` selects the Gaussian/mixture/donor settings on the auxiliary predictions;
    ``mode='fixed'`` uses the registered smaller settings (NestedAligned and pooled records).
    """
    halves = half_split(panel, fit_rows)
    if halves is None:
        return None, None
    sc = cfg["saer"]
    parts, banks, eps_by = {}, {}, {}
    fixed = cfg["nested_fixed"]
    for d, (src, dst) in enumerate(((0, 1), (1, 0))):
        rows_src, rows_dst = fit_rows[halves == src], fit_rows[halves == dst]
        bank = ExpertBank(panel, rows_src, cfg, dict(fixed))
        idx = sample_episodes(ep, rows_dst, sc["aux_max_episodes"], sc["aux_seed"] + d)
        banks[d], eps_by[d], parts[d] = bank, (ep.i[idx], ep.q[idx], ep.s[idx]), idx
        if len(idx) == 0:
            return None, None
    settings = dict(fixed)
    if mode == "tuned":
        sel_eps = {}
        rng = np.random.default_rng(sc["aux_seed"] + 7)
        for d in banks:
            i, q, S = eps_by[d]
            pick = np.sort(rng.choice(len(i), min(sc["select_sample"], len(i)), replace=False))
            sel_eps[d] = (i[pick], q[pick], S[pick])
        settings = select_settings(panel, cfg, banks, sel_eps)
    feats = []
    for d in banks:
        bank = banks[d]
        bank.settings = settings
        i, q, S = eps_by[d]
        f = featurize(bank, i, q, S)
        f["y"] = panel.y[i, q]
        feats.append(f)
    merged = {}
    for key in feats[0]:
        if key in ("s", "avail"):
            merged[key] = feats[0][key]
        else:
            merged[key] = np.concatenate([f[key] for f in feats])
    # normalise each direction by its own scale: store u and keep P/A normalised per bank (already done)
    merged["s_per_episode"] = np.concatenate([np.full(len(f["i"]), f["s"]) for f in feats])
    y = merged["y"]
    u = (y - merged["b"]) / merged["s_per_episode"]
    merged["u"] = u
    iall = merged["i"]
    return AuxSet(merged, y, iall, episode_weights(panel, iall)), settings


def records_from_aux(aux: AuxSet, skill: np.ndarray) -> rt.Records:
    f = aux.feats
    Phi, C, P0, A0 = feature_arrays(f, skill)
    return rt.Records(Phi, C, P0, f["u"], aux.w, f["avail"].copy(),
                      {"raw": f["raw"], "y": aux.y, "A": A0, "rho": f["rho"], "r": f["r"], "prior_q": f["prior_q"],
                       "q": f["q"], "b": f["b"], "s_ep": f["s_per_episode"], "Pn": f["P"], "An": f["A"]})


def aux_skill(aux: AuxSet) -> np.ndarray:
    return rt.relative_skill(expert_mae(aux.feats["raw"], aux.y, aux.w, aux.feats["avail"]))


class PoolProvider:
    """Routers fitted on the declared development pool (other development panels), never lockbox panels.

    ``records(exclude, k)`` returns pooled auxiliary records with each panel contributing equal total weight.
    """

    def __init__(self, panels: dict, cfg: dict):
        self.panels, self.cfg = panels, cfg
        self.cache = {}

    def panel_records(self, name: str, k: int):
        key = (name, k)
        if key not in self.cache:
            panel = self.panels[name]
            self.cache[key] = None
            if k in panel.budgets():
                ep = dp.build_episodes(panel, k)
                aux, _ = build_aux(panel, ep, self.cfg, np.arange(panel.n), "fixed", k)
                if aux is not None:
                    skill = aux_skill(aux)
                    rec = records_from_aux(aux, skill)
                    self.cache[key] = (rec, head_features_fallback(rec, aux), skill)
        return self.cache[key]

    def records(self, exclude: str | None, k: int):
        parts = [self.panel_records(n, k) for n in self.panels if n != exclude]
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        pooled = rt.stack_records([p[0] for p in parts])
        w_fb = np.concatenate([rt.normalise_total_weight(p[0].w) for p in parts])
        pooled.extra = {"X_fb": np.concatenate([p[1] for p in parts]), "w_fb": w_fb, "u_fb": pooled.u,
                        "skill": np.mean([p[2] for p in parts], axis=0)}
        return pooled


def head_features_fallback(rec: rt.Records, aux: AuxSet) -> np.ndarray:
    """Cross-study head features: no raw inhibitor index, query prior relative to b on scale s, pooled output zeroed."""
    f = aux.feats
    prior_rel = (f["prior_q"] - f["b"]) / f["s_per_episode"]
    return rt.aligned_features(rec.extra["Pn"], rec.extra["An"], f["rho"], f["r"], np.zeros(len(f["b"])), f["prior_q"],
                               f["q"], include_query_id=False, pool_zero=True, prior_relative=prior_rel)


def router_outputs(params: rt.RouterParams, feats: dict, skill: np.ndarray):
    Phi, C, P0, _ = feature_arrays(feats, skill)
    return rt.router_predict(params, Phi, C, P0)


def training_expert_error(rec: rt.Records) -> np.ndarray:
    """Weighted absolute error of each expert's proposal on the router's training episodes (normalised units)."""
    err = np.abs(rec.P - rec.u[:, None])
    return (err * rec.w[:, None]).sum(0) / rec.w.sum()


def _head(panel, cfg, seed, fallback=False, n_features=None):
    h = cfg["aligned"]
    fb = h["fallback_head"]
    return SafeHGB(loss="absolute_error", max_iter=fb["max_iter"] if fallback else h["max_iter"],
                   learning_rate=h["learning_rate"], max_leaf_nodes=fb["max_leaf_nodes"] if fallback else h["max_leaf_nodes"],
                   min_samples_leaf=fb["min_samples_leaf"] if fallback else h["min_samples_leaf"],
                   l2_regularization=h["l2_regularization"], early_stopping=False, random_state=seed,
                   categorical_index=(n_features - 1) if (not fallback and panel.n_drugs <= 255 and n_features) else None)


def routed_fold(panel: dp.Panel, ep: dp.EpisodeSet, outer: int, cfg: dict, pool: PoolProvider | None,
                methods=("SAER-7g", "AlignedAddAudit", "NestedAligned")) -> tuple[dict, dict]:
    """Predict the outer-test episodes of one fold with the routed models.

    Returns ``({method: predictions on test episodes}, info)``. Outer fold ``o`` uses the next fold as its
    designated validation partition; the remaining folds form the fitting partition. Fallbacks (no local
    auxiliary split, no pooled records, no residual head) follow the design document and are logged.
    """
    folds = panel.folds
    ef = folds[ep.i]
    present = sorted(int(f) for f in np.unique(folds))
    v = present[(present.index(int(outer)) + 1) % len(present)]
    fit_rows = np.flatnonzero(~np.isin(folds, [outer, v]))
    val_idx, test_idx = np.flatnonzero(ef == v), np.flatnonzero(ef == outer)
    k = ep.k
    info = {"outer_fold": int(outer), "validation_fold": int(v), "fit_variants": int(len(fit_rows))}
    preds, failures = {}, {}
    pooled_rec = pool.records(panel.name, k) if pool is not None else None
    info["pooled_records"] = 0 if pooled_rec is None else int(len(pooled_rec))
    grid_m, grid_tau, grid_lam = cfg["aligned"]["multipliers"], cfg["aligned"]["tau"], cfg["aligned"]["lambda"]

    def sub(idx):
        return ep.i[idx], ep.q[idx], ep.s[idx]

    def run(style: str):
        nested = style == "nested"
        aux, settings = (build_aux(panel, ep, cfg, fit_rows, "fixed" if nested else "tuned", k)
                         if len(fit_rows) >= 2 else (None, None))
        bank = ExpertBank(panel, fit_rows, cfg, settings)
        avail, s = bank.avail, bank.s
        fv, ft = featurize(bank, *sub(val_idx)), featurize(bank, *sub(test_idx))
        yv = panel.y[ep.i[val_idx], ep.q[val_idx]]
        wv = episode_weights(panel, ep.i[val_idx])
        local_rec = None
        if aux is not None:
            skill = aux_skill(aux) if nested else rt.relative_skill(expert_mae(fv["raw"], yv, wv, avail))
            local_rec = records_from_aux(aux, skill)
        elif nested and pooled_rec is not None:
            skill = pooled_rec.extra["skill"]                       # no local auxiliary data: pooled skill
        else:
            skill = rt.relative_skill(expert_mae(fv["raw"], yv, wv, avail))
        local = rt.fit_router(local_rec, cfg["saer"]["router_max_iter"]) if local_rec is not None else None
        pooled = rt.fit_router(pooled_rec, cfg["saer"]["router_max_iter"]) if pooled_rec is not None else None
        used = {"local_router": local is not None, "pooled_router": pooled is not None}
        if local is None and pooled is None:
            return None, {"reason": "no_router_available", **used}
        local_p, pooled_p = local or pooled, pooled or local
        local_rec_e = local_rec if local is not None else pooled_rec
        pooled_rec_e = pooled_rec if pooled is not None else local_rec
        errs = {"local": training_expert_error(local_rec_e), "pooled": training_expert_error(pooled_rec_e)}

        def outputs(feats):
            wl, Ppl, fl = router_outputs(local_p, feats, skill)
            wp, Ppp, fp = router_outputs(pooled_p, feats, skill)
            return {"wl": wl, "Ppl": Ppl, "fl": fl, "wp": wp, "Ppp": Ppp, "fp": fp,
                    "I": feats["b"] + s * (fl + fp) / 2.0, "Ppool": feats["b"] + s * fp}
        ov, ot = outputs(fv), outputs(ft)

        def q_median(feats, o, mult):
            absA = np.abs(np.nan_to_num(feats["A"])).mean(axis=2)                          # (E, J)
            cen = np.concatenate([o["Ppl"], o["Ppp"]], axis=1)
            a = np.concatenate([o["wl"], o["wp"]], axis=1) / 2.0
            d = np.concatenate([mult * np.maximum(absA, 0.25 * errs["local"][None, :]),
                                mult * np.maximum(absA, 0.25 * errs["pooled"][None, :])], axis=1)
            keep = np.concatenate([avail, avail])
            return feats["b"] + s * rt.laplace_mixture_median(cen[:, keep], a[:, keep], np.maximum(d[:, keep], 1e-8))

        # validation selection of the reference blend (a tie keeps the identity / smaller intervention)
        best = (rt.weighted_mae(ov["I"], yv, wv), 0.0, None)
        for tau in grid_tau:
            if tau == 0:
                continue
            for mult in grid_m:
                cand = ov["I"] + tau * (q_median(fv, ov, mult) - ov["I"])
                score = rt.weighted_mae(cand, yv, wv)
                if score < best[0] - 1e-12:
                    best = (score, tau, mult)
        _, tau_sel, m_sel = best
        A_v = ov["I"] if tau_sel == 0.0 else ov["I"] + tau_sel * (q_median(fv, ov, m_sel) - ov["I"])
        A_t = ot["I"] if tau_sel == 0.0 else ot["I"] + tau_sel * (q_median(ft, ot, m_sel) - ot["I"])
        st = {"settings": settings, "skill": [float(x) for x in skill], "tau": tau_sel, "multiplier": m_sel, "used": used}
        seed = cfg["aligned"]["nested_seed"] if nested else cfg["aligned"]["seed"]
        Bv = Bt = None
        if local_rec is not None:
            P_a, A_a = local_rec.P, np.nan_to_num(local_rec.extra["An"])
            _, _, fp_a = rt.router_predict(pooled_p, local_rec.Phi, local_rec.C, P_a)
            ex = local_rec.extra
            Xa = rt.aligned_features(P_a, A_a, ex["rho"], ex["r"], fp_a, ex["prior_q"], ex["q"])
            target = (ex["y"] - (ex["b"] + ex["s_ep"] * fp_a)) / ex["s_ep"]
            head = _head(panel, cfg, seed, False, Xa.shape[1]).fit(Xa, target, sample_weight=local_rec.w / local_rec.w.mean())

            def head_pred(feats, o):
                X = rt.aligned_features(np.nan_to_num(feats["P"]), np.nan_to_num(feats["A"]), feats["rho"], feats["r"],
                                        o["fp"], feats["prior_q"], feats["q"])
                return o["Ppool"] + s * head.predict(X)
            Bv, Bt = head_pred(fv, ov), head_pred(ft, ot)
            st["head"] = "aligned_residual"
        elif pooled_rec is not None and "X_fb" in pooled_rec.extra:
            head = _head(panel, cfg, seed, True).fit(pooled_rec.extra["X_fb"], pooled_rec.extra["u_fb"],
                                                      sample_weight=pooled_rec.extra["w_fb"] / pooled_rec.extra["w_fb"].mean())

            def head_pred(feats, o):
                prior_rel = (feats["prior_q"] - feats["b"]) / s
                X = rt.aligned_features(feats["P"], feats["A"], feats["rho"], feats["r"], np.zeros(len(feats["b"])),
                                        feats["prior_q"], feats["q"], include_query_id=False, pool_zero=True,
                                        prior_relative=prior_rel)
                return feats["b"] + s * head.predict(X)
            Bv, Bt = head_pred(fv, ov), head_pred(ft, ot)
            st["head"] = "cross_study_fallback"
        else:
            st["head"] = "none_zero_correction"
        lam_sel = 0.0
        if Bv is not None:
            best_l = (rt.weighted_mae(A_v, yv, wv), 0.0)
            for lam in grid_lam:
                if lam == 0:
                    continue
                score = rt.weighted_mae(A_v + lam * (Bv - ov["I"]), yv, wv)
                if score < best_l[0] - 1e-12:
                    best_l = (score, lam)
            lam_sel = best_l[1]
        st["lambda"] = lam_sel
        final = A_t if Bt is None else A_t + lam_sel * (Bt - ot["I"])
        return {"identity": ot["I"], "final": final}, st

    if "SAER-7g" in methods or "AlignedAddAudit" in methods:
        out, st = run("aligned")
        if out is None:
            for m_ in ("SAER-7g", "AlignedAddAudit"):
                failures[m_] = st["reason"]
        else:
            preds["SAER-7g"], preds["AlignedAddAudit"] = out["identity"], out["final"]
            info["aligned"] = st
    if "NestedAligned" in methods:
        out, st = run("nested")
        if out is None:
            failures["NestedAligned"] = st["reason"]
        else:
            preds["NestedAligned"] = out["final"]
            info["nested"] = st
    info["failures"] = failures
    return {m: preds[m] for m in preds if m in methods}, info


# ======================================================================================
# Driver: one prediction (or explicit failure) per method and episode
# ======================================================================================
@dataclass
class PanelRun:
    panel: str
    k: int
    ids: np.ndarray
    folds: np.ndarray
    preds: dict
    failures: dict                  # method -> {episode index: reason}
    decisions: dict
    ser_intervened: np.ndarray


def run_panel(panel: dp.Panel, k: int, cfg: dict, methods=METHODS, pool: PoolProvider | None = None,
              ep: dp.EpisodeSet | None = None, log=None) -> PanelRun:
    """Run the requested methods on the same episode set with nested residue-group folds."""
    log = log or (lambda *a, **kw: None)
    ep = ep or dp.build_episodes(panel, k)
    E = len(ep)
    ef = panel.folds[ep.i]
    preds = {m: np.full(E, np.nan) for m in methods}
    failures = {m: {} for m in methods}
    decisions: dict = {"folds": {}}
    intervened = np.zeros(E, dtype=bool)
    if set(methods) & {"XGBoost", "CovarianceLocalMedian", "SER", "QueryMedian", "RidgeSupports"}:
        log(f"{panel.name} k={k}: baseline engine")
        out = run_baselines(panel, ep, cfg)
        for name, arr in (("XGBoost", out.xgb), ("CovarianceLocalMedian", out.local), ("SER", out.ser),
                          ("QueryMedian", out.qmed), ("RidgeSupports", out.ridge)):
            if name in preds:
                preds[name] = arr
        for name, reasons in out.failures.items():
            if name in failures:
                for key, reason in reasons.items():
                    if isinstance(key, int):
                        failures[name][key] = reason
                    else:
                        failures[name].update({int(e): reason for e in np.flatnonzero(ef == key)})
        decisions["SER_gate"], decisions["baseline_selection"] = out.decisions, out.selected
        intervened = out.ser_intervened
    routed = [m for m in methods if m in ("SAER-7g", "AlignedAddAudit", "NestedAligned")]
    if routed:
        decisions["routed"] = []
        for outer in sorted(int(f) for f in np.unique(panel.folds)):
            log(f"{panel.name} k={k}: routed models, outer fold {outer}")
            test_idx = np.flatnonzero(ef == outer)
            try:
                got, info = routed_fold(panel, ep, outer, cfg, pool, tuple(routed))
            except Exception as exc:                                  # explicit failure row, never a silent gap
                got, info = {}, {"outer_fold": outer, "failures": {m: f"exception:{type(exc).__name__}:{exc}" for m in routed}}
            decisions["routed"].append({k_: v for k_, v in info.items()})
            for m in routed:
                if m in got:
                    preds[m][test_idx] = got[m]
                else:
                    reason = info.get("failures", {}).get(m, "not_produced")
                    failures[m].update({int(e): reason for e in test_idx})
    for m in methods:                                                 # every missing value carries a reason
        for e in np.flatnonzero(~np.isfinite(preds[m])):
            failures[m].setdefault(int(e), "no_prediction")
    return PanelRun(panel.name, k, ep.ids, ef, preds, failures, decisions, intervened)


def save_run(run: PanelRun, directory: str | Path) -> dict:
    """Write predictions by episode id (no truth) and the decision log."""
    import csv
    import gzip
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    methods = list(run.preds)
    path = directory / f"{run.panel}_k{run.k}_predictions.csv.gz"
    lines = ["episode_id,fold," + ",".join(methods) + ",failure_reasons\n"]
    for e, eid in enumerate(run.ids):
        vals = ",".join("" if not np.isfinite(run.preds[m][e]) else repr(float(run.preds[m][e])) for m in methods)
        reasons = ";".join(f"{m}={run.failures[m][e]}" for m in methods if e in run.failures[m]).replace(",", " ")
        lines.append(f"{eid},{int(run.folds[e])},{vals},{reasons}\n")
    dp.write_gz_deterministic(path, "".join(lines))
    (directory / f"{run.panel}_k{run.k}_decisions.json").write_text(
        json.dumps({"ser_intervention_rate": float(run.ser_intervened.mean()), **run.decisions}, indent=2, default=_json_default) + "\n")
    return {"file": path.name, "episodes": len(run.ids)}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
