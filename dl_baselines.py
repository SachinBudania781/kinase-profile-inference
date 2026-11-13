"""Development-only deep-learning reference baselines (never run on a lockbox, never part of the primary test).

Four small PyTorch models predict the hidden query response from k observed support responses:
  * set     - masked-attention set transformer over inhibitor tokens
  * graph   - two-layer graph convolution over inhibitors (edges = training-fold Spearman correlation)
  * cnn     - 1D convolution over the canonical inhibitor axis (value + observed channels)
  * lstm    - LSTM over the supports in canonical inhibitor order, then the query embedding

All use an L1 loss, the same episodes and the same residue-group folds as the main study. The protocol
reports validation MAE at k in {2,3} on Estevam2025_MET and OKL2026_multi_kinase only, next to XGBoost
trained on the identical fitting partition. These numbers only confirm that the reference learner is not
trivially weak; they never decide the primary comparison.

    python dl_baselines.py --archive-root <data> --panels Estevam2025_MET,OKL2026_multi_kinase --out results/dl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

import core_models as cm
import data_pipeline as dp

SEED = dp.SEED
ARCHS = ("set", "graph", "cnn", "lstm")
DEV_ONLY_PANELS = ("Estevam2025_MET", "OKL2026_multi_kinase")
DL_DEFAULTS = {"dim": 48, "heads": 4, "layers": 2, "epochs": 12, "batch": 256, "lr": 2e-3, "weight_decay": 1e-4,
               "max_train_episodes": 36000, "patience": 3, "graph_neighbors": 6}


# ---------------------------------------------------------------------------------- tensors
def episode_tensors(panel: dp.Panel, ep: dp.EpisodeSet, scale: tuple):
    """Dense (E, D) value / observed matrices plus the query index, in canonical inhibitor order."""
    E, D = len(ep), panel.n_drugs
    center, spread = scale
    val = np.zeros((E, D), dtype=np.float32)
    obs = np.zeros((E, D), dtype=np.float32)
    rows = np.arange(E)[:, None]
    val[rows, ep.s] = ((panel.y[ep.i[:, None], ep.s] - center[ep.s]) / spread[ep.s]).astype(np.float32)
    obs[rows, ep.s] = 1.0
    q = torch.as_tensor(ep.q, dtype=torch.long)
    return torch.as_tensor(val), torch.as_tensor(obs), q


def target_scale(panel: dp.Panel, fit_rows: np.ndarray):
    """Training-fold-only per-inhibitor centre and spread (robust, never zero)."""
    y = panel.y[fit_rows]
    center = np.nanmedian(y, axis=0)
    spread = np.maximum(np.nanstd(y, axis=0), 1e-3)
    return center, spread


class SetNet(nn.Module):
    def __init__(self, D, dim, heads, layers):
        super().__init__()
        self.drug = nn.Embedding(D, dim)
        self.value = nn.Linear(1, dim)
        self.flag = nn.Embedding(3, dim)               # 0 absent, 1 support, 2 query
        enc = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, val, obs, q):
        B, D = val.shape
        state = obs.long()
        state[torch.arange(B), q] = 2
        ids = torch.arange(D, device=val.device).expand(B, D)
        x = self.drug(ids) + self.flag(state) + self.value(val.unsqueeze(-1)) * obs.unsqueeze(-1)
        keep = state > 0                                # absent inhibitors are masked out of attention
        h = self.enc(x, src_key_padding_mask=~keep)
        return self.out(h[torch.arange(B), q]).squeeze(-1)


class GraphNet(nn.Module):
    def __init__(self, D, dim, adj):
        super().__init__()
        self.register_buffer("adj", adj)
        self.drug = nn.Embedding(D, dim)
        self.inp = nn.Linear(3, dim)
        self.g1, self.g2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.out = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, val, obs, q):
        B, D = val.shape
        qflag = torch.zeros_like(val)
        qflag[torch.arange(B), q] = 1.0
        h = self.inp(torch.stack([val, obs, qflag], dim=-1)) + self.drug.weight.unsqueeze(0)
        h = torch.relu(self.g1(self.adj @ h)) + h
        h = torch.relu(self.g2(self.adj @ h)) + h
        hq = h[torch.arange(B), q]
        return self.out(torch.cat([hq, h.mean(1)], dim=-1)).squeeze(-1)


class CNNNet(nn.Module):
    def __init__(self, D, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(3, dim, 3, padding=1), nn.GELU(), nn.Conv1d(dim, dim, 3, padding=1), nn.GELU())
        self.out = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, val, obs, q):
        B, D = val.shape
        qflag = torch.zeros_like(val)
        qflag[torch.arange(B), q] = 1.0
        h = self.conv(torch.stack([val, obs, qflag], dim=1))               # (B, dim, D)
        return self.out(torch.cat([h[torch.arange(B), :, q], h.mean(2)], dim=-1)).squeeze(-1)


class LSTMNet(nn.Module):
    def __init__(self, D, dim):
        super().__init__()
        self.drug = nn.Embedding(D, dim)
        self.value = nn.Linear(1, dim)
        self.rnn = nn.LSTM(dim, dim, batch_first=True)
        self.out = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, val, obs, q):
        B, D = val.shape
        order = torch.argsort(-obs, dim=1, stable=True)                    # supports first, canonical order kept
        ids = order
        v = torch.gather(val, 1, order)
        o = torch.gather(obs, 1, order)
        x = (self.drug(ids) + self.value(v.unsqueeze(-1))) * o.unsqueeze(-1)
        lengths = obs.sum(1).long().clamp(min=1)
        h, _ = self.rnn(x)
        last = h[torch.arange(B), lengths - 1]
        return self.out(torch.cat([last, self.drug(q)], dim=-1)).squeeze(-1)


def correlation_graph(panel: dp.Panel, fit_rows: np.ndarray, neighbors: int) -> torch.Tensor:
    """Row-normalised adjacency with self loops over inhibitors, built from training residues only."""
    D = panel.n_drugs
    y = panel.y[fit_rows]
    corr = np.zeros((D, D))
    for a in range(D):
        for b in range(a + 1, D):
            ok = np.isfinite(y[:, a]) & np.isfinite(y[:, b])
            if ok.sum() >= 5:
                c = spearmanr(y[ok, a], y[ok, b])[0]
                corr[a, b] = corr[b, a] = 0.0 if not np.isfinite(c) else abs(c)
    adj = np.eye(D)
    for a in range(D):
        top = np.argsort(-corr[a])[:neighbors]
        adj[a, top] = np.maximum(adj[a, top], corr[a, top])
    adj = np.maximum(adj, adj.T)
    return torch.as_tensor(adj / adj.sum(1, keepdims=True), dtype=torch.float32)


def build_model(arch: str, panel: dp.Panel, fit_rows, cfg: dict) -> nn.Module:
    D, dim = panel.n_drugs, cfg["dim"]
    if arch == "set":
        return SetNet(D, dim, cfg["heads"], cfg["layers"])
    if arch == "graph":
        return GraphNet(D, dim, correlation_graph(panel, fit_rows, cfg["graph_neighbors"]))
    if arch == "cnn":
        return CNNNet(D, dim)
    if arch == "lstm":
        return LSTMNet(D, dim)
    raise ValueError(arch)


# ---------------------------------------------------------------------------------- training
def train_predict(arch, panel, ep, fit_idx, val_idx, cfg, seed=SEED):
    """Fit on ``fit_idx`` episodes, early-stop on ``val_idx``, return validation predictions in raw units."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fit_rows = np.unique(ep.i[fit_idx])
    scale = target_scale(panel, fit_rows)
    val, obs, q = episode_tensors(panel, ep, scale)
    y_scaled = torch.as_tensor(((panel.y[ep.i, ep.q] - scale[0][ep.q]) / scale[1][ep.q]).astype(np.float32))
    model = build_model(arch, panel, fit_rows, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    fit_idx = cm.subsample_indices(fit_idx, cfg["max_train_episodes"], seed)
    best, best_state, stale = np.inf, None, 0

    def predict(idx):
        model.eval()
        out = []
        with torch.no_grad():
            for part in np.array_split(idx, max(1, len(idx) // 2048)):
                out.append(model(val[part], obs[part], q[part]))
        return torch.cat(out).numpy() * scale[1][ep.q[idx]] + scale[0][ep.q[idx]]

    for epoch in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(fit_idx)
        for start in range(0, len(order), cfg["batch"]):
            b = order[start:start + cfg["batch"]]
            opt.zero_grad()
            loss = (model(val[b], obs[b], q[b]) - y_scaled[b]).abs().mean()
            loss.backward()
            opt.step()
        mae = float(np.abs(predict(val_idx) - panel.y[ep.i[val_idx], ep.q[val_idx]]).mean())
        if mae < best - 1e-6:
            best, stale = mae, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return predict(val_idx)


def validation_report(panel: dp.Panel, k: int, cfg: dict, archs=ARCHS, cm_cfg: dict | None = None):
    """Validation-fold MAE of each DL architecture next to XGBoost fitted on the same partition."""
    cm_cfg = cm_cfg or cm.load_config()
    ep = dp.build_episodes(panel, k)
    folds = panel.folds[ep.i]
    n_folds = int(folds.max()) + 1
    outer = 0
    val_fold = (outer + 1) % n_folds
    fit_idx = np.flatnonzero((folds != outer) & (folds != val_fold))
    val_idx = np.flatnonzero(folds == val_fold)
    y_val = panel.y[ep.i[val_idx], ep.q[val_idx]]
    X = cm.xgb_features(panel, ep)
    y_all = panel.y[ep.i, ep.q]
    xgb_mae = {}
    for row in cm_cfg["xgb"]["grid"]:
        model = cm.fit_xgb(X, y_all, fit_idx, row, cm_cfg)
        xgb_mae[str(tuple(row))] = float(np.abs(model.predict(X[val_idx]) - y_val).mean())
    report = {"panel": panel.name, "k": k, "outer_fold": outer, "validation_fold": int(val_fold), "fit_episodes": int(len(fit_idx)),
              "validation_episodes": int(len(val_idx)), "xgboost_mae_by_grid_row": xgb_mae, "xgboost_best_mae": min(xgb_mae.values()),
              "query_median_mae": float(np.abs(np.nanmedian(panel.y[ep.i[fit_idx]], axis=0)[ep.q[val_idx]] - y_val).mean()),
              "dl_mae": {}}
    for arch in archs:
        pred = train_predict(arch, panel, ep, fit_idx, val_idx, cfg)
        report["dl_mae"][arch] = float(np.abs(pred - y_val).mean())
    report["best_dl"] = min(report["dl_mae"], key=report["dl_mae"].get)
    report["xgboost_competitive"] = bool(report["xgboost_best_mae"] <= 1.05 * min(report["dl_mae"].values()))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive-root", required=True)
    p.add_argument("--panels", default=",".join(DEV_ONLY_PANELS))
    p.add_argument("--archs", default=",".join(ARCHS))
    p.add_argument("--budgets", default="2,3")
    p.add_argument("--epochs", type=int, default=DL_DEFAULTS["epochs"])
    p.add_argument("--registry", default=None)
    p.add_argument("--out", default="results/dl")
    args = p.parse_args(argv)
    registry = dp.load_registry(args.registry)
    cfg = {**DL_DEFAULTS, "epochs": args.epochs}
    reports = []
    for name in args.panels.split(","):
        role = registry["panels"][name]["role"]
        if role in dp.LOCKBOX_ROLES:
            raise SystemExit(f"{name} is a lockbox panel; deep-learning baselines are development-only.")
        panel = dp.load_panel(registry, name, {"archive": args.archive_root})
        for k in (int(b) for b in args.budgets.split(",")):
            if k in panel.budgets():
                rep = validation_report(panel, k, cfg, tuple(args.archs.split(",")))
                reports.append(rep)
                print(json.dumps({x: rep[x] for x in ("panel", "k", "xgboost_best_mae", "dl_mae", "xgboost_competitive")}))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dl_baseline_validation.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
