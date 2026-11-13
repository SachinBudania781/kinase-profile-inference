"""Step 1 - data pipeline for the Kinase Profile Inference study.

Responsibilities
----------------
* Read the cleaned study panels (development matrices and lockbox ``SER_responses.csv`` tables).
* Keep lockbox response tables out of any development path (explicit access flag, private output dir).
* Assign residue/group folds by the registered seeded SHA-256 rule.
* Enumerate prediction episodes ``e = (study, i, q, S)`` with stable IDs, the same for every method.
* Write manifests, fold tables and audit reports, and compute SHA-256 bundle hashes.

The hidden query response is never part of an ``EpisodeSet``. Truth is joined only by
``EpisodeSet.truth(panel)`` inside the scoring step.

Command line (see ``python data_pipeline.py --help``)::

    python data_pipeline.py audit    --archive-root <data> --roles core_dev,stress_test
    python data_pipeline.py episodes --archive-root <data> --roles core_dev,stress_test --out splits
    python data_pipeline.py lockbox  --archive-root <data> [--lockbox-b-root <dirB>] --private-dir lockbox_private
    python data_pipeline.py hashes   --archive-root <data> [--lockbox-b-root <dirB>]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

SEED = 20261112
BUDGETS = (2, 3)
MAX_FOLDS = 5
SCHEMA_VERSION = "1.0"
ROLES = ("core_dev", "stress_test", "lockbox_A", "lockbox_B")
DEV_ROLES = ("core_dev", "stress_test")
LOCKBOX_ROLES = ("lockbox_A", "lockbox_B")
CENSOR_FLAGS = (-1, 0, 1)
REPO_ROOT = Path(__file__).resolve().parent


class PanelValidationError(ValueError):
    """A panel violates the registered input contract."""


class LockboxAccessError(PermissionError):
    """A lockbox response table was requested without the explicit access flag."""


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
def load_registry(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "panels.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    for name, spec in registry["panels"].items():
        if spec["role"] not in ROLES:
            raise PanelValidationError(f"{name}: unknown role {spec['role']}")
    return registry


def panels_by_role(registry: dict, roles) -> list[str]:
    roles = set(roles)
    return [n for n, s in registry["panels"].items() if s["role"] in roles]


# --------------------------------------------------------------------------------------
# Folds
# --------------------------------------------------------------------------------------
def site_key(site) -> str:
    """Canonical string for a site/group id (ints print without a decimal point)."""
    if isinstance(site, (float, np.floating)) and float(site).is_integer():
        return str(int(site))
    return str(site.item() if hasattr(site, "item") else site)


def fold_assignment(sites, seed: int = SEED, max_folds: int = MAX_FOLDS, min_groups: int = 4) -> np.ndarray:
    """Registered rule: sort site IDs by SHA-256 of ``"{seed}:{site}"`` and deal them cyclically.

    ``n_folds = min(max_folds, n_sites)``; the released loader requires at least four sites.
    Panels with a broader source grouping contract (e.g. three FGFR genes) may declare a lower
    ``min_groups`` in ``configs/panels.json``. All members of a site always share a fold.
    """
    keys = [site_key(s) for s in sites]
    unique = sorted(set(keys), key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).digest())
    n_folds = min(max_folds, len(unique))
    if n_folds < min_groups:
        raise PanelValidationError(f"{len(unique)} groups; at least {min_groups} are required for grouped folds.")
    mapping = {p: i % n_folds for i, p in enumerate(unique)}
    return np.array([mapping[k] for k in keys], dtype=np.int16)


def fold_table(panel: "Panel") -> list[tuple[str, int, int]]:
    """Rows ``(site, fold, n_variants)`` sorted by site for hashing."""
    keys = [site_key(s) for s in panel.sites]
    out = {}
    for key, fold in zip(keys, panel.folds):
        rec = out.setdefault(key, [int(fold), 0])
        rec[1] += 1
    return [(k, v[0], v[1]) for k, v in sorted(out.items())]


# --------------------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------------------
@dataclass
class Panel:
    name: str
    role: str
    kinase: str
    drugs: tuple
    modes: tuple
    y: np.ndarray                       # (n, D) float64, NaN = not measured
    censor: np.ndarray                  # (n, D) int8, 0 exact, +1 lower bound, -1 upper bound
    labels: tuple
    sites: np.ndarray                   # (n,) object, residue position or declared group
    genes: np.ndarray                   # (n,) object
    response: str
    scale_kind: str
    companion: np.ndarray | None = None  # (n, D) permitted companion/reference matrix, or None
    context: np.ndarray | None = None    # (n, F) permitted context features, or None
    spec: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    folds: np.ndarray = None

    @property
    def n(self) -> int:
        return self.y.shape[0]

    @property
    def n_drugs(self) -> int:
        return self.y.shape[1]

    @property
    def multi_gene(self) -> bool:
        return bool(self.spec.get("multi_gene", False))

    @property
    def n_sites(self) -> int:
        return len({site_key(s) for s in self.sites})

    @property
    def observed(self) -> np.ndarray:
        return np.isfinite(self.y)

    def budgets(self) -> tuple:
        """Allowed budgets: the query must remain, so k <= D - 1."""
        declared = self.spec.get("declared", {}).get("budgets")
        allowed = tuple(k for k in BUDGETS if k <= self.n_drugs - 1)
        return tuple(k for k in allowed if declared is None or k in declared)


def _validate_core(name, y, censor, drugs, sites, min_groups):
    if y.ndim != 2 or y.shape[1] != len(drugs):
        raise PanelValidationError(f"{name}: response matrix and drug list disagree.")
    if len(set(drugs)) != len(drugs):
        raise PanelValidationError(f"{name}: duplicate drug names.")
    if not set(np.unique(censor).tolist()) <= set(CENSOR_FLAGS):
        raise PanelValidationError(f"{name}: censor flags must be in {CENSOR_FLAGS}.")
    if np.isinf(y).any():
        raise PanelValidationError(f"{name}: infinite responses are not allowed.")
    if ((censor != 0) & ~np.isfinite(y)).any():
        raise PanelValidationError(f"{name}: censor flag on a missing response.")
    if len({site_key(s) for s in sites}) < min_groups:
        raise PanelValidationError(f"{name}: fewer than {min_groups} residue groups.")


def _read_study_json(directory: Path) -> dict:
    path = directory / "study.json"
    if not path.exists():
        raise PanelValidationError(f"{directory.name}: study.json is missing.")
    return json.loads(path.read_text(encoding="utf-8"))


def _scale_kind_from_response(text: str) -> str:
    t = text.lower()
    if "ic50" in t:
        return "ic50_log10_ratio"
    if "enrichment" in t or "fold change" in t:
        return "enrichment"
    return "other"


def load_ser_csv_panel(directory: Path, name: str, spec: dict, role: str) -> Panel:
    """Lockbox-format panel: ``study.json['ser']`` recipe + a long ``SER_responses.csv`` table.

    A complete drug profile is required for every admitted variant (same rule as the released
    SER loader). Duplicate rows or inconsistent positions are rejected.
    """
    meta = _read_study_json(directory)
    if "ser" not in meta:
        raise PanelValidationError(f"{name}: study.json has no 'ser' recipe.")
    recipe = meta["ser"]
    drugs = tuple(recipe["drugs"])
    modes = tuple(recipe.get("modes", ["unspecified"] * len(drugs)))
    if len(modes) != len(drugs):
        raise PanelValidationError(f"{name}: modes and drugs disagree.")
    rows: dict[str, tuple] = {}
    n_source_rows = 0
    with (directory / recipe["table"]).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            n_source_rows += 1
            label, drug = row["variant"], row["drug"]
            value, censor = float(row["response"]), int(row["censor"])
            if drug not in drugs:
                raise PanelValidationError(f"{name}: drug {drug!r} is not in the declared order.")
            if censor not in CENSOR_FLAGS or not math.isfinite(value):
                raise PanelValidationError(f"{name}: invalid response for {label}/{drug}.")
            pos = int(float(row["position"]))
            rec = rows.setdefault(label, (pos, row.get("kinase", meta.get("kinase", "")), {}))
            if rec[0] != pos or drug in rec[2]:
                raise PanelValidationError(f"{name}: duplicate or inconsistent row for {label}/{drug}.")
            rec[2][drug] = (value, censor)
    labels = tuple(sorted(rows))
    if not labels or any(set(rows[l][2]) != set(drugs) for l in labels):
        raise PanelValidationError(f"{name}: a complete drug profile is required for every admitted variant.")
    y = np.array([[rows[l][2][d][0] for d in drugs] for l in labels], dtype=float)
    c = np.array([[rows[l][2][d][1] for d in drugs] for l in labels], dtype=np.int8)
    sites = np.array([rows[l][0] for l in labels], dtype=object)
    genes = np.array([rows[l][1] for l in labels], dtype=object)
    _validate_core(name, y, c, drugs, sites, spec.get("min_groups", 4))
    response = recipe.get("response", "")
    panel = Panel(name, role, meta.get("kinase", ""), drugs, modes, y, c, labels, sites, genes, response,
                  _scale_kind_from_response(response), None, None, spec)
    panel.folds = fold_assignment(sites, min_groups=spec.get("min_groups", 4))
    panel.audit = {"source_rows": n_source_rows, "admitted_profiles": len(labels), "excluded_profiles": 0,
                   "exclusion_reasons": {}}
    return panel


def load_matrix_npz_panel(directory: Path, name: str, spec: dict, role: str) -> Panel:
    """Development-format panel: ``response_matrix.npz`` with Y (profiles x drugs), NaN = missing.

    Profiles need at least ``k_min + 1`` finite cells (k_min = 2) to form any episode; other profiles
    are excluded with a recorded reason. Development panels carry no declared censor flags (all exact).
    """
    meta = _read_study_json(directory)
    z = np.load(directory / "response_matrix.npz", allow_pickle=True)
    y_all = np.asarray(z["Y"], dtype=float)
    drugs = tuple(str(d) for d in z["drugs"])
    labels_all = tuple(str(l) for l in z["labels"])
    groups_all = np.array([site_key(g) for g in z["groups"]], dtype=object)
    genes_all = np.array([str(g) for g in z["genes"]], dtype=object)
    finite = np.isfinite(y_all)
    keep = finite.sum(axis=1) >= BUDGETS[0] + 1
    reasons = {}
    if (~keep).any():
        reasons["fewer_than_3_finite_cells"] = int((~keep).sum())
    dup = {}
    for idx, lab in enumerate(labels_all):
        dup.setdefault(lab, []).append(idx)
    if any(len(v) > 1 for v in dup.values()):
        raise PanelValidationError(f"{name}: duplicate profile labels.")
    y = y_all[keep]
    c = np.zeros(y.shape, dtype=np.int8)
    W = np.asarray(z["W"], dtype=float)[keep] if "W" in z.files else None
    ctx = None
    if spec.get("prior") == "companion_context":
        cpath = directory / "MET_context_features.npz"
        if cpath.exists():
            ctx = np.asarray(np.load(cpath, allow_pickle=True)["features"], dtype=float)[keep]
    sites = groups_all[keep]
    _validate_core(name, y, c, drugs, sites, spec.get("min_groups", 4))
    panel = Panel(name, role, meta.get("kinase", ""), drugs, tuple(["unspecified"] * len(drugs)), y, c,
                  tuple(l for l, k in zip(labels_all, keep) if k), sites, genes_all[keep],
                  f"source-defined scale ({spec.get('scale_kind', 'other')})", spec.get("scale_kind", "other"),
                  W, ctx, spec)
    panel.folds = fold_assignment(sites, min_groups=spec.get("min_groups", 4))
    panel.audit = {"source_profiles": int(len(keep)), "admitted_profiles": int(keep.sum()),
                   "excluded_profiles": int((~keep).sum()), "exclusion_reasons": reasons}
    return panel


def load_panel(registry: dict, name: str, roots: dict, allow_lockbox: bool = False) -> Panel:
    """Load one registered panel. Lockbox roles require ``allow_lockbox=True``."""
    spec = registry["panels"][name]
    role = spec["role"]
    if role in LOCKBOX_ROLES and not allow_lockbox:
        raise LockboxAccessError(f"{name} is a {role} panel; response tables are curator-only.")
    root = roots.get(spec["root"])
    if root is None:
        raise PanelValidationError(f"{name}: root {spec['root']!r} was not provided.")
    directory = Path(root) / name
    if not directory.is_dir():
        raise PanelValidationError(f"{name}: folder not found at {directory}.")
    if spec["format"] == "ser_csv":
        return load_ser_csv_panel(directory, name, spec, role)
    if spec["format"] == "matrix_npz":
        return load_matrix_npz_panel(directory, name, spec, role)
    raise PanelValidationError(f"{name}: unknown format {spec['format']}.")


# --------------------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------------------
def episode_id(study: str, k: int, label: str, drugs, q: int, s) -> str:
    text = f"{study}|k{k}|{label}|q:{drugs[q]}|S:{'+'.join(drugs[j] for j in s)}"
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def _profile_rng(seed: int, study: str, k: int, label: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{study}:{k}:{label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def n_masks(n_obs: int, k: int) -> int:
    return n_obs * math.comb(n_obs - 1, k) if n_obs - 1 >= k else 0


def enumerate_profile_masks(obs: np.ndarray, k: int, max_masks: int | None, rng: np.random.Generator):
    """All ``(q, S)`` over observed drug indices ``obs`` (ascending), or a uniform sample of ``max_masks``.

    Support sets are always ordered by ascending drug index (the versioned inhibitor order).
    """
    total = n_masks(len(obs), k)
    if total == 0:
        return []
    if max_masks is None or total <= max_masks:
        out = []
        for q in obs:
            rest = [j for j in obs if j != q]
            out.extend((int(q), tuple(int(j) for j in s)) for s in combinations(rest, k))
        return out
    chosen: dict = {}
    while len(chosen) < max_masks:
        q = int(obs[rng.integers(len(obs))])
        rest = obs[obs != q]
        s = tuple(sorted(int(j) for j in rng.choice(rest, size=k, replace=False)))
        chosen.setdefault((q, s), None)
    return sorted(chosen)


@dataclass
class EpisodeSet:
    panel: str
    k: int
    i: np.ndarray            # (E,) variant index
    q: np.ndarray            # (E,) query drug index
    s: np.ndarray            # (E, k) support drug indices, ascending
    ids: np.ndarray          # (E,) stable episode ids
    rule: str                # support-selection rule, recorded with the manifest

    def __len__(self):
        return len(self.i)

    def inputs(self, panel: Panel) -> dict:
        """Permitted model inputs: identities and observed support responses. No hidden query value."""
        obs = panel.y[self.i[:, None], self.s]
        cens = panel.censor[self.i[:, None], self.s]
        return {"i": self.i, "q": self.q, "s": self.s, "support_y": obs, "support_censor": cens}

    def truth(self, panel: Panel) -> dict:
        """Hidden query response and its censor flag. Call only inside the scoring join."""
        return {"y": panel.y[self.i, self.q], "censor": panel.censor[self.i, self.q]}

    def subset(self, mask: np.ndarray) -> "EpisodeSet":
        return EpisodeSet(self.panel, self.k, self.i[mask], self.q[mask], self.s[mask], self.ids[mask], self.rule)


def build_episodes(panel: Panel, k: int, max_masks: int | None | str = "spec", seed: int = SEED) -> EpisodeSet:
    if k not in BUDGETS:
        raise ValueError("budget must be 2 or 3")
    if k > panel.n_drugs - 1:
        raise PanelValidationError(f"{panel.name}: k={k} leaves no query inhibitor.")
    if max_masks == "spec":
        max_masks = panel.spec.get("max_masks_per_profile", {}).get(str(k))
    I, Q, S, ids = [], [], [], []
    for i in range(panel.n):
        obs = np.flatnonzero(panel.observed[i])
        rng = _profile_rng(seed, panel.name, k, panel.labels[i])
        for q, s in enumerate_profile_masks(obs, k, max_masks, rng):
            I.append(i); Q.append(q); S.append(s)
            ids.append(episode_id(panel.name, k, panel.labels[i], panel.drugs, q, s))
    if not I:
        raise PanelValidationError(f"{panel.name}: no eligible episodes at k={k}.")
    rule = ("all eligible masks over measured cells" if max_masks is None else
            f"uniform sample of {max_masks} masks per profile, seed {seed}, per-profile SHA-256 stream")
    ep = EpisodeSet(panel.name, k, np.array(I, dtype=np.int32), np.array(Q, dtype=np.int16),
                    np.array(S, dtype=np.int16).reshape(len(I), k), np.array(ids), rule)
    if len(set(ep.ids.tolist())) != len(ep):
        raise PanelValidationError(f"{panel.name}: episode id collision.")
    if (ep.s == ep.q[:, None]).any():
        raise PanelValidationError(f"{panel.name}: query appears in its own support set.")
    return ep


def expected_full_episodes(n_profiles: int, n_drugs: int, k: int) -> int:
    """``D x C(D-1, k)`` masks per complete profile."""
    return n_profiles * n_masks(n_drugs, k)


# --------------------------------------------------------------------------------------
# Manifests, audit and hashing
# --------------------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_text(panel: Panel, ep: EpisodeSet, include_inputs: bool) -> str:
    """Canonical CSV text. Never contains the hidden query response."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    head = ["episode_id", "study", "k", "variant", "site", "fold", "query"] + [f"support{j + 1}" for j in range(ep.k)]
    if include_inputs:
        head += [f"y_support{j + 1}" for j in range(ep.k)] + [f"c_support{j + 1}" for j in range(ep.k)]
    w.writerow(head)
    for e in range(len(ep)):
        i = int(ep.i[e])
        row = [ep.ids[e], panel.name, ep.k, panel.labels[i], site_key(panel.sites[i]), int(panel.folds[i]),
               panel.drugs[ep.q[e]]] + [panel.drugs[j] for j in ep.s[e]]
        if include_inputs:
            row += [repr(float(panel.y[i, j])) for j in ep.s[e]] + [int(panel.censor[i, j]) for j in ep.s[e]]
        w.writerow(row)
    return buf.getvalue()


def write_gz_deterministic(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        gz.write(data)
    return sha256_bytes(data)


def write_manifest(panel: Panel, ep: EpisodeSet, directory: Path, include_inputs: bool) -> dict:
    text = manifest_text(panel, ep, include_inputs)
    path = directory / f"{panel.name}_k{ep.k}_manifest.csv.gz"
    digest = write_gz_deterministic(path, text)
    return {"panel": panel.name, "k": ep.k, "episodes": len(ep), "file": path.name, "sha256_uncompressed": digest,
            "rule": ep.rule}


def write_truth(panel: Panel, ep: EpisodeSet, directory: Path) -> str:
    """Private truth file for the scoring join (lockbox or development)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["episode_id", "y", "censor"])
    t = ep.truth(panel)
    for e in range(len(ep)):
        w.writerow([ep.ids[e], repr(float(t["y"][e])), int(t["censor"][e])])
    return write_gz_deterministic(directory / f"{panel.name}_k{ep.k}_truth.csv.gz", buf.getvalue())


def audit_panel(panel: Panel, episodes: dict | None = None, public: bool = True) -> dict:
    """Admitted/excluded profiles, missing cells, exact/censored counts, drugs and sites per fold."""
    obs = panel.observed
    per_fold = {}
    for f in np.unique(panel.folds):
        m = panel.folds == f
        per_fold[int(f)] = {"profiles": int(m.sum()), "sites": len({site_key(s) for s in panel.sites[m]})}
    out = {
        "panel": panel.name, "role": panel.role, "response": panel.response, "scale_kind": panel.scale_kind,
        "profiles_admitted": panel.n, "profiles_excluded": panel.audit.get("excluded_profiles", 0),
        "exclusion_reasons": panel.audit.get("exclusion_reasons", {}),
        "drugs": list(panel.drugs), "modes": list(panel.modes), "n_drugs": panel.n_drugs,
        "n_sites": panel.n_sites, "n_folds": int(len(per_fold)),
        "folds": per_fold, "fold_seed": SEED,
    }
    if public and panel.role in LOCKBOX_ROLES:
        return {k: out[k] for k in ("panel", "role", "scale_kind", "profiles_admitted", "drugs", "modes", "n_drugs", "n_sites")}
    out.update({
        "missing_cells": int((~obs).sum()), "measured_cells": int(obs.sum()),
        "exact_cells": int(((panel.censor == 0) & obs).sum()),
        "lower_bound_cells": int((panel.censor == 1).sum()), "upper_bound_cells": int((panel.censor == -1).sum()),
        "response_min": float(np.nanmin(panel.y)), "response_max": float(np.nanmax(panel.y)),
        "response_median": float(np.nanmedian(panel.y)),
    })
    if episodes:
        out["episodes"] = {str(k): len(v) for k, v in episodes.items()}
        out["full_enumeration_masks"] = {str(k): int(sum(n_masks(int(o), k) for o in obs.sum(axis=1))) for k in episodes}
    return out


def hash_files(files: list[tuple[str, Path]]) -> str:
    """SHA-256 over ``relpath<TAB>file-sha256`` lines sorted by relpath (OS independent)."""
    lines = "".join(f"{rel}\t{sha256_file(p)}\n" for rel, p in sorted(files, key=lambda t: t[0]))
    return sha256_bytes(lines.encode())


def panel_files(root: Path, name: str) -> list[tuple[str, Path]]:
    base = Path(root) / name
    return [(f"{name}/{p.relative_to(base).as_posix()}", p) for p in sorted(base.rglob("*")) if p.is_file()]


def bundle_hash(registry: dict, roots: dict, roles) -> dict:
    files, per_panel = [], {}
    for name in panels_by_role(registry, roles):
        spec = registry["panels"][name]
        root = roots.get(spec["root"])
        if root is None or not (Path(root) / name).is_dir():
            continue
        pf = panel_files(Path(root), name)
        per_panel[name] = hash_files(pf)
        files += pf
    return {"bundle_sha256": hash_files(files) if files else None, "panels": per_panel}


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def _roots(args) -> dict:
    roots = {"archive": args.archive_root}
    if getattr(args, "lockbox_b_root", None):
        roots["lockbox_b"] = args.lockbox_b_root
    return roots


def cmd_audit(args):
    registry = load_registry(args.registry)
    roles = args.roles.split(",")
    if set(roles) & set(LOCKBOX_ROLES):
        sys.exit("The audit command is for development roles. Use the 'lockbox' command for lockbox panels.")
    reports = {}
    for name in panels_by_role(registry, roles):
        panel = load_panel(registry, name, _roots(args))
        reports[name] = audit_panel(panel)
        print(f"{name}: {panel.n} profiles, {panel.n_drugs} drugs, {panel.n_sites} groups, folds {sorted(reports[name]['folds'])}")
    out = Path(args.out) / "dev_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


def cmd_episodes(args):
    registry = load_registry(args.registry)
    roles = args.roles.split(",")
    if set(roles) & set(LOCKBOX_ROLES):
        sys.exit("The episodes command is for development roles. Use the 'lockbox' command for lockbox panels.")
    out = Path(args.out)
    records, folds = [], []
    for name in panels_by_role(registry, roles):
        panel = load_panel(registry, name, _roots(args))
        for k in panel.budgets():
            ep = build_episodes(panel, k)
            rec = write_manifest(panel, ep, out / "episodes", include_inputs=False)
            records.append(rec)
            print(f"{name} k={k}: {len(ep)} episodes")
        folds += [{"panel": name, "site": s, "fold": f, "variants": n} for s, f, n in fold_table(panel)]
    (out / "dev_episode_manifest_index.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    text = "panel,site,fold,variants\n" + "".join(f"{r['panel']},{r['site']},{r['fold']},{r['variants']}\n" for r in folds)
    (out / "dev_fold_table.csv").write_text(text, encoding="utf-8")
    index_hash = sha256_bytes(("".join(f"{r['panel']}|{r['k']}|{r['sha256_uncompressed']}\n" for r in records)).encode())
    print("development episode manifest hash:", index_hash)
    print("fold table hash:", sha256_bytes(text.encode()))


def cmd_lockbox(args):
    """Build lockbox episode manifests into a private, git-ignored directory. No model is run."""
    registry = load_registry(args.registry)
    private = Path(args.private_dir)
    if REPO_ROOT in private.resolve().parents and not (REPO_ROOT / ".gitignore").read_text().count(private.name):
        sys.exit(f"{private} is inside the repository but is not listed in .gitignore.")
    records, public = [], {}
    for name in panels_by_role(registry, args.roles.split(",")):
        spec = registry["panels"][name]
        if spec["root"] not in _roots(args):
            print(f"skip {name}: root {spec['root']} not provided")
            continue
        panel = load_panel(registry, name, _roots(args), allow_lockbox=True)
        declared = spec.get("declared")
        if declared:
            got = {"variants": panel.n, "sites": panel.n_sites, "drugs": panel.n_drugs}
            bad = {k: (got[k], declared[k]) for k in got if got[k] != declared[k]}
            if bad:
                sys.exit(f"{name}: declared cohort mismatch (got, declared): {bad}")
        epi = {}
        for k in panel.budgets():
            ep = build_episodes(panel, k)
            records.append(write_manifest(panel, ep, private / name, include_inputs=True))
            write_truth(panel, ep, private / name)
            epi[k] = ep
        (private / name / "audit.json").write_text(json.dumps(audit_panel(panel, epi, public=False), indent=2) + "\n")
        public[name] = {"role": panel.role, "variants": panel.n, "sites": panel.n_sites, "drugs": panel.n_drugs,
                        "budgets": list(epi), "episodes": {str(k): len(v) for k, v in epi.items()}}
        print(f"{name}: {public[name]}")
    index_hash = sha256_bytes(("".join(f"{r['panel']}|{r['k']}|{r['sha256_uncompressed']}\n" for r in records)).encode())
    out = Path(args.public_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cohort_counts": public, "lockbox_episode_manifest_sha256": index_hash}, indent=2) + "\n")
    print("lockbox episode manifest hash:", index_hash)


def cmd_hashes(args):
    registry = load_registry(args.registry)
    roots = _roots(args)
    result = {"development": bundle_hash(registry, roots, DEV_ROLES),
              "lockbox_A": bundle_hash(registry, roots, ("lockbox_A",)),
              "lockbox_B": bundle_hash(registry, roots, ("lockbox_B",))}
    print(json.dumps(result, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("audit", cmd_audit), ("episodes", cmd_episodes), ("lockbox", cmd_lockbox), ("hashes", cmd_hashes)):
        s = sub.add_parser(name)
        s.add_argument("--archive-root", required=True)
        s.add_argument("--lockbox-b-root", default=None)
        s.add_argument("--roles", default="core_dev,stress_test" if name in ("audit", "episodes") else "lockbox_A,lockbox_B")
        if name in ("audit", "episodes"):
            s.add_argument("--out", default="splits")
        if name == "lockbox":
            s.add_argument("--private-dir", default="lockbox_private")
            s.add_argument("--public-out", default="splits/lockbox_cohort_counts.json")
        s.set_defaults(fn=fn)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
