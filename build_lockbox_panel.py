"""Turn a transcribed published IC50 table into a lockbox-format panel folder.

The folder layout is the same as the Lockbox A panels (``SER_responses.csv``, ``source_IC50.csv``, ``EDA.json``,
``study.json``). The generic builder lives in the repository; the transcribed source tables and the panel
folders live outside it (``LockboxB_Data/``, git-ignored).

Response scale (same as every lockbox panel): ``log10(IC50_variant / IC50_native)`` per inhibitor, where the
native reference is the wild-type row measured in the same table. Censoring follows the source literal:
``>x`` is a lower bound (+1), ``<x`` an upper bound (-1), a plain number is exact (0). The response of a bounded
cell is the boundary ratio. A variant is admitted only if every inhibitor of the panel has a numeric value.

    python build_lockbox_panel.py --spec LockboxB_Data/_specs/Villa2024_ALK.json --out LockboxB_Data
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

LITERAL = re.compile(r"^\s*(?P<rel>[<>≥≤~=]?)\s*(?P<num>\d[\d,]*(?:\.\d+)?|\.\d+)\s*(?:±.*)?(?:[\^*†‡a-z]{0,3})\s*$")
MISSING = {"", "na", "n/a", "nd", "n.d", "n.d.", "-", "–", "—", "nt", "ns"}


class BuildError(ValueError):
    pass


def parse_literal(text: str):
    """Return (value, relation) with relation in {'=', '>', '<'}; None for a missing value."""
    t = str(text).strip()
    if t.lower() in MISSING:
        return None
    m = LITERAL.match(t.replace(" ", " "))
    if not m:
        raise BuildError(f"cannot parse IC50 literal: {text!r}")
    rel = {"": "=", "=": "=", "~": "=", ">": ">", "≥": ">", "<": "<", "≤": "<"}[m.group("rel")]
    return float(m.group("num").replace(",", "")), rel


def variant_position(label: str, rule: str = "first_position") -> int:
    """Residue number of the first or last listed mutation. An ``@background`` suffix (for example ``G2032R@CD74``
    for a variant measured on a given fusion partner) names the construct and is ignored for the position."""
    nums = [int(x) for x in re.findall(r"(\d+)", label.split("@")[0])]
    if not nums:
        raise BuildError(f"no residue number in variant label {label!r}")
    if rule == "first_position":
        return nums[0]
    if rule == "last_position":
        return nums[-1]
    raise BuildError(f"unknown site rule {rule}")


def clean_variant(label: str) -> str:
    return re.sub(r"\s*\+\s*", "+", re.sub(r"[\^*†‡]|\s*\(.*?\)$", "", label.strip())).replace(" ", "")


def build(spec: dict):
    """Parse a transcribed table. ``native_label`` is the wild-type row; panels measured on several backgrounds
    (for example two fusion partners) give ``native_labels`` (all native rows) and ``native_of`` (variant -> its native row);
    a variant without an entry in ``native_of`` is scaled by the first native row."""
    drugs = spec["drugs"]
    unit_factor = {"nM": 1.0, "uM": 1000.0, "µM": 1000.0, "μM": 1000.0}[spec.get("unit", "nM")]
    native_labels = [clean_variant(x) for x in (spec.get("native_labels") or [spec.get("native_label", "WT")])]
    native_of = {clean_variant(k): clean_variant(v) for k, v in spec.get("native_of", {}).items()}
    excluded = {clean_variant(k): v for k, v in spec.get("exclude", {}).items()}
    source_rows, natives, kept = [], {}, []
    for raw in spec["rows"]:
        label, cells = raw[0], raw[1:]
        if len(cells) != len(drugs):
            raise BuildError(f"{label}: {len(cells)} values for {len(drugs)} drugs")
        parsed = [parse_literal(c) for c in cells]
        name = clean_variant(label)
        for d, c, p in zip(drugs, cells, parsed):
            if p is not None:
                source_rows.append((name, d, p[0] * unit_factor, p[1], str(c).strip()))
        match = [n for n in native_labels if name.upper() == n.upper()]
        if match:
            natives[match[0]] = parsed
            continue
        kept.append((name, parsed))
    for n in native_labels:
        if n not in natives:
            raise BuildError(f"native row {n!r} not found")
        for d, p in zip(drugs, natives[n]):
            if p is None or p[1] != "=":
                raise BuildError(f"native reference {n!r} for {d} is missing or censored; panel cannot be scaled")
    for v, n in native_of.items():
        if n not in natives:
            raise BuildError(f"native_of maps {v!r} to unknown native row {n!r}")
    admitted, dropped = [], []
    for name, parsed in kept:
        if name in excluded:
            dropped.append((name, excluded[name]))
        elif any(p is None for p in parsed):
            dropped.append((name, "incomplete profile (missing inhibitor value)"))
        else:
            admitted.append((name, parsed))
    resp = []
    for name, parsed in admitted:
        native = natives[native_of.get(name, native_labels[0])]
        for d, p, n in zip(drugs, parsed, native):
            ratio = (p[0] * unit_factor) / (n[0] * unit_factor)
            resp.append({"variant": name, "kinase": spec["kinase"], "position": variant_position(name, spec.get("site_rule", "first_position")),
                         "drug": d, "response": math.log10(ratio), "censor": {"=": 0, ">": 1, "<": -1}[p[1]]})
    return resp, source_rows, dropped


def eda(resp, source, drugs):
    names = list(dict.fromkeys(r["variant"] for r in resp))
    mat = np.array([[next(r["response"] for r in resp if r["variant"] == v and r["drug"] == d) for d in drugs] for v in names])
    flags = {}
    for r in resp:
        flags[str(r["censor"])] = flags.get(str(r["censor"]), 0) + 1
    per_drug = {}
    for j, d in enumerate(drugs):
        x = mat[:, j]
        per_drug[d] = {"finite": int(np.isfinite(x).sum()), "min": float(x.min()), "q25": float(np.percentile(x, 25)),
                       "median": float(np.median(x)), "q75": float(np.percentile(x, 75)), "max": float(x.max())}
    rho = np.eye(len(drugs))
    for a in range(len(drugs)):
        for b in range(a + 1, len(drugs)):
            c = spearmanr(mat[:, a], mat[:, b])[0] if len(names) > 2 else float("nan")
            rho[a, b] = rho[b, a] = float(c) if np.isfinite(c) else 0.0
    ic = np.array([s[2] for s in source], dtype=float)
    return {"SER_responses.csv": {"profiles": len(names), "unique_profiles": len({tuple(np.round(m, 9)) for m in mat}), "drugs": len(drugs),
                                   "matrix_cells": int(mat.size), "missing_response_cells": 0, "per_drug": per_drug,
                                   "groups": len({r["position"] for r in resp}), "censor_flags": flags,
                                   "spearman_drug_order": list(drugs), "spearman_response": rho.tolist()},
            "source_IC50.csv": {"rows": len(source), "missing_by_column": {},
                                 "numeric": {"reported_ic50": {"finite": int(np.isfinite(ic).sum()), "min": float(ic.min()),
                                                                "q25": float(np.percentile(ic, 25)), "median": float(np.median(ic)),
                                                                "q75": float(np.percentile(ic, 75)), "max": float(ic.max())}},
                                 "distinct": {"construct": len({s[0] for s in source}), "drug": len({s[1] for s in source})}}}


def write_panel(spec: dict, out_root: Path) -> Path:
    resp, source, dropped = build(spec)
    folder = Path(out_root) / spec["study"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "SER_responses.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["variant", "kinase", "position", "drug", "response", "censor"])
        for r in resp:
            w.writerow([r["variant"], r["kinase"], r["position"], r["drug"], repr(float(r["response"])), r["censor"]])
    admitted = {r["variant"] for r in resp}
    with (folder / "source_IC50.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["construct", "drug", "reported_ic50", "relation", "source_literal", "in_SER_cohort"])
        for c, d, v, rel, lit in source:
            w.writerow([c, d, v, rel, lit, c in admitted])
    study = {"study": spec["study"], "kinase": spec["kinase"], "source": spec["source"],
             "ser": {"table": "SER_responses.csv", "drugs": list(spec["drugs"]), "modes": list(spec["modes"]),
                     "response": "log10 mutant/native IC50", "censor": {"-1": "upper_bound", "0": "exact", "1": "lower_bound"}}}
    if spec.get("note"):
        study["note"] = spec["note"]
    (folder / "study.json").write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
    (folder / "EDA.json").write_text(json.dumps(eda(resp, source, spec["drugs"]), indent=2) + "\n", encoding="utf-8")
    (folder / "dropped_variants.json").write_text(json.dumps([{"variant": n, "reason": r} for n, r in dropped], indent=2) + "\n", encoding="utf-8")
    return folder


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    folder = write_panel(spec, Path(args.out))
    resp, _, dropped = build(spec)
    n = len({r["variant"] for r in resp})
    print(f"{spec['study']}: {n} variants, {len({r['position'] for r in resp})} sites, {len(spec['drugs'])} drugs; dropped {len(dropped)} -> {folder}")
    for name, why in dropped:
        print(f"  dropped {name}: {why}")


if __name__ == "__main__":
    main()
