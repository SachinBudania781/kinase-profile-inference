"""Run the registered methods on development or lockbox panels and write predictions by episode ID.

Development example (no lockbox access)::

    python run_study.py --archive-root <data> --roles core_dev,stress_test --out runs/dev

Lockbox example (curator step after the release freeze; one shot per lockbox)::

    python run_study.py --archive-root <data> --lockbox-b-root <B> --roles lockbox_A,lockbox_B \
        --allow-lockbox --integrity-record protocol/integrity_record.json --out runs/lockbox

The lockbox path refuses to start unless the integrity record matches the current code and data, and it
refuses to run a second time into the same output folder. Predictions never contain the hidden responses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import core_models as cm
import data_pipeline as dp
import integrity as ig

LOCKBOX_MARKER = "LOCKBOX_RUN_STARTED.json"


def load_roles(registry: dict, roots: dict, roles, allow_lockbox: bool, only=None) -> dict:
    out = {}
    for name in dp.panels_by_role(registry, roles):
        if only and name not in only:
            continue
        spec = registry["panels"][name]
        if spec["root"] not in roots:
            print(f"skip {name}: root '{spec['root']}' not provided")
            continue
        out[name] = dp.load_panel(registry, name, roots, allow_lockbox=allow_lockbox)
    return out


def check_lockbox_gate(args, out: Path, roots: dict):
    if not args.allow_lockbox:
        sys.exit("Lockbox roles need --allow-lockbox (curator step after the release freeze).")
    if not args.integrity_record:
        sys.exit("Lockbox runs need --integrity-record so the frozen code and data are verified first.")
    recorded = json.loads(Path(args.integrity_record).read_text())
    current = ig.compute(Path(args.repo), roots.get("archive"), roots.get("lockbox_b"), args.registry, args.episode_index)
    problems = ig.compare(recorded, current)
    if problems:
        sys.exit("Integrity check failed; no lockbox model was run:\n - " + "\n - ".join(problems))
    marker = out / LOCKBOX_MARKER
    if marker.exists():
        sys.exit(f"{marker} exists: the lockbox has already been run once. Reruns are not part of the protocol.")
    out.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"code_bundle_sha256": current["code"]["bundle_sha256"],
                                  "started_unix": int(time.time())}, indent=2) + "\n")


def run(args) -> dict:
    registry = dp.load_registry(args.registry)
    roots = {"archive": args.archive_root}
    if args.lockbox_b_root:
        roots["lockbox_b"] = args.lockbox_b_root
    roles = args.roles.split(",")
    uses_lockbox = bool(set(roles) & set(dp.LOCKBOX_ROLES))
    out = Path(args.out)
    if uses_lockbox:
        check_lockbox_gate(args, out, roots)
    cfg = cm.load_config(args.config)
    only = set(args.panels.split(",")) if args.panels else None
    panels = load_roles(registry, roots, roles, args.allow_lockbox, only)
    methods = tuple(args.methods.split(",")) if args.methods else cm.ALL_METHODS
    unknown = set(methods) - set(cm.ALL_METHODS)
    if unknown:
        sys.exit(f"unknown methods: {sorted(unknown)}")
    pool = None
    if set(methods) & {"SAER-7g", "AlignedAddAudit", "NestedAligned"}:
        pool_panels = load_roles(registry, roots, dp.DEV_ROLES, False)
        pool = cm.PoolProvider(pool_panels, cfg)
    budgets = [int(b) for b in args.budgets.split(",")]
    manifest = {"seed": dp.SEED, "methods": list(methods), "roles": roles, "runs": [], "errors": []}
    for name, panel in panels.items():
        for k in panel.budgets():
            if k not in budgets:
                continue
            t0 = time.time()
            try:
                result = cm.run_panel(panel, k, cfg, methods, pool, log=print if args.verbose else None)
                info = cm.save_run(result, out)
                info.update({"panel": name, "k": k, "seconds": round(time.time() - t0, 1),
                             "failures": {m: len(v) for m, v in result.failures.items() if v}})
                manifest["runs"].append(info)
                print(f"{name} k={k}: {info['episodes']} episodes in {info['seconds']} s")
            except Exception as exc:                                   # recorded, never silently skipped
                manifest["errors"].append({"panel": name, "k": k, "error": repr(exc), "trace": traceback.format_exc()})
                print(f"{name} k={k}: FAILED {exc!r}", file=sys.stderr)
            finally:
                cm.clear_caches()
    out.mkdir(parents=True, exist_ok=True)
    manifest["predictions_sha256"] = ig.directory_hash(out)["bundle_sha256"]
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive-root", required=True)
    p.add_argument("--lockbox-b-root", default=None)
    p.add_argument("--roles", default="core_dev,stress_test")
    p.add_argument("--panels", default=None, help="comma-separated subset of panel names")
    p.add_argument("--methods", default=None, help="comma-separated; default all registered methods")
    p.add_argument("--budgets", default="2,3")
    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs" / "models.json"))
    p.add_argument("--registry", default=None)
    p.add_argument("--repo", default=str(Path(__file__).resolve().parent))
    p.add_argument("--episode-index", default=None)
    p.add_argument("--out", default="runs/dev")
    p.add_argument("--allow-lockbox", action="store_true")
    p.add_argument("--integrity-record", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    manifest = run(args)
    if manifest["errors"]:
        sys.exit(f"{len(manifest['errors'])} panel/budget run(s) failed; see run_manifest.json")


if __name__ == "__main__":
    main()
