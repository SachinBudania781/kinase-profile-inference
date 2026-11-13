"""Step 5b - integrity: SHA-256 hashes of code, configuration and data, recorded before the lockbox runs.

The record lists what was frozen. ``verify`` recomputes everything and fails if any file differs, so a
result file can be tied to the exact code and data that produced it. Lockbox panels are hashed as files;
their contents are never read here and never copied into the repository.

    python integrity.py record --archive-root <data> --lockbox-b-root <B> --out protocol/integrity_record.json
    python integrity.py verify --record protocol/integrity_record.json --archive-root <data> --lockbox-b-root <B>
    python integrity.py hash-dir runs/            # hash a folder of prediction files before scoring
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import data_pipeline as dp

REPO = Path(__file__).resolve().parent
CODE_GLOBS = ("*.py", "tests/*.py", "configs/*.json", "requirements.txt")
CONFIG_GLOBS = ("configs/*.json",)
EXCLUDED_PARTS = {"__pycache__", "lockbox_private", "LockboxB_Data", "runs", "results", ".git", ".pytest_cache"}


def _collect(repo: Path, globs) -> list[tuple[str, Path]]:
    found = {}
    for pattern in globs:
        for p in sorted(repo.glob(pattern)):
            if p.is_file() and not (EXCLUDED_PARTS & set(p.relative_to(repo).parts)):
                found[p.relative_to(repo).as_posix()] = p
    return sorted(found.items())


def code_hash(repo: Path = REPO) -> dict:
    files = _collect(repo, CODE_GLOBS)
    return {"bundle_sha256": dp.hash_files(files) if files else None,
            "files": {rel: dp.sha256_file(p) for rel, p in files}}


def config_hash(repo: Path = REPO) -> str | None:
    files = _collect(repo, CONFIG_GLOBS)
    return dp.hash_files(files) if files else None


def directory_hash(directory: Path) -> dict:
    directory = Path(directory)
    files = [(p.relative_to(directory).as_posix(), p) for p in sorted(directory.rglob("*")) if p.is_file()]
    return {"bundle_sha256": dp.hash_files(files) if files else None,
            "files": {rel: dp.sha256_file(p) for rel, p in files}}


def compute(repo: Path = REPO, archive_root=None, lockbox_b_root=None, registry_path=None, episode_index=None) -> dict:
    registry = dp.load_registry(registry_path)
    roots = {}
    if archive_root:
        roots["archive"] = archive_root
    if lockbox_b_root:
        roots["lockbox_b"] = lockbox_b_root
    rec = {"schema": 1, "seed": dp.SEED, "code": code_hash(repo), "configs_sha256": config_hash(repo)}
    if roots:
        rec["development_data"] = dp.bundle_hash(registry, roots, dp.DEV_ROLES)
        rec["lockbox_A"] = dp.bundle_hash(registry, roots, ("lockbox_A",))
        rec["lockbox_B"] = dp.bundle_hash(registry, roots, ("lockbox_B",))
    index = Path(episode_index) if episode_index else repo / "splits" / "dev_episode_manifest_index.json"
    if index.exists():
        records = json.loads(index.read_text())
        rec["development_episode_manifest_sha256"] = dp.sha256_bytes(
            "".join(f"{r['panel']}|{r['k']}|{r['sha256_uncompressed']}\n" for r in records).encode())
    fold_table = (Path(episode_index).parent if episode_index else repo / "splits") / "dev_fold_table.csv"
    if fold_table.exists():
        rec["development_fold_table_sha256"] = dp.sha256_file(fold_table)
    return rec


def compare(recorded: dict, current: dict) -> list[str]:
    """Return human-readable differences (empty list means everything matches)."""
    problems = []
    if recorded.get("configs_sha256") != current.get("configs_sha256"):
        problems.append("configuration files differ from the record")
    rc, cc = recorded["code"], current["code"]
    if rc["bundle_sha256"] != cc["bundle_sha256"]:
        for rel in sorted(set(rc["files"]) | set(cc["files"])):
            a, b = rc["files"].get(rel), cc["files"].get(rel)
            if a != b:
                problems.append(f"code file {'added' if a is None else 'removed' if b is None else 'changed'}: {rel}")
    for key in ("development_data", "lockbox_A", "lockbox_B"):
        if key not in recorded:
            continue
        if key not in current or (current[key]["bundle_sha256"] is None and recorded[key]["bundle_sha256"] is not None):
            problems.append(f"{key}: data root not provided or panels missing, not checked")
            continue
        if recorded[key]["bundle_sha256"] != current[key]["bundle_sha256"]:
            panels = set(recorded[key]["panels"]) | set(current[key]["panels"])
            diff = sorted(n for n in panels if recorded[key]["panels"].get(n) != current[key]["panels"].get(n))
            problems.append(f"{key} data differ from the record: {', '.join(diff) or 'bundle hash'}")
    for key in ("development_episode_manifest_sha256", "development_fold_table_sha256"):
        if key in recorded and recorded[key] != current.get(key):
            problems.append(f"{key} differs from the record")
    return problems


def cmd_record(args):
    rec = compute(Path(args.repo), args.archive_root, args.lockbox_b_root, args.registry, args.episode_index)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("code bundle:", rec["code"]["bundle_sha256"])
    for key in ("configs_sha256", "development_episode_manifest_sha256", "development_fold_table_sha256"):
        print(f"{key}:", rec.get(key))
    for key in ("development_data", "lockbox_A", "lockbox_B"):
        if key in rec:
            print(f"{key}:", rec[key]["bundle_sha256"])
    print("written", out)


def cmd_verify(args):
    recorded = json.loads(Path(args.record).read_text())
    current = compute(Path(args.repo), args.archive_root, args.lockbox_b_root, args.registry, args.episode_index)
    problems = compare(recorded, current)
    if problems:
        print("INTEGRITY CHECK FAILED")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("integrity check passed: code, configuration and data match the record")


def cmd_hash_dir(args):
    h = directory_hash(Path(args.directory))
    print(json.dumps(h, indent=2) if args.verbose else h["bundle_sha256"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("record", cmd_record), ("verify", cmd_verify)):
        s = sub.add_parser(name)
        s.add_argument("--repo", default=str(REPO))
        s.add_argument("--archive-root", default=None)
        s.add_argument("--lockbox-b-root", default=None)
        s.add_argument("--episode-index", default=None)
        s.add_argument("--out" if name == "record" else "--record", required=True)
        s.set_defaults(fn=fn)
    s = sub.add_parser("hash-dir")
    s.add_argument("directory")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(fn=cmd_hash_dir)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
