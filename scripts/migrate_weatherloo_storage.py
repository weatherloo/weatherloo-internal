#!/usr/bin/env python3
"""Plan/apply idempotent migration to the canonical WEATHERLOO_DATA_ROOT layout.

Default mode is dry-run and prints an explicit source -> destination action plan.
Use --apply to execute copies/links. Migration is resumable and validates files
(size + sha256) before replacing or removing anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Iterable


@dataclass(frozen=True)
class Mapping:
    label: str
    src: Path
    dst: Path


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_data_root(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("WEATHERLOO_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root_from_script() / "data"


def canonical_layout(root: Path) -> dict[str, Path]:
    return {
        "raw": root / "raw",
        "processed": root / "processed",
        "cache": root / "cache",
        "experiments": root / "experiments",
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
        "published": root / "published",
        "tmp": root / "tmp",
    }


def build_mappings(source_root: Path, data_root: Path, repo_root: Path) -> list[Mapping]:
    return [
        Mapping("raw-hrrr", source_root / "hrrr", data_root / "raw" / "hrrr"),
        Mapping("raw-era5", source_root / "era5", data_root / "raw" / "era5"),
        Mapping("raw-observations", source_root / "observations", data_root / "raw" / "observations"),
        Mapping(
            "processed-hrrr-bias",
            source_root / "hrrr_bias_correction",
            data_root / "processed" / "hrrr_bias_correction",
        ),
        Mapping("cache-root", source_root / ".cache", data_root / "cache"),
        Mapping("experiments-unet", source_root / "unet", data_root / "experiments" / "unet"),
        Mapping("repo-cache", repo_root / ".cache", data_root / "cache" / "repo"),
        Mapping("repo-data", repo_root / "data", data_root / "processed" / "repo_data"),
        Mapping("repo-artifacts", repo_root / "artifacts", data_root / "published" / "repo_artifacts"),
    ]


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def load_state(state_path: Path) -> dict[str, dict[str, str | int]]:
    if not state_path.is_file():
        return {}
    return json.loads(state_path.read_text())


def save_state(state_path: Path, state: dict[str, dict[str, str | int]]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n")


def key_for(src: Path, dst: Path) -> str:
    return f"{src} -> {dst}"


def ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def same_file(a: Path, b: Path) -> bool:
    try:
        if a.stat().st_size != b.stat().st_size:
            return False
    except FileNotFoundError:
        return False
    return sha256(a) == sha256(b)


def copy_or_link(src: Path, dst: Path, *, symlink: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def remove_existing_path(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def prune_empty_dirs(start: Path, stop: Path, *, dry_run: bool) -> None:
    cur = start
    while cur != stop and cur.exists():
        try:
            cur.rmdir()
        except OSError:
            break
        cur = cur.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=None,
        help="Canonical WEATHERLOO_DATA_ROOT destination (defaults to env, then repo data/)",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="Legacy source root to migrate from (defaults to WEATHERLOO_LEGACY_DATA_ROOT, else --data-root)",
    )
    parser.add_argument("--repo-root", default=None, help="Repository root for repo-local migration sources")
    parser.add_argument("--apply", action="store_true", help="Execute migration actions (default is dry-run plan)")
    parser.add_argument("--symlink", action="store_true", help="Create symlinks instead of copying")
    parser.add_argument("--replace-existing", action="store_true", help="Replace mismatched destination files")
    parser.add_argument("--remove-source", action="store_true", help="Delete source files after validated copy/link")
    parser.add_argument("--include-repo-local", action="store_true", help="Also migrate repo-local .cache/data/artifacts")
    args = parser.parse_args()

    if args.symlink and args.remove_source:
        raise SystemExit("--symlink and --remove-source cannot be used together")

    dry_run = not args.apply
    data_root = resolve_data_root(args.data_root)
    source_root = (
        Path(args.source_root).expanduser().resolve()
        if args.source_root
        else Path(os.environ.get("WEATHERLOO_LEGACY_DATA_ROOT", str(data_root))).expanduser().resolve()
    )
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else repo_root_from_script()

    layout = canonical_layout(data_root)
    for directory in layout.values():
        ensure_dir(directory, dry_run=dry_run)

    mappings = build_mappings(source_root, data_root, repo_root)
    if not args.include_repo_local:
        mappings = [m for m in mappings if not m.label.startswith("repo-")]

    state_path = data_root / ".migration" / "weatherloo_storage_state.json"
    state = load_state(state_path)

    planned = 0
    copied = 0
    skipped = 0
    conflicts = 0

    print(f"WEATHERLOO_DATA_ROOT={data_root}")
    print(f"LEGACY_SOURCE_ROOT={source_root}")
    print(f"mode={'dry-run' if dry_run else 'apply'} symlink={args.symlink}")
    print("\nPlan:")

    for mapping in mappings:
        if not mapping.src.exists():
            continue
        print(f"\n[{mapping.label}] {mapping.src} -> {mapping.dst}")
        for src_file in iter_files(mapping.src):
            rel = src_file.relative_to(mapping.src)
            dst_file = mapping.dst / rel
            action_key = key_for(src_file, dst_file)
            planned += 1
            record = state.get(action_key)
            dst_present = dst_file.exists() or dst_file.is_symlink()

            if record and dst_present:
                expected_size = int(record.get("size", -1))
                expected_hash = str(record.get("sha256", ""))
                if (
                    src_file.stat().st_size == expected_size
                    and sha256(src_file) == expected_hash
                    and same_file(src_file, dst_file)
                ):
                    print(f"  = resumed {rel}")
                    skipped += 1
                    continue

            if dst_present:
                if same_file(src_file, dst_file):
                    print(f"  = keep {rel}")
                    skipped += 1
                    continue
                if not args.replace_existing:
                    print(f"  ! conflict {rel}")
                    conflicts += 1
                    continue
                print(f"  ~ replace {rel}")
                remove_existing_path(dst_file, dry_run=dry_run)
            else:
                print(f"  + {'link' if args.symlink else 'copy'} {rel}")

            if dry_run:
                continue

            dst_file.parent.mkdir(parents=True, exist_ok=True)
            copy_or_link(src_file, dst_file, symlink=args.symlink, dry_run=False)
            if not same_file(src_file, dst_file):
                raise RuntimeError(f"Validation failed after write: {src_file} -> {dst_file}")

            state[action_key] = {
                "size": src_file.stat().st_size,
                "sha256": sha256(src_file),
            }
            copied += 1

            if args.remove_source:
                if not same_file(src_file, dst_file):
                    raise RuntimeError(f"Refusing to remove unvalidated source: {src_file}")
                src_file.unlink()
                prune_empty_dirs(src_file.parent, mapping.src, dry_run=False)

    if not dry_run:
        save_state(state_path, state)

    print("\nSummary:")
    print(f"  planned={planned}")
    print(f"  copied_or_linked={copied}")
    print(f"  skipped_identical={skipped}")
    print(f"  conflicts={conflicts}")
    if dry_run:
        print("  (dry-run only; re-run with --apply to execute)")
    else:
        print(f"  state_file={state_path}")

    return 1 if conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
