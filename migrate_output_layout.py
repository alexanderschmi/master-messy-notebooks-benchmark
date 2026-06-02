#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path

from src.core import build_run_dir


logger = logging.getLogger(__name__)

OLD_RUN_DIR_RE = re.compile(r"^(?P<model>.+)_complexity_(?P<complexity>\d+)_run_(?P<run>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate benchmark outputs from output/<nb>/<runner>/<model>_complexity_<N>_run_<R> "
            "to output/<nb>/<runner>/<complexity>/<run>/<model>."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Benchmark output root")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned moves without changing the filesystem",
    )
    parser.add_argument(
        "--on-conflict",
        choices=["error", "skip", "overwrite"],
        default="error",
        help="How to handle an existing destination directory",
    )
    parser.add_argument(
        "--keep-empty-old-dirs",
        action="store_true",
        help="Do not remove empty legacy runner directories after moves",
    )
    return parser.parse_args()


def parse_legacy_run_dir(path: Path) -> dict | None:
    match = OLD_RUN_DIR_RE.match(path.name)
    if not match:
        return None
    return {
        "model": match.group("model"),
        "complexity": int(match.group("complexity")),
        "run": int(match.group("run")),
    }


def find_legacy_run_dirs(output_dir: Path) -> list[tuple[Path, Path]]:
    migrations: list[tuple[Path, Path]] = []

    for notebook_dir in sorted(output_dir.iterdir()):
        if not notebook_dir.is_dir():
            continue

        for runner_dir in sorted(notebook_dir.iterdir()):
            if not runner_dir.is_dir():
                continue

            for child in sorted(runner_dir.iterdir()):
                if not child.is_dir():
                    continue

                meta = parse_legacy_run_dir(child)
                if meta is None:
                    continue

                destination = build_run_dir(
                    output_dir,
                    notebook_dir.name,
                    runner_dir.name,
                    meta["complexity"],
                    meta["run"],
                    meta["model"],
                )
                migrations.append((child, destination))

    return migrations


def remove_empty_parents(start_dir: Path, stop_dir: Path, dry_run: bool) -> None:
    current = start_dir
    while current != stop_dir and current.exists():
        try:
            next(current.iterdir())
            break
        except StopIteration:
            logger.info("Remove empty directory: %s", current)
            if not dry_run:
                current.rmdir()
            current = current.parent


def apply_migration(source: Path, destination: Path, dry_run: bool, on_conflict: str, keep_empty_old_dirs: bool) -> str:
    if destination.exists():
        if on_conflict == "skip":
            logger.warning("Skip existing destination: %s", destination)
            return "skipped"
        if on_conflict == "overwrite":
            logger.warning("Overwrite existing destination: %s", destination)
            if not dry_run:
                shutil.rmtree(destination)
        else:
            raise FileExistsError(f"Destination already exists: {destination}")

    logger.info("Move %s -> %s", source, destination)
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        if not keep_empty_old_dirs:
            remove_empty_parents(source.parent, source.parents[2], dry_run=False)

    return "moved"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    output_dir = args.output_dir.resolve()

    if not output_dir.exists() or not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    migrations = find_legacy_run_dirs(output_dir)
    moved = 0
    skipped = 0

    logger.info("Legacy run directories found: %d", len(migrations))
    for source, destination in migrations:
        status = apply_migration(
            source,
            destination,
            dry_run=args.dry_run,
            on_conflict=args.on_conflict,
            keep_empty_old_dirs=args.keep_empty_old_dirs,
        )
        if status == "moved":
            moved += 1
        elif status == "skipped":
            skipped += 1

    logger.info("Migration summary: moved=%d skipped=%d dry_run=%s", moved, skipped, args.dry_run)


if __name__ == "__main__":
    main()