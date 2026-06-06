from __future__ import annotations

from pathlib import Path
from typing import Iterator


def build_run_dir(
    output_root: str | Path,
    notebook_id: str,
    runner: str,
    complexity: int | str,
    run: int,
    model: str,
    notebook_order: str = "original",
) -> Path:
    return Path(output_root) / notebook_id / runner / str(complexity) / notebook_order / str(run) / model


def resolve_run_dir(
    output_root: str | Path,
    notebook_id: str,
    runner: str,
    complexity: int | str,
    run: int,
    model: str,
    notebook_order: str = "original",
) -> Path:
    preferred = build_run_dir(
        output_root,
        notebook_id,
        runner,
        complexity,
        run,
        model,
        notebook_order=notebook_order,
    )
    if preferred.exists() or notebook_order != "original":
        return preferred

    legacy = Path(output_root) / notebook_id / runner / str(complexity) / str(run) / model
    if legacy.exists():
        return legacy

    return preferred


def parse_run_dir(model_dir: Path, output_root: str | Path) -> dict | None:
    try:
        rel = model_dir.relative_to(Path(output_root))
    except ValueError:
        return None

    if len(rel.parts) == 5:
        notebook_id, runner, complexity_str, run_str, model = rel.parts
        notebook_order = "original"
    elif len(rel.parts) == 6:
        notebook_id, runner, complexity_str, notebook_order, run_str, model = rel.parts
    else:
        return None

    try:
        complexity = int(complexity_str)
        run = int(run_str)
    except ValueError:
        return None

    return {
        "notebook_id": notebook_id,
        "runner": runner,
        "complexity": complexity,
        "notebook_order": notebook_order,
        "run": run,
        "model": model,
    }


def iter_run_dirs(output_root: str | Path) -> Iterator[tuple[str, str, Path, dict]]:
    root = Path(output_root)
    for notebook_dir in sorted(root.iterdir()):
        if not notebook_dir.is_dir():
            continue

        for runner_dir in sorted(notebook_dir.iterdir()):
            if not runner_dir.is_dir():
                continue

            for complexity_dir in sorted(runner_dir.iterdir()):
                if not complexity_dir.is_dir():
                    continue

                for level_dir in sorted(complexity_dir.iterdir()):
                    if not level_dir.is_dir():
                        continue

                    direct_model_dirs = [child for child in sorted(level_dir.iterdir()) if child.is_dir()]
                    if direct_model_dirs and level_dir.name.isdigit():
                        for model_dir in direct_model_dirs:
                            meta = parse_run_dir(model_dir, root)
                            if meta is None:
                                continue

                            yield meta["notebook_id"], meta["runner"], model_dir, meta
                        continue

                    for run_dir in sorted(level_dir.iterdir()):
                        if not run_dir.is_dir():
                            continue

                        for model_dir in sorted(run_dir.iterdir()):
                            if not model_dir.is_dir():
                                continue

                            meta = parse_run_dir(model_dir, root)
                            if meta is None:
                                continue

                            yield meta["notebook_id"], meta["runner"], model_dir, meta