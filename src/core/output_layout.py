from __future__ import annotations

from pathlib import Path
from typing import Iterator


def build_run_dir(output_root: str | Path, notebook_id: str, runner: str, complexity: int | str, run: int, model: str) -> Path:
    return Path(output_root) / notebook_id / runner / str(complexity) / str(run) / model


def parse_run_dir(model_dir: Path, output_root: str | Path) -> dict | None:
    try:
        rel = model_dir.relative_to(Path(output_root))
    except ValueError:
        return None

    if len(rel.parts) != 5:
        return None

    notebook_id, runner, complexity_str, run_str, model = rel.parts
    try:
        complexity = int(complexity_str)
        run = int(run_str)
    except ValueError:
        return None

    return {
        "notebook_id": notebook_id,
        "runner": runner,
        "complexity": complexity,
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

                for run_dir in sorted(complexity_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue

                    for model_dir in sorted(run_dir.iterdir()):
                        if not model_dir.is_dir():
                            continue

                        meta = parse_run_dir(model_dir, root)
                        if meta is None:
                            continue

                        yield meta["notebook_id"], meta["runner"], model_dir, meta