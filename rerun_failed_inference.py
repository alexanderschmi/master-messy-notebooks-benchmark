#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.core.output_layout import build_run_dir
from src.scoring.pipeline import SCORE_KEY_COLS, _save_reports
from src.scoring.pipeline import _score_inference as score_inference

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "notebook_id",
    "model",
    "runner",
    "complexity",
    "run",
    "train_execution_success",
    "artifact_match_score",
    "inference_syntax_valid",
    "own_inference_success",
    "llm_inference_success",
    "outputs_match",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry failed inference-scoring rows and update scoring_report.csv when results improve."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--notebook", type=str, default=None)
    parser.add_argument("--runner", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--complexity", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--failure-mode",
        choices=["own", "llm", "either", "both"],
        default="own",
        help="Which failed inference rows to retry. Default targets benchmark-side loader failures only.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def compute_inference_score(row: pd.Series | dict) -> float:
    inference_syntax_valid = bool(row.get("inference_syntax_valid", False))
    own_inference_success = bool(row.get("own_inference_success", False))
    llm_inference_success = bool(row.get("llm_inference_success", False))
    outputs_match = bool(row.get("outputs_match", False))
    return float(
        (20 if inference_syntax_valid else 0)
        + (40 if own_inference_success else 0)
        + (30 if llm_inference_success else 0)
        + (10 if outputs_match else 0)
    )


def filter_candidates(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    candidates = df.copy()
    candidates = candidates[candidates["train_execution_success"].fillna(False).astype(bool)]
    candidates = candidates[candidates["artifact_match_score"].fillna(0).eq(1.0)]
    candidates = candidates[candidates["inference_syntax_valid"].fillna(False).astype(bool)]

    own_failed = ~candidates["own_inference_success"].fillna(False).astype(bool)
    llm_failed = ~candidates["llm_inference_success"].fillna(False).astype(bool)
    if args.failure_mode == "own":
        inference_failed = own_failed
    elif args.failure_mode == "llm":
        inference_failed = llm_failed
    elif args.failure_mode == "both":
        inference_failed = own_failed & llm_failed
    else:
        inference_failed = own_failed | llm_failed
    candidates = candidates[inference_failed]

    if args.notebook:
        candidates = candidates[candidates["notebook_id"] == args.notebook]
    if args.runner:
        candidates = candidates[candidates["runner"] == args.runner]
    if args.model:
        candidates = candidates[candidates["model"] == args.model]
    if args.complexity is not None:
        candidates = candidates[candidates["complexity"].astype(int) == args.complexity]
    if args.max_runs is not None:
        candidates = candidates.head(args.max_runs)

    return candidates.copy()


def locate_model_dir(output_path: Path, row: pd.Series) -> Path:
    return build_run_dir(
        output_path,
        row["notebook_id"],
        row["runner"],
        int(row["complexity"]),
        int(row["run"]),
        row["model"],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    project_root = args.project_root.resolve()
    output_path = (project_root / args.output_dir).resolve()
    report_path = (project_root / args.report_path).resolve() if args.report_path else output_path / "scoring_report.csv"

    if not report_path.exists():
        raise FileNotFoundError(f"Scoring report not found: {report_path}")

    df = pd.read_csv(report_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"scoring_report.csv is missing required columns: {sorted(missing)}")

    candidates = filter_candidates(df, args)
    logger.info("Retry candidates: %s", len(candidates))
    if candidates.empty:
        return

    improvements = 0
    attempted = 0

    for candidate in candidates.itertuples(index=False):
        row = pd.Series(candidate._asdict())
        model_dir = locate_model_dir(output_path, row)
        inference_py = model_dir / "inference.py"
        if not inference_py.exists():
            logger.warning(
                "Skipping %s/%s/%s complexity=%s run=%s: missing inference.py",
                row["notebook_id"],
                row["runner"],
                row["model"],
                row["complexity"],
                row["run"],
            )
            continue

        attempted += 1
        new_result = score_inference(inference_py, model_dir, row["notebook_id"])
        old_score = float(row.get("inference_score", compute_inference_score(row)))
        new_score = compute_inference_score(new_result)

        if new_score <= old_score:
            logger.info(
                "No improvement for %s/%s/%s complexity=%s run=%s (old=%s, new=%s)",
                row["notebook_id"],
                row["runner"],
                row["model"],
                row["complexity"],
                row["run"],
                old_score,
                new_score,
            )
            continue

        mask = pd.Series(True, index=df.index)
        for key in SCORE_KEY_COLS:
            mask &= df[key] == row[key]

        if not mask.any():
            logger.warning(
                "Could not locate row for %s/%s/%s complexity=%s run=%s",
                row["notebook_id"],
                row["runner"],
                row["model"],
                row["complexity"],
                row["run"],
            )
            continue

        for column, value in new_result.items():
            df.loc[mask, column] = value
        if "inference_score" in df.columns:
            df.loc[mask, "inference_score"] = round(new_score, 2)

        improvements += int(mask.sum())
        logger.info(
            "Improved %s/%s/%s complexity=%s run=%s (old=%s, new=%s)",
            row["notebook_id"],
            row["runner"],
            row["model"],
            row["complexity"],
            row["run"],
            old_score,
            new_score,
        )

    logger.info("Retried %s rows; improved %s rows", attempted, improvements)

    if args.dry_run:
        logger.info("Dry run enabled; not writing updated reports.")
        return

    if improvements:
        _save_reports(df, output_path)
    else:
        logger.info("No rows improved; leaving reports unchanged.")


if __name__ == "__main__":
    main()