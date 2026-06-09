#!/usr/bin/env python3
"""
Compare generated-artifact inference outputs against real notebook artifacts.

Generates a continuous similarity index (0-1) for each model showing how well
its inferred outputs match the real artifacts' outputs.

Scope:
- Read output/scoring_report.csv
- Keep rows where own_inference_success is true
- For each kept row, run the notebook-specific inference strategy twice:
  1) on generated artifacts under output/nb*/...
  2) on reference artifacts under data/nb*/output
- Compute continuous similarity (0-1) for numeric values using relative difference
  For strings, similarity is 0 or 1 (exact match only)
- Save row-level and model-level (averaged) summary CSVs
- Output statistics on mean similarity and match rates per model
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.output_layout import resolve_run_dir
from src.scoring.utils import get_inference_strategy

logger = logging.getLogger(__name__)


def _bool_like(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _continuous_similarity(a: Any, b: Any) -> float:
    """
    Return a similarity score in [0, 1].

    For numeric values this is scale-aware:
      score = max(0, 1 - |a-b| / max(|a|, |b|, 1))
    For non-numeric values this is exact-match only.
    """
    if str(a) == str(b):
        return 1.0

    try:
        af = float(a)
        bf = float(b)
        denom = max(abs(af), abs(bf), 1.0)
        score = 1.0 - (abs(af - bf) / denom)
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def _outputs_match(a: Any, b: Any, rel_tol: float = 1e-4, abs_tol: float = 1e-6) -> bool:
    """Return True if outputs match exactly or are numerically close."""
    if str(a) == str(b):
        return True
    try:
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    except Exception:
        return False


def _compare_values(
    a: Any,
    b: Any,
    continuous_threshold: float,
    rel_tol: float,
    abs_tol: float,
) -> tuple[float, bool, bool]:
    """
    Compute continuous similarity score and determine match status.

    Returns:
      (similarity: float [0, 1],
       outputs_match: bool — True if exact or numerically close,
       above_threshold: bool — True if similarity >= threshold)
    """
    similarity = _continuous_similarity(a, b)
    match = _outputs_match(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    above_threshold = similarity >= continuous_threshold
    return similarity, match, above_threshold


def _build_generated_model_dir(root: Path, row: pd.Series) -> Path:
    return resolve_run_dir(
        root / "output",
        str(row["notebook_id"]),
        str(row["runner"]),
        int(row["complexity"]),
        int(row["run"]),
        str(row["model"]),
        notebook_order=str(row.get("notebook_order", "original")),
    )


def run_comparison(
    project_root: Path,
    scoring_report: Path,
    detailed_output: Path,
    summary_output: Path,
    continuous_threshold: float,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(scoring_report)

    if "own_inference_success" not in df.columns:
        raise ValueError("Column 'own_inference_success' not found in scoring report")

    passed = df[df["own_inference_success"].map(_bool_like)].copy()

    records: list[dict[str, Any]] = []

    for _, row in passed.iterrows():
        nb_id = str(row["notebook_id"])
        generated_dir = _build_generated_model_dir(project_root, row)
        real_dir = project_root / "data" / nb_id / "output"

        record = {
            "notebook_id": nb_id,
            "runner": str(row["runner"]),
            "model": str(row["model"]),
            "complexity": int(row["complexity"]),
            "notebook_order": str(row.get("notebook_order", "original")),
            "run": int(row["run"]),
            "generated_model_dir": str(generated_dir),
            "real_artifact_dir": str(real_dir),
            "generated_model_dir_exists": generated_dir.exists(),
            "real_artifact_dir_exists": real_dir.exists(),
            "generated_inference_success": False,
            "real_artifact_inference_success": False,
            "generated_result": None,
            "real_result": None,
            "outputs_match": False,
            "continuous_similarity": 0.0,
            "continuous_threshold": continuous_threshold,
            "continuous_match": False,
            "error": "",
        }

        strategy = get_inference_strategy(nb_id)
        if strategy is None:
            record["error"] = f"No inference strategy found for {nb_id}"
            records.append(record)
            continue

        try:
            prompt = strategy.get_prompt()
        except Exception as exc:
            record["error"] = f"Prompt generation failed: {exc}"
            records.append(record)
            continue

        try:
            generated_result = strategy.testInference(generated_dir, copy.deepcopy(prompt))
            record["generated_inference_success"] = True
            record["generated_result"] = generated_result
        except Exception as exc:
            record["error"] = f"generated_failed: {exc}"

        try:
            real_result = strategy.testInference(real_dir, copy.deepcopy(prompt))
            record["real_artifact_inference_success"] = True
            record["real_result"] = real_result
        except Exception as exc:
            previous = record["error"]
            suffix = f"real_failed: {exc}"
            record["error"] = f"{previous} | {suffix}" if previous else suffix

        if record["generated_inference_success"] and record["real_artifact_inference_success"]:
            similarity, match, above_threshold = _compare_values(
                record["generated_result"],
                record["real_result"],
                continuous_threshold=continuous_threshold,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            record["outputs_match"] = match
            record["continuous_similarity"] = round(similarity, 6)
            record["continuous_match"] = above_threshold

        records.append(record)

    detailed_df = pd.DataFrame(records)
    detailed_df.to_csv(detailed_output, index=False)

    if detailed_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "model",
                "total_rows",
                "both_inference_success",
                "outputs_match_count",
                "outputs_match_rate",
                "avg_continuous_similarity",
                "min_continuous_similarity",
                "max_continuous_similarity",
                "continuous_match_count",
                "continuous_match_rate",
            ]
        )
    else:
        detailed_df["_both_success"] = (
            detailed_df["generated_inference_success"] & detailed_df["real_artifact_inference_success"]
        )
        grouped = detailed_df.groupby("model", dropna=False)
        summary_df = grouped.agg(
            total_rows=("model", "size"),
            both_inference_success=("_both_success", "sum"),
            outputs_match_count=("outputs_match", lambda s: int(pd.Series(s).sum())),
            avg_continuous_similarity=("continuous_similarity", "mean"),
            min_continuous_similarity=("continuous_similarity", "min"),
            max_continuous_similarity=("continuous_similarity", "max"),
            continuous_match_count=("continuous_match", lambda s: int(pd.Series(s).sum())),
        ).reset_index()

        summary_df["outputs_match_rate"] = (
            summary_df["outputs_match_count"] / summary_df["total_rows"]
        ).round(4)
        summary_df["continuous_match_rate"] = (
            summary_df["continuous_match_count"] / summary_df["total_rows"]
        ).round(4)
        summary_df["avg_continuous_similarity"] = summary_df["avg_continuous_similarity"].round(6)
        summary_df["min_continuous_similarity"] = summary_df["min_continuous_similarity"].round(6)
        summary_df["max_continuous_similarity"] = summary_df["max_continuous_similarity"].round(6)

        summary_df = summary_df.sort_values(
            ["avg_continuous_similarity", "continuous_match_rate", "total_rows"],
            ascending=[False, False, False],
        )

    summary_df.to_csv(summary_output, index=False)
    return detailed_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare generated-artifact inference outputs against real notebook artifacts "
            "for rows where own_inference_success=True."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Repository root (default: current directory)",
    )
    parser.add_argument(
        "--scoring-report",
        type=Path,
        default=Path("output/scoring_report.csv"),
        help="Path to scoring_report.csv",
    )
    parser.add_argument(
        "--detailed-output",
        type=Path,
        default=Path("output/own_inference_vs_real_artifacts.csv"),
        help="Path for row-level output CSV",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("output/own_inference_vs_real_artifacts_summary.csv"),
        help="Path for model-level summary CSV",
    )
    parser.add_argument(
        "--continuous-threshold",
        type=float,
        default=0.1,
        help="Continuous similarity threshold for match classification (default: 0.1, range [0, 1])",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=1e-6,
        help="Relative tolerance for outputs_match (default: 1e-4)",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for outputs_match (default: 1e-6)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    project_root = args.project_root.resolve()
    scoring_report = (project_root / args.scoring_report).resolve()
    detailed_output = (project_root / args.detailed_output).resolve()
    summary_output = (project_root / args.summary_output).resolve()

    detailed_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    detailed_df, summary_df = run_comparison(
        project_root=project_root,
        scoring_report=scoring_report,
        detailed_output=detailed_output,
        summary_output=summary_output,
        continuous_threshold=args.continuous_threshold,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
    )

    print(f"Rows with own_inference_success=True: {len(detailed_df)}")
    print(f"Detailed CSV: {detailed_output}")
    print(f"Summary CSV: {summary_output}")

    if detailed_df.empty:
        print("No rows to compare.")
        return

    print("\n=== Overall Continuous Similarity Stats ===")
    if len(detailed_df) > 0 and "continuous_similarity" in detailed_df.columns:
        similarities = detailed_df["continuous_similarity"]
        print(f"Mean similarity: {similarities.mean():.6f}")
        print(f"Median similarity: {similarities.median():.6f}")
        print(f"Min similarity: {similarities.min():.6f}")
        print(f"Max similarity: {similarities.max():.6f}")
        print(f"Std dev: {similarities.std():.6f}")

    continuous_match_count = detailed_df["continuous_match"].sum() if "continuous_match" in detailed_df.columns else 0
    print(f"\nContinuous matches above threshold ({args.continuous_threshold}): {continuous_match_count}/{len(detailed_df)}")

    if not summary_df.empty:
        print("\n=== Top 10 Models by Avg Continuous Similarity ===")
        print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
