import logging
import pandas as pd
from pathlib import Path

from src.core.output_layout import iter_run_dirs
from src.scoring.evaluator import analyze_script_complexity
from src.scoring.utils import (
    check_syntax,
    compare_requirements,
    get_inference_strategy,
    test_generated_inference,
    test_own_inference,
    run_dynamic_analysis,
)

logger = logging.getLogger(__name__)

SCORE_KEY_COLS = ["notebook_id", "runner", "model", "complexity", "notebook_order", "run"]


def _iter_model_dirs(output_path: Path, target_nb, target_runner, target_model, target_complexity, target_notebook_order):
    """
    Walk ``output_path`` and yield ``(nb_id, runner_name, model_dir, meta)``
    for every model directory that passes all target filters.
    """
    for nb_id, runner_name, model_dir, meta in iter_run_dirs(output_path):
        if target_nb and nb_id != target_nb:
            continue
        if target_runner and runner_name != target_runner:
            continue
        if target_model and meta["model"] != target_model:
            continue
        if target_complexity and meta["complexity"] != int(target_complexity):
            continue
        if target_notebook_order and meta["notebook_order"] != target_notebook_order:
            continue

        yield nb_id, runner_name, model_dir, meta


# ---------------------------------------------------------------------------
# Already-scored check
# ---------------------------------------------------------------------------

def _is_already_scored(
    existing_df: pd.DataFrame,
    nb_id: str,
    runner: str,
    model: str,
    complexity: int,
    notebook_order: str,
    run: int,
) -> bool:
    """Return True if this exact run already exists in the existing report."""
    if existing_df.empty:
        return False
    mask = (
        (existing_df["notebook_id"] == nb_id) &
        (existing_df["runner"] == runner) &
        (existing_df["model"] == model) &
        (existing_df["complexity"] == complexity) &
        (existing_df["notebook_order"] == notebook_order) &
        (existing_df["run"] == run)
    )
    return not existing_df[mask].empty


# ---------------------------------------------------------------------------
# Per-file scoring helpers
# ---------------------------------------------------------------------------

def _score_syntax_and_complexity(py_file: Path) -> dict:
    """Check syntax and measure complexity for a single Python file."""
    syntax_valid = check_syntax(py_file)
    complexity_score = 0
    if syntax_valid:
        res = analyze_script_complexity(py_file)
        if "error" not in res:
            complexity_score = res.get("final_score", 0)
    return {"syntax_valid": syntax_valid, "complexity_score": complexity_score}


def _score_training(train_py: Path, original_output_dir: Path, model_dir: Path) -> dict:
    """Run static + dynamic analysis for train.py."""
    result = _score_syntax_and_complexity(train_py)
    execution_success = False
    artifact_match_score = 0.0

    if result["syntax_valid"]:
        dyn = run_dynamic_analysis(train_py, original_output_dir, model_dir)
        execution_success = dyn["execution_success"]
        artifact_match_score = dyn["artifact_match_score"]

    return {
        "train_syntax_valid": result["syntax_valid"],
        "train_complexity_score": result["complexity_score"],
        "train_execution_success": execution_success,
        "artifact_match_score": artifact_match_score,
    }


def _score_inference(inference_py: Path, model_dir: Path, nb_id: str) -> dict:
    """Run static analysis + inference tests for inference.py."""
    result = _score_syntax_and_complexity(inference_py)
    own_success = llm_success = outputs_match = False

    if result["syntax_valid"]:
        strategy = get_inference_strategy(nb_id)
        if strategy:
            prompt = strategy.get_prompt()
            own_success, own_res = test_own_inference(model_dir, strategy, prompt)
            llm_success, llm_res = test_generated_inference(inference_py, model_dir, prompt)
            outputs_match = own_success and llm_success and str(own_res) == str(llm_res)

    return {
        "inference_syntax_valid": result["syntax_valid"],
        "inference_complexity_score": result["complexity_score"],
        "own_inference_success": own_success,
        "llm_inference_success": llm_success,
        "outputs_match": outputs_match,
    }


def _score_requirements(data_path: Path, nb_id: str, model_dir: Path) -> dict:
    """Compare notebook requirements against generated requirements."""
    original_req = data_path / nb_id / "requirements.txt"
    generated_req = model_dir / "requirements.txt"
    res = compare_requirements(original_req, generated_req)
    match_score = res["requirements_match_score"]
    return {
        "requirements_match_score": match_score,
        "requirements_score": round(match_score * 100, 2),
    }


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _load_existing_report(report_path: Path) -> pd.DataFrame:
    """Load the existing scoring report CSV, returning an empty DataFrame on failure."""
    if not report_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(report_path, dtype={"complexity": int})
        if "notebook_order" not in df.columns:
            df["notebook_order"] = "original"
        logger.info(f"Loaded existing report from {report_path} ({len(df)} rows)")
        return df
    except Exception as exc:
        logger.warning(f"Could not load existing report: {exc}")
        return pd.DataFrame()


def _save_reports(df: pd.DataFrame, output_path: Path) -> None:
    """Persist the detailed and aggregated scoring reports."""
    report_path = output_path / "scoring_report.csv"
    df.to_csv(report_path, index=False)
    logger.info(f"Detailed report saved to {report_path}")

    individual_cols = ["train_execution_success", "own_inference_success", "llm_inference_success"]
    required_cols = individual_cols + ["train_syntax_valid", "inference_syntax_valid", "artifact_match_score", "requirements_match_score"]
    if all(c in df.columns for c in required_cols):
        work = df.copy()

        # Strict AND success flags (0/1) matching visualize.py logic
        work["train_success"] = (
            work["train_syntax_valid"].fillna(False)
            & work["train_execution_success"].fillna(False)
            & work["artifact_match_score"].fillna(0).ge(1.0)
        ).astype(float)

        work["inference_success"] = (
            work["inference_syntax_valid"].fillna(False)
            & work["own_inference_success"].fillna(False)
            & work["llm_inference_success"].fillna(False)
        ).astype(float)

        work["overall_success"] = work["inference_success"] * work["requirements_match_score"].fillna(0)

        agg_cols = individual_cols + ["train_success", "inference_success", "overall_success"]
        agg_df = (
            work.groupby(["model", "runner", "complexity", "notebook_order"], as_index=False)[agg_cols]
            .mean()
            .round(4)
        )
        # Conditional inference rate: P(inference | train succeeded); 0 when train_success = 0
        agg_df["inference_success_cond"] = (
            agg_df["inference_success"] / agg_df["train_success"]
        ).where(agg_df["train_success"] > 0, other=0.0).round(4)

        # Overall: mean of the three independent success dimensions
        req = work.groupby(["model", "runner", "complexity", "notebook_order"], as_index=False)["requirements_match_score"].mean().round(4)
        agg_df = agg_df.merge(req, on=["model", "runner", "complexity", "notebook_order"], how="left")
        agg_df["overall_success"] = (
            (agg_df["train_success"] + agg_df["inference_success_cond"] + agg_df["requirements_match_score"]) / 3
        ).round(4)
        agg_df = (
            agg_df
            .sort_values(by=["overall_success", "train_success", "inference_success_cond"], ascending=False)
            .reset_index(drop=True)
        )
        agg_path = output_path / "scoring_report_aggregated.csv"
        agg_df.to_csv(agg_path, index=False)
        logger.info(f"Aggregated report saved to {agg_path}")


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def score_pipeline(
    output_dir,
    data_dir,
    target_nb=None,
    target_model=None,
    target_complexity=None,
    target_runner=None,
    target_notebook_order=None,
):
    """
    Walk the output directory, score every model run that hasn't been scored
    yet, append results to the existing report, and save both detailed and
    aggregated CSVs.
    """
    output_path = Path(output_dir)
    data_path = Path(data_dir)

    if not output_path.exists():
        logger.error(f"Output directory does not exist: {output_path}")
        return

    report_path = output_path / "scoring_report.csv"
    existing_df = _load_existing_report(report_path)

    new_rows = []

    for nb_id, runner_name, model_dir, meta in _iter_model_dirs(
        output_path, target_nb, target_runner, target_model, target_complexity, target_notebook_order
    ):
        model, complexity, notebook_order, run = meta["model"], meta["complexity"], meta["notebook_order"], meta["run"]

        if _is_already_scored(existing_df, nb_id, runner_name, model, complexity, notebook_order, run):
            logger.info(
                f"Skipping (already scored): {nb_id} / {runner_name} / {model} complexity={complexity} order={notebook_order} run={run}"
            )
            continue

        logger.info(
            f"Scoring: {nb_id} / {runner_name} / {model} complexity={complexity} order={notebook_order} run={run}"
        )

        train_py = model_dir / "train.py"
        inference_py = model_dir / "inference.py"
        original_output_dir = data_path / nb_id / "output"

        row: dict = {
            "notebook_id": nb_id,
            "model": model,
            "runner": runner_name,
            "complexity": complexity,
            "notebook_order": notebook_order,
            "run": run,
        }

        row.update(_score_requirements(data_path, nb_id, model_dir))

        if train_py.exists():
            row.update(_score_training(train_py, original_output_dir, model_dir))
        else:
            row.update({
                "train_syntax_valid": False,
                "train_complexity_score": 0,
                "train_execution_success": False,
                "artifact_match_score": 0.0,
            })

        # Only run inference scoring if training was successful
        if row.get("train_execution_success") and inference_py.exists():
            row.update(_score_inference(inference_py, model_dir, nb_id))
        else:
            row.update({
                "inference_syntax_valid": False,
                "inference_complexity_score": 0,
                "own_inference_success": False,
                "llm_inference_success": False,
                "outputs_match": False,
            })

        new_rows.append(row)

    if not new_rows:
        logger.warning("No new notebooks to score — all entries are already in the report (or no matching outputs found).")
        if not existing_df.empty:
            _save_reports(existing_df, output_path)
        return

    new_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
    _save_reports(combined_df, output_path)


if __name__ == "__main__":
    from src.core.logger import setup_logger
    setup_logger(log_file="scoring.log")
    base_dir = Path(__file__).resolve().parent.parent
    score_pipeline(base_dir / "output", base_dir / "data")
