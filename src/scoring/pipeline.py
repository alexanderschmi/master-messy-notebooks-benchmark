import logging
import pandas as pd
from pathlib import Path

from src.scoring.evaluator import analyze_script_complexity
from src.scoring.utils import (
    check_syntax,
    compare_artifacts,
    compare_requirements,
    get_inference_strategy,
    test_generated_inference,
    test_own_inference,
    run_dynamic_analysis,
)

logger = logging.getLogger(__name__)


def score_pipeline(output_dir, data_dir, target_nb=None, target_model=None, target_complexity=None, target_runner=None):
    """
    Iterates through the output directory, runs static and dynamic analysis,
    and produces a scoring report.
    """
    output_path = Path(output_dir)
    data_path = Path(data_dir)

    results = []
    
    report_path = output_path / "scoring_report.csv"
    existing_df = pd.DataFrame()
    if report_path.exists():
        try:
            existing_df = pd.read_csv(report_path, dtype={"complexity": int})
            logger.info(f"Loaded existing report from {report_path}")
        except Exception as e:
            logger.warning(f"Could not load existing report: {e}")

    if not output_path.exists():
        logger.error(f"Output directory {output_dir} does not exist.")
        return

    for nb_dir in output_path.iterdir():
        if not nb_dir.is_dir():
            continue
        nb_id = nb_dir.name
        if target_nb and nb_id != target_nb:
            continue
        
        original_output_dir = data_path / nb_id / "output"

        # The structure is output/nb_id/runner/model_complexity_X
        # We need to find all such directories
        for runner_dir in nb_dir.iterdir():
            if not runner_dir.is_dir():
                continue
            
            runner_name = runner_dir.name
            if target_runner and runner_name != target_runner:
                continue

            for model_dir in runner_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                # Extract model, complexity, and run from name "model_complexity_X_run_Y"
                folder_name = model_dir.name
                if "_complexity_" not in folder_name:
                    continue
                
                run_id = 1
                if "_run_" in folder_name:
                    parts = folder_name.split("_run_")
                    run_id = int(parts[1])
                    folder_without_run = parts[0]
                else:
                    folder_without_run = folder_name

                model_name, complexity_str = folder_without_run.split("_complexity_")
                
                if target_model and model_name != target_model:
                    continue
                if target_complexity and complexity_str != str(target_complexity):
                    continue

                logger.info(f"Scoring {nb_id} - {runner_name} - {model_name} (Complexity {complexity_str})...")

                train_py = model_dir / "train.py"
                inference_py = model_dir / "inference.py"

                row = {
                    "notebook_id": nb_id,
                    "model": model_name,
                    "runner": runner_name,
                    "complexity": int(complexity_str),
                    "run": int(run_id),
                    "train_syntax_valid": False,
                    "inference_syntax_valid": False,
                    "train_complexity_score": 0,
                    "inference_complexity_score": 0,
                    "train_execution_success": False,
                    "artifact_match_score": 0.0,
                    "requirements_match_score": 0.0,
                    "requirements_score": 0.0,
                    "own_inference_success": False,
                    "llm_inference_success": False,
                    "outputs_match": False,
                    "train_score": 0.0,
                    "inference_score": 0.0,
                }

                original_req_path = data_path / nb_id / "requirements.txt"
                generated_req_path = model_dir / "requirements.txt"
                req_res = compare_requirements(original_req_path, generated_req_path)
                row["requirements_match_score"] = req_res["requirements_match_score"]
                row["requirements_score"] = round(row["requirements_match_score"] * 100, 2)

                # Static Analysis
                if train_py.exists():
                    row["train_syntax_valid"] = check_syntax(train_py)
                    if row["train_syntax_valid"]:
                        complexity_res = analyze_script_complexity(train_py)
                        if "error" not in complexity_res:
                            row["train_complexity_score"] = complexity_res.get(
                                "final_score"
                            )

                if inference_py.exists():
                    row["inference_syntax_valid"] = check_syntax(inference_py)
                    if row["inference_syntax_valid"]:
                        complexity_res = analyze_script_complexity(inference_py)
                        if "error" not in complexity_res:
                            row["inference_complexity_score"] = complexity_res.get(
                                "final_score"
                            )

                # Dynamic Analysis
                if train_py.exists() and row["train_syntax_valid"]:
                    dyn_res = run_dynamic_analysis(train_py, original_output_dir, model_dir)
                    row["train_execution_success"] = dyn_res["execution_success"]
                    row["artifact_match_score"] = dyn_res["artifact_match_score"]

                if inference_py.exists() and row["inference_syntax_valid"]:
                    strategy = get_inference_strategy(nb_id)
                    if strategy:
                        prompt = strategy.get_prompt()
                        own_succ, own_res = test_own_inference(model_dir, strategy, prompt)
                        llm_succ, llm_res = test_generated_inference(
                            inference_py, model_dir, prompt
                        )

                        row["own_inference_success"] = own_succ
                        row["llm_inference_success"] = llm_succ

                        # Award 1 point or some metric if outputs strictly match
                        if own_succ and llm_succ and str(own_res) == str(llm_res):
                            row["outputs_match"] = True
                        else:
                            row["outputs_match"] = False

                # Calculate final scores on a 0-100 scale for each
                train_score = 0.0
                if row.get("train_syntax_valid", False): train_score += 20.0
                if row.get("train_execution_success", False): train_score += 60.0
                train_score += 20.0 * row.get("artifact_match_score", 0.0)
                
                inference_score = 0.0
                if row.get("inference_syntax_valid", False): inference_score += 20.0
                if row.get("own_inference_success", False): inference_score += 40.0
                if row.get("llm_inference_success", False): inference_score += 30.0
                if row.get("outputs_match", False): inference_score += 10.0
                
                row["train_score"] = round(train_score, 2)
                row["inference_score"] = round(inference_score, 2)

                results.append(row)

    if not results:
        logger.warning("No benchmarked outputs found matching the filters.")
        return

    new_df = pd.DataFrame(results)
    
    if not existing_df.empty:
        # Merge existing and new results
        # We identify a row by notebook_id, runner, model, complexity, and run
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Drop duplicates, keeping the newest one (the one from new_df)
        combined_df = combined_df.drop_duplicates(
            subset=["notebook_id", "runner", "model", "complexity", "run"], 
            keep="last"
        )
        df = combined_df
    else:
        df = new_df
    
    # Save raw detailed report
    df.to_csv(report_path, index=False)
    logger.info(f"Scoring complete. Detailed report updated at {report_path}")

    # Calculate aggregated report (average score per model)
    if not df.empty and "train_score" in df.columns and "inference_score" in df.columns:
        agg_df = df.groupby("model", as_index=False)[["train_score", "inference_score", "requirements_score"]].mean().round(2)
        # Sort by best train score then inference score then requirements score descending
        agg_df = agg_df.sort_values(by=["train_score", "inference_score", "requirements_score"], ascending=[False, False, False]).reset_index(drop=True)
        agg_report_path = output_path / "scoring_report_aggregated.csv"
        agg_df.to_csv(agg_report_path, index=False)
        logger.info(f"Aggregated scoring complete. Average model scores saved to {agg_report_path}")


if __name__ == "__main__":
    # Test script standalone
    file_handler = logging.FileHandler("scoring.log")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, stream_handler]
    )
    base_dir = Path(__file__).resolve().parent.parent
    score_pipeline(base_dir / "output", base_dir / "data")
