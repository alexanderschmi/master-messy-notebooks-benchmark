import ast
import subprocess
import logging
import pandas as pd
from pathlib import Path

from src.evaluator import analyze_script_complexity

logger = logging.getLogger(__name__)


def check_syntax(file_path):
    """Checks if a python file has valid syntax."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        if not code.strip():
            logger.warning(f"File is empty: {file_path}")
            return False

        ast.parse(code)
        return True
    except SyntaxError as e:
        logger.warning(f"Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        logger.warning(f"Error reading {file_path}: {e}")
        return False


def get_dir_structure(directory):
    """Returns a set of relative file paths in a directory."""
    paths = set()
    dir_path = Path(directory)
    if not dir_path.exists():
        return paths
    for p in dir_path.rglob("*"):
        if p.is_file():
            paths.add(p.relative_to(dir_path).as_posix())
    return paths


def compare_artifacts(original_output_dir, generated_dir):
    """
    Compares artifacts generated in the generated_dir vs the original_output_dir.
    Returns a score or match status.
    """
    original_files = get_dir_structure(original_output_dir)
    if not original_files:
        return {
            "artifact_match_score": 0.0,
            "details": "No original artifacts to compare against.",
        }

    generated_files = get_dir_structure(generated_dir)
    original_basenames = {Path(p).name for p in original_files}
    generated_basenames = {Path(p).name for p in generated_files}

    intersection = original_basenames.intersection(generated_basenames)

    score = len(intersection) / len(original_basenames) if original_basenames else 0.0

    return {
        "artifact_match_score": round(score, 2),
        "details": f"Matched {len(intersection)} out of {len(original_basenames)} artifacts.",
    }


def get_inference_strategy(nb_id):
    try:
        import src.inference_tester as tester

        class_name = f"Inference{nb_id.capitalize()}"
        if hasattr(tester, class_name):
            return getattr(tester, class_name)()
    except Exception as e:
        logger.warning(f"Could not load strategy for {nb_id}: {e}")
    return None


def test_generated_inference(inference_py_path, model_dir, prompt):
    """
    Executes the LLM generated inference.py in a subprocess, passing the prompt
    as an argument or via python -c execution importing their inference function.
    """
    logger.info(f"Running LLM inference on {inference_py_path}...")
    import json

    prompt_str = json.dumps(prompt)

    runner_code = f"""
import sys
import json
try:
    from inference import inference
    prompt = json.loads('{prompt_str}')
    result = inference(prompt)
    print(json.dumps({{"result": result}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
    sys.exit(1)
"""
    runner_path = model_dir / "_runner.py"
    with open(runner_path, "w") as f:
        f.write(runner_code)

    try:
        import sys

        result = subprocess.run(
            [sys.executable, "_runner.py"],
            cwd=model_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if runner_path.exists():
            runner_path.unlink()

        if result.returncode == 0:
            out = json.loads(result.stdout)
            if "result" in out:
                return True, out["result"]
        return False, None
    except Exception:
        if runner_path.exists():
            runner_path.unlink()
        return False, None


def test_own_inference(model_dir, strategy, prompt):
    """
    Executes our own predefined InferenceStrategy on the generated artifacts.
    """
    try:
        result = strategy.testInference(model_dir, prompt)
        return True, result
    except Exception as e:
        logger.warning(f"Own inference test failed using artifacts at {model_dir}: {e}")
        return False, None


def run_dynamic_analysis(train_py_path, original_output_dir, model_dir):
    """
    Executes train.py securely (subprocess) without installing requirements.
    Compares the generated artifacts to the original ones.
    """
    get_dir_structure(model_dir)

    logger.info(f"Running dynamic analysis on {train_py_path}...")
    try:
        import sys

        # Run with a timeout of 10 minutes (600 seconds)
        result = subprocess.run(
            [sys.executable, "train.py"],
            cwd=model_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        success = result.returncode == 0
        if not success:
            logger.warning(
                f"Execution failed for {train_py_path}. Return code: {result.returncode}"
            )
            logger.debug(f"Stderr: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.warning(f"Execution timed out for {train_py_path}")
        success = False
    except Exception as e:
        logger.warning(f"Error executing {train_py_path}: {e}")
        success = False

    # Compare artifacts
    matching_result = compare_artifacts(original_output_dir, model_dir)
    matching_result["execution_success"] = success
    return matching_result


def score_pipeline(output_dir, data_dir, target_nb=None, target_model=None):
    """
    Iterates through the output directory, runs static and dynamic analysis,
    and produces a scoring report.
    """
    output_path = Path(output_dir)
    data_path = Path(data_dir)

    results = []

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

        model_dirs = [p.parent for p in nb_dir.rglob("train.py")]

        for model_dir in model_dirs:
            if not model_dir.is_dir():
                continue
            model_name = model_dir.relative_to(nb_dir).as_posix()
            if target_model and model_name != target_model:
                continue

            logger.info(f"Scoring {nb_id} - {model_name}...")

            train_py = model_dir / "train.py"
            inference_py = model_dir / "inference.py"

            row = {
                "notebook_id": nb_id,
                "model": model_name,
                "train_syntax_valid": False,
                "inference_syntax_valid": False,
                "train_complexity_score": 0,
                "inference_complexity_score": 0,
                "train_execution_success": False,
                "artifact_match_score": 0.0,
                "own_inference_success": False,
                "llm_inference_success": False,
                "outputs_match": False,
                "model_score": 0.0,
            }

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
                    # (converting to strings or handling floats carefully but direct == is a simple start)
                    if own_succ and llm_succ and str(own_res) == str(llm_res):
                        row["outputs_match"] = True
                    else:
                        row["outputs_match"] = False

            # Calculate final model score
            score = 0.0
            if row.get("train_syntax_valid", False): score += 10.0
            if row.get("inference_syntax_valid", False): score += 10.0
            if row.get("train_execution_success", False): score += 30.0
            score += 10.0 * row.get("artifact_match_score", 0.0)
            if row.get("own_inference_success", False): score += 20.0
            if row.get("llm_inference_success", False): score += 10.0
            if row.get("outputs_match", False): score += 10.0
            
            row["model_score"] = round(score, 2)

            results.append(row)

    df = pd.DataFrame(results)
    
    # Save raw detailed report
    report_path = output_path / "scoring_report.csv"
    df.to_csv(report_path, index=False)
    logger.info(f"Scoring complete. Detailed report saved to {report_path}")

    # Calculate aggregated report (average score per model)
    if not df.empty and "model_score" in df.columns:
        agg_df = df.groupby("model", as_index=False)["model_score"].mean()
        # Sort by best score descending
        agg_df = agg_df.sort_values(by="model_score", ascending=False).reset_index(drop=True)
        agg_report_path = output_path / "scoring_report_aggregated.csv"
        agg_df.to_csv(agg_report_path, index=False)
        logger.info(f"Aggregated scoring complete. Average model scores saved to {agg_report_path}")


if __name__ == "__main__":
    # Test script standalone
    logging.basicConfig(level=logging.INFO)
    base_dir = Path(__file__).resolve().parent.parent
    score_pipeline(base_dir / "output", base_dir / "data")
