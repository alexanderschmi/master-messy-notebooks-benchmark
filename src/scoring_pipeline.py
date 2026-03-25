import os
import ast
import subprocess
import logging
import pandas as pd
import re
from pathlib import Path
import importlib
from contextlib import contextmanager

from src.evaluator import analyze_script_complexity

logger = logging.getLogger(__name__)

@contextmanager
def run_in_directory(target_directory):
    original_cwd = os.getcwd()
    os.chdir(target_directory)
    try:
        yield
    finally:
        os.chdir(original_cwd)

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


def parse_requirements(file_path):
    if not file_path.exists():
        return set()
    reqs = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle standard formats, ignoring version specifiers like ==, >=, <=, ~, >, <
                match = re.search(r"^([a-zA-Z0-9_\-]+)", line)
                if match:
                    reqs.add(match.group(1).lower())
    return reqs


def compare_requirements(original_req_path, generated_req_path):
    """
    Compares the generated requirements.txt to the original notebook's requirements.txt.
    Returns a match score (0.0 to 1.0) and whether it was successfully parsed.
    """
    original_reqs = parse_requirements(original_req_path)
    if not original_reqs:
        return {
            "requirements_match_score": 0.0,
            "details": "No original requirements found.",
            "success": False
        }

    generated_reqs = parse_requirements(generated_req_path)
    if not generated_reqs:
        return {
            "requirements_match_score": 0.0,
            "details": "No generated requirements found.",
            "success": False
        }

    intersection = original_reqs.intersection(generated_reqs)
    score = len(intersection) / len(original_reqs)

    return {
        "requirements_match_score": round(score, 2),
        "details": f"Matched {len(intersection)} out of {len(original_reqs)} requirements.",
        "success": True
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

    try:
        with run_in_directory(model_dir):
            spec = importlib.util.spec_from_file_location("inference", str(inference_py_path.absolute()))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result = module.inference(prompt)
        return True, result
    except Exception as e:
        logger.warning(f"Error running inference on {inference_py_path}: {e}")
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
