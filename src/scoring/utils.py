import os
import ast
import subprocess
import logging
import re
from pathlib import Path
import importlib
from contextlib import contextmanager

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
        import src.inference.notebooks as tester

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
