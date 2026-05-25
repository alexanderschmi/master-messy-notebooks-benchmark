"""
scoring.utils
=============
Low-level utilities used by the scoring pipeline.

All public functions follow a consistent return convention:
  - pure data helpers   → return a plain value (bool, set, …)
  - analysis functions  → return a dict with at least the keys
                          documented in their docstring
"""

import ast
import importlib.util
import logging
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

def check_syntax(file_path: Path) -> bool:
    """Return True if *file_path* contains syntactically valid Python."""
    try:
        code = Path(file_path).read_text(encoding="utf-8")
        if not code.strip():
            logger.warning(f"File is empty: {file_path}")
            return False
        ast.parse(code)
        return True
    except SyntaxError as exc:
        logger.warning(f"Syntax error in {file_path}: {exc}")
        return False
    except Exception as exc:
        logger.warning(f"Could not read {file_path}: {exc}")
        return False


def _file_basenames(directory: Path) -> set[str]:
    """Return the set of file *basenames* found (recursively) under *directory*."""
    if not directory.exists():
        return set()
    return {p.name for p in directory.rglob("*") if p.is_file()}


# ---------------------------------------------------------------------------
# Requirements comparison
# ---------------------------------------------------------------------------

def _parse_requirements(file_path: Path) -> set[str]:
    """
    Parse a requirements.txt and return a set of lowercase package names
    (version specifiers are stripped).
    """
    if not file_path.exists():
        return set()
    names: set[str] = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z0-9_\-]+)", line)
        if match:
            names.add(match.group(1).lower())
    return names


def compare_requirements(original_req_path: Path, generated_req_path: Path) -> dict:
    """
    Compare generated requirements.txt against the original.

    Returns
    -------
    dict with keys:
      ``requirements_match_score`` (float 0–1)
      ``details``                  (str)
      ``success``                  (bool)
    """
    original = _parse_requirements(Path(original_req_path))
    if not original:
        return {"requirements_match_score": 0.0, "details": "No original requirements found.", "success": False}

    generated = _parse_requirements(Path(generated_req_path))
    if not generated:
        return {"requirements_match_score": 0.0, "details": "No generated requirements found.", "success": False}

    matched = original & generated
    score = len(matched) / len(original)
    return {
        "requirements_match_score": round(score, 2),
        "details": f"Matched {len(matched)} of {len(original)} requirements.",
        "success": True,
    }


# ---------------------------------------------------------------------------
# Artifact comparison
# ---------------------------------------------------------------------------

def _compare_artifacts(original_output_dir: Path, generated_dir: Path) -> dict:
    """
    Compare file basenames produced by the generated code against the reference outputs.

    Returns
    -------
    dict with keys:
      ``artifact_match_score`` (float 0–1)
      ``details``              (str)
    """
    original_files = _file_basenames(Path(original_output_dir))
    if not original_files:
        return {"artifact_match_score": 0.0, "details": "No reference artifacts to compare against."}

    generated_files = _file_basenames(Path(generated_dir))
    matched = original_files & generated_files
    score = len(matched) / len(original_files)
    return {
        "artifact_match_score": round(score, 2),
        "details": f"Matched {len(matched)} of {len(original_files)} artifacts.",
    }


# ---------------------------------------------------------------------------
# Dynamic analysis (train.py execution)
# ---------------------------------------------------------------------------

def run_dynamic_analysis(train_py_path: Path, original_output_dir: Path, model_dir: Path) -> dict:
    """
    Execute ``train.py`` in a subprocess and compare produced artifacts.

    Returns
    -------
    dict with keys:
      ``execution_success``    (bool)
      ``artifact_match_score`` (float 0–1)
      ``details``              (str)
    """
    logger.info(f"Running dynamic analysis on {train_py_path}...")
    success = False
    try:
        result = subprocess.run(
            [sys.executable, "train.py"],
            cwd=model_dir,
            capture_output=True,
            text=True,
            timeout=600,       # 10 minutes
        )
        success = result.returncode == 0
        if not success:
            logger.warning(f"train.py exited with code {result.returncode} for {train_py_path}")
            logger.debug(f"stderr: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.warning(f"train.py timed out for {train_py_path}")
    except Exception as exc:
        logger.warning(f"Could not execute {train_py_path}: {exc}")

    artifact_result = _compare_artifacts(original_output_dir, model_dir)
    return {
        "execution_success": success,
        **artifact_result,
    }


# ---------------------------------------------------------------------------
# Inference testing
# ---------------------------------------------------------------------------

def get_inference_strategy(nb_id: str):
    """
    Load and instantiate the ``InferenceXxx`` strategy class for *nb_id*,
    returning ``None`` if none is registered.
    """
    try:
        import src.inference.notebooks as nb_strategies
        class_name = f"Inference{nb_id.capitalize()}"
        cls = getattr(nb_strategies, class_name, None)
        if cls is not None:
            return cls()
    except Exception as exc:
        logger.warning(f"Could not load inference strategy for {nb_id}: {exc}")
    return None


def test_generated_inference(inference_py_path: Path, model_dir: Path, prompt) -> tuple[bool, object]:
    """
    Dynamically import the generated ``inference.py`` and call its
    ``inference(prompt)`` function.

    Returns
    -------
    ``(success: bool, result: any)``
    """
    logger.info(f"Running LLM inference: {inference_py_path}")
    try:
        spec = importlib.util.spec_from_file_location("inference", str(Path(inference_py_path).resolve()))
        module = importlib.util.module_from_spec(spec)
        # Run the module's code from its own directory so any relative paths resolve correctly.
        import os
        original_cwd = os.getcwd()
        os.chdir(model_dir)
        try:
            spec.loader.exec_module(module)
            result = module.inference(prompt)
        finally:
            os.chdir(original_cwd)
        return True, result
    except Exception as exc:
        logger.warning(f"Generated inference failed for {inference_py_path}: {exc}")
        return False, None


def test_own_inference(model_dir: Path, strategy, prompt) -> tuple[bool, object]:
    """
    Run the benchmark's own inference strategy against the generated artifacts.

    Returns
    -------
    ``(success: bool, result: any)``
    """
    try:
        result = strategy.testInference(model_dir, prompt)
        return True, result
    except Exception as exc:
        logger.warning(f"Own inference failed for artifacts at {model_dir}: {exc}")
        return False, None
