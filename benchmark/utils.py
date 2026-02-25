import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import dspy
import radon.complexity as cc
import radon.metrics as mi
import radon.raw as raw
from dspy import Prediction

import dspy_config


def load_notebooks(path) -> List[tuple[str, str]]:
    nbs = []
    for filepath in Path(path).rglob("*.ipynb"):
        nb_content = parse_notebook(filepath)
        nb_id = "unknown"
        for part in filepath.parts:
            if part.startswith("nb") and part[2:].isdigit():
                nb_id = part
                break
        nbs.append((nb_id, nb_content))

    print(f"Loaded {len(nbs)} notebooks.")
    return nbs


def sanitize_json_output(output_str):
    """Cleans up LLM output to ensure valid JSON parsing."""
    # Remove markdown code blocks if the LLM adds them
    cleaned = re.sub(r'^```json', '', output_str.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r'^```python', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^```', '', cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def parse_notebook(file_path):
    """Reads a .ipynb file and converts it to a readable string format."""
    with open(file_path, 'r', encoding='utf-8') as f:
        nb_data = json.load(f)

    extracted_text = []
    for cell in nb_data.get('cells', []):
        cell_type = cell.get('cell_type')
        source = "".join(cell.get('source', []))
        if cell_type == 'code':
            extracted_text.append(f"--- CODE CELL ---\n{source}\n")
        elif cell_type == 'markdown':
            extracted_text.append(f"--- MARKDOWN CELL ---\n{source}\n")

    return "\n".join(extracted_text)


def get_model_response(
    model: str,
    api_key: str,
    notebooks: List[tuple[str, str]],
    temperature: float,
    api_base: Optional[str] = None,
) -> dict[str, list[tuple[str, Prediction]]]:
    """Query an LLM for each notebook and return the predictions.

    Args:
        model: LiteLLM model string, e.g. ``"openai/Llama-3.3-70B-Instruct"``.
        api_key: API key for the model provider.
        notebooks: List of (nb_id, notebook string) to process.
        temperature: Sampling temperature.
        api_base: Optional custom API base URL (required for models hosted on
            non-OpenAI endpoints such as the Academic Cloud).
    """
    lm_kwargs = dict(model=model, api_key=api_key, temperature=temperature, num_retries=0, timeout=120)
    if api_base:
        lm_kwargs["api_base"] = api_base

    lm = dspy.LM(**lm_kwargs)

    with dspy.context(lm=lm):
        module = dspy.Predict(dspy_config.CodeGenSignature)

        tasks = []
        for nb_id, nb_content in notebooks:
            try:
                answer = module(notebook=nb_content)
            except Exception as e:
                answer = str(e)

            tasks.append((nb_id, answer))
        tasks = [(nb_id, module(notebook=nb_content)) for nb_id, nb_content in notebooks]

    return {"result": tasks}


def generate_files_from_answers(answers: dict[str, list[tuple[str, Prediction]]] | dict[str, str],
                                output_dir: str, model: str):
    if "error" in answers:
        print(f"Error: {answers['error']}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        for nb_id, answer in answers["result"]:
            answer = answer.output
            base_path = Path(output_dir) / f"{nb_id}/{model}"
            base_path.mkdir(parents=True, exist_ok=True)

            with open(base_path / "requirements.txt", "w") as f:
                f.write(sanitize_json_output(answer.requirements))

            with open(base_path / "train.py", "w") as f:
                f.write(sanitize_json_output(answer.train))

            with open(base_path / "inference.py", "w") as f:
                f.write(sanitize_json_output(answer.inference))

            # Copy the input data for this notebook into the model output folder
            src_input = Path(f"../data/{nb_id}/input")
            if src_input.exists():
                shutil.copytree(src_input, base_path / "input", dirs_exist_ok=True)

"""
def run_docker(models: List[str], output_dir: str, amount: int, dockerfile: str) -> List[
    tuple[float | List[set[int | str]]]]:
    client = docker.from_env()
    base_path = Path(output_dir)
    results = []
    for model in models:
        model_logs = []
        run_successfully = 0
        for n in range(amount):
            try:
                image, logs = client.images.build(path=base_path / f"nb{n}/{model}", dockerfile=dockerfile,
                                                  tag=f"nb{n}_{model}")

                model_logs.append({n, str(logs)})
                run_successfully += 1

            except Exception as e:
                model_logs.append({n, str(e)})

        print(f"{run_successfully}/{amount} Dockerfiles for model: {model} run successfully")
        logs.append((run_successfully / amount, model_logs))

    return logs
"""

def analyze_script_complexity(file_path):
    """
    Analyzes a Python script using radon metrics and returns a complexity score (1-5).

    Args:
        file_path (str): The path to the .py file.

    Returns:
        dict: A dictionary containing the 'final_score' (1-5) and detailed metrics.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except FileNotFoundError:
        return {"error": "File not found"}

    # --- 1. Calculate Maintainability Index (MI) ---
    # MI is a compound metric (0-100) combining Halstead Volume, Complexity, and LOC.
    # Higher is better.
    mi_score = mi.mi_visit(code, multi=True)

    # --- 2. Calculate Cyclomatic Complexity (CC) ---
    # CC measures the number of linearly independent paths (if/else/loops).
    # Lower is better.
    cc_blocks = cc.cc_visit(code)

    # Get average and max complexity
    if cc_blocks:
        avg_cc = sum(block.complexity for block in cc_blocks) / len(cc_blocks)
        max_cc = max(block.complexity for block in cc_blocks)
    else:
        avg_cc = 0
        max_cc = 0

    # --- 3. Raw Metrics (LOC) ---
    raw_metrics = raw.analyze(code)
    loc = raw_metrics.loc

    # --- SCORING ALGORITHM ---

    # Step A: Determine Base Score from Maintainability Index (MI)
    # Radon/VS define MI > 20 as "Maintainable", but for a 1-5 scale we need more granularity.
    if mi_score >= 80:
        score = 5
    elif mi_score >= 60:
        score = 4
    elif mi_score >= 40:
        score = 3
    elif mi_score >= 20:
        score = 2
    else:
        score = 1

    # Step B: Penalize for High Complexity "Hotspots"
    # Even if the file is small (good MI), a single complex function (God object) is bad.
    # Radon Ranks: A(1-5), B(6-10), C(11-20), D(21-30), E(31-40), F(41+)

    if max_cc > 40:  # Rank F
        score = 1  # Immediate failure for dangerous complexity
    elif max_cc > 30:  # Rank E
        score = min(score, 2)  # Cap max score at 2
    elif max_cc > 20:  # Rank D
        score = min(score, 3)  # Cap max score at 3
    elif max_cc > 10:  # Rank C
        score -= 1  # Minor penalty

    # Ensure score stays within 1-5 bounds
    final_score = max(1, min(5, score))

    return {
        "final_score": final_score,
        "details": {
            "maintainability_index": round(mi_score, 2),
            "average_complexity": round(avg_cc, 2),
            "max_complexity": max_cc,
            "loc": loc,
            "rating_description": _get_rating_desc(final_score)
        }
    }


def _get_rating_desc(score):
    descriptions = {
        5: "Excellent - Clean, simple, and highly maintainable.",
        4: "Good - Well structured, minor improvements possible.",
        3: "Fair - Functional but complex; consider refactoring.",
        2: "Poor - Hard to read/maintain; significant refactoring needed.",
        1: "Critical - High risk of bugs; immediate refactoring required."
    }
    return descriptions.get(score, "Unknown")
