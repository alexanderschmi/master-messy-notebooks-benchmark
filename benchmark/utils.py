from typing import List
import nbformat
import dspy
import dspy_config
from pathlib import Path
import os


def load_notebooks(path) -> List[str]:
    nbs = []
    print(path)
    for filepath in Path(path).rglob("*.ipynb"):
        print(filepath)
        nb = nbformat.read(filepath, as_version=nbformat.NO_CONVERT)
        nbs.append(nb)

    print(f"Loaded {len(nbs)} notebooks.")
    return nbs


def get_model_response(model: str, api_key: str, notebooks: List[str], temperature: float) -> List[dspy.Prediction]:
    # 1. Global Configuration
    lm = dspy.LM(model=model, api_key=api_key, temperature=temperature)
    dspy.configure(lm=lm)

    module = dspy.Predict(dspy_config.CodeGenSignature)
    # 4. Create and gather tasks
    tasks = [module(notebook=nb) for nb in notebooks]
    return tasks


def generate_files_from_answers(answers: List[dspy.Prediction], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for i, answer in enumerate(answers):
        base_path = Path(output_dir) / f"notebook_{i + 1}"
        base_path.mkdir(parents=True, exist_ok=True)

        with open(base_path / "requirements.txt", "w") as f:
            f.write(answer.requirements)

        with open(base_path / "train.py", "w") as f:
            f.write(answer.train)

        with open(base_path / "inference.py", "w") as f:
            f.write(answer.inference)

        with open(base_path / "Dockerfile_train", "w") as f:
            f.write(answer.docker_file_train)

        with open(base_path / "Dockerfile_inference", "w") as f:
            f.write(answer.docker_file_inference)
