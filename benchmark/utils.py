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

def get_model_response(model: str, notebooks:List[str], temperature: float) -> List[dspy.Prediction]:
    # 1. Global Configuration
    lm = dspy.LM(model=model, temperature=temperature)
    dspy.configure(lm=lm)
    
    module = dspy.Predict(dspy_config.CodeGenSignature)
    # 4. Create and gather tasks
    tasks = [module(notebook = nb) for nb in notebooks]
    return tasks