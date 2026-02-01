from typing import List
import nbformat
import dspy
import dspy_config
from pathlib import Path


def load_notebooks(path) -> List[str]:
    nbs = []
    print(path)
    for filepath in Path(path).rglob("*.ipynb"):
        print(filepath)
        nb = nbformat.read(filepath, as_version=nbformat.NO_CONVERT)
        nbs.append(nb)

    print(f"Loaded {len(nbs)} notebooks.")
    return nbs

def get_model_response(model: str, api_key: str, notebooks:List[str], temperature: float):
    # 1. Global Configuration
    lm = dspy.LM(model=model, api_key=api_key, temperature=temperature)
    dspy.configure(lm=lm)
    
    module = dspy.Predict(dspy_config.CodeGenSignature)
    # 4. Create and gather tasks
    tasks = [module(notebook = nb) for nb in notebooks]
    return tasks