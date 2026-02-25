import pandas as pd
import argparse
import dspy_config
import utils
import yaml
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import itertools


def proxy(params, notebooks, temperature):
    model_name = params["model"]
    try:
        response = utils.get_model_response(
            notebooks=notebooks,
            temperature=temperature,
            **params
        )
        return model_name, response

    except Exception as e:
        error_msg = f"Failed on {model_name}: {type(e).__name__} - {str(e)}"
        print(f"\n[Warning] {error_msg}")

        return model_name, {"error": error_msg}


def run_benchmark(notebooks, models_to_test, temperature):
    print(f"Starting Benchmark on {len(models_to_test)} models ...\n")

    answers = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        answers = list(tqdm(executor.map(proxy,
                                         models_to_test,
                                         itertools.repeat(notebooks),
                                         itertools.repeat(temperature)),
                            total=len(models_to_test)))

    print("Prompts complete")
    print("Saving files in output folder...\n")

    for m, answer in answers:
        utils.generate_files_from_answers(answer, str(Path("../output")), m)

    print("Generation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the notebook benchmark.")
    parser.add_argument(
        "--complexity",
        type=int,
        default=1,
        help="The complexity of the prompt to use (1-5)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to the config.yml file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to run. If not set, runs all from config."
    )
    parser.add_argument(
        "--notebook",
        type=str,
        default=None,
        help="Specific notebook ID to run (e.g. nb1). If not set, runs all from ../data."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    try:
        with open(config_path) as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Config file {config_path} not found.")
        import sys

        sys.exit(1)

    models_to_test = config["models"]
    if args.model:
        models_to_test = [m for m in models_to_test if m["name"] == args.model]
        if not models_to_test:
            print(f"Model {args.model} not found in config.")
            import sys

            sys.exit(1)

    temperature = config.get("settings", {}).get("temperature", 0.1)
    complexity = args.complexity

    try:
        with open(Path("prompt.yml"), "r", encoding="utf-8") as prompt_file:
            prompts = yaml.safe_load(prompt_file)
        if complexity in prompts:
            dspy_config.set_prompt(prompts[complexity])
            print(f"Loaded prompt for complexity: '{complexity}'")
        else:
            print(f"Complexity '{complexity}' not found in prompt.yml. Using default prompt.")
    except Exception as e:
        print(f"Error loading prompt.yml: {e}. Using default prompt.")

    notebooks = utils.load_notebooks(Path("../data"))
    if args.notebook:
        notebooks = [nb for nb in notebooks if nb[0] == args.notebook]
        if not notebooks:
            print(f"Notebook {args.notebook} not found in data dir.")

    print(f"Running models: {[m['model'] for m in models_to_test]} against {[nb[0] for nb in notebooks]}")
    run_benchmark(notebooks=notebooks, models_to_test=models_to_test, temperature=temperature)
