import argparse
from envyaml import EnvYAML
import logging
import sys
import concurrent.futures
import itertools
from pathlib import Path

from src.core import dspy_config
from src.core.notebook_parser import load_notebooks
from src.core.output_layout import build_run_dir
from src.runners import get_runner, validate_runner
from src.scoring.pipeline import score_pipeline
from src.core.logger import setup_logger, worker_init

logger = logging.getLogger(__name__)

def proxy(params, notebook, temperature, complexity, save_history, output_dir, run_id):
    model_name = params.get("model", params.get("name", "unknown_model"))
    runner = params.get("runner", "simple")
    nb_id, _ = notebook

    base_path = build_run_dir(output_dir, nb_id, runner, complexity, run_id, model_name)
    if base_path.exists():
        logger.info(f"Skipping already generated: {base_path}")
        return
    
    try:
        get_runner(runner).run(
            params=params,
            notebook=notebook,
            temperature=temperature,
            save_history=save_history,
            output_dir=output_dir,
            complexity=complexity,
            run=run_id,
        )

    except Exception as e:
        error_msg = f"Failed on {model_name}: {type(e).__name__} - {str(e)}"
        logger.warning(error_msg)


def run_benchmark(notebooks, models_to_test, temperature, output_dir, complexity, save_history, runs, log_queue=None):
    logger.info(f"Starting Benchmark on {len(models_to_test)} models, {len(notebooks)} notebooks for {runs} runs ...\n")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=5,
        initializer=worker_init if log_queue else None,
        initargs=(log_queue,) if log_queue else ()
    ) as executor:
        for notebook in notebooks:
            logger.info(f"Running models for notebook: {notebook[0]}...")
            for run_id in range(1, runs + 1):
                list(executor.map(proxy, 
                             models_to_test, 
                             itertools.repeat(notebook), 
                             itertools.repeat(temperature), 
                             itertools.repeat(complexity), 
                             itertools.repeat(save_history), 
                             itertools.repeat(output_dir),
                             itertools.repeat(run_id)))

    logger.info("Generation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the notebook benchmark.")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to run the benchmark on the models."
    )
    parser.add_argument(
        "--complexity",
        type=str,
        default='4',
        help="The complexity of the prompt to use (1-5)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yml",
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
        help="Specific notebook ID to run (e.g. nb1). If not set, runs all from data."
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Run the scoring pipeline after generating the model outputs."
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Only run the scoring pipeline without running the model benchmark."
    )
    parser.add_argument(
        "--runner",
        type=str,
        default="simple",
        help="The runner engine to use."
    )
    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Save the prompt and response history for each notebook to a history.json file."
    )
    args = parser.parse_args()

    log_queue, queue_listener = setup_logger(log_file="benchmark.log", use_multiprocessing=True)
    queue_listener.start()

    BASE_DIR = Path(__file__).resolve().parent

    config_path = BASE_DIR / args.config
    try:
        config = EnvYAML(config_path)
    except FileNotFoundError:
        logger.error(f"Config file {config_path} not found.")
        sys.exit(1)

    models_to_test = config["models"]
    if args.model:
        models_to_test = [m for m in models_to_test if m["model"] == args.model]
        if not models_to_test:
            logger.error(f"Model {args.model} not found in config.")
            sys.exit(1)
            
    # Modify models_to_test to include runner and prompt context
    for m in models_to_test:
        m["runner"] = args.runner

    try:
        validate_runner(args.runner)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    temperature = config.get("settings", {}).get("temperature", 0.1)
    complexity = args.complexity

    try:
        prompt_file_path = BASE_DIR / "configs" / "prompt.yml"
        prompts = EnvYAML(str(prompt_file_path))
        if complexity in prompts:
            dspy_config.set_prompt(prompts[complexity])
            for m in models_to_test:
                m["prompt"] = prompts[complexity]
            logger.info(f"Loaded prompt for complexity: '{complexity}'")
        else:
            logger.warning(f"Complexity '{complexity}' not found in prompt.yml. Using default prompt.")
    except Exception as e:
        logger.error(f"Error loading prompt.yml: {e}. Using default prompt.")

    data_dir = BASE_DIR / "data"
    notebooks = load_notebooks(data_dir)
    if args.notebook:
        notebooks = [nb for nb in notebooks if nb[0] == args.notebook]
        if not notebooks:
            logger.error(f"Notebook {args.notebook} not found in data dir.")

    output_dir = str(BASE_DIR / "output")
    
    save_history = args.save_history or config.get("settings", {}).get("save_history", False)

    if not args.score_only:
        logger.info(f"Running models: {[m.get('model', m.get('name')) for m in models_to_test]} against {[nb[0] for nb in notebooks]}")
        run_benchmark(notebooks=notebooks, models_to_test=models_to_test, temperature=temperature, output_dir=output_dir, complexity=complexity, save_history=save_history, runs=args.runs, log_queue=log_queue)
    else:
        logger.info("Score-only mode enabled. Skipping model execution.")

    if args.score or args.score_only:
        logger.info("Initializing scoring pipeline...")
        # Only score what was selected
        target_nb = args.notebook if args.notebook else None
        target_model = args.model if args.model else None
        
        score_pipeline(
            output_dir=output_dir, 
            data_dir=data_dir, 
            target_nb=target_nb, 
            target_model=target_model,
            target_complexity=complexity,
            target_runner=args.runner
        )
        
    queue_listener.stop()

