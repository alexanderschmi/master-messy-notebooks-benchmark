import argparse
import yaml
import logging
import logging.handlers
import multiprocessing
import sys
import concurrent.futures
import itertools
from pathlib import Path
from tqdm import tqdm

from src import dspy_config
from src.notebook_parser import load_notebooks
from src.llm_runner import get_model_response
from src.file_io import generate_files_from_answers
from src.scoring_pipeline import score_pipeline

logger = logging.getLogger(__name__)

def worker_init(q):
    """Initialize logging in worker processes."""
    qh = logging.handlers.QueueHandler(q)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(qh)

def proxy(params, notebooks, temperature, use_cot):
    model_name = params.get("model", params.get("name", "unknown_model"))
    try:
        response = get_model_response(
            notebooks=notebooks,
            temperature=temperature,
            api_base=params.get("api_base"),
            use_cot=use_cot,
            model=params.get("model", ""),
            api_key=params.get("api_key", ""),
        )
        return model_name, response

    except Exception as e:
        error_msg = f"Failed on {model_name}: {type(e).__name__} - {str(e)}"
        logger.warning(error_msg)

        return model_name, {"error": error_msg}


def run_benchmark(notebooks, models_to_test, temperature, output_dir, use_cot, log_queue=None):
    logger.info(f"Starting Benchmark on {len(models_to_test)} models ...\n")

    answers = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=10,
        initializer=worker_init if log_queue else None,
        initargs=(log_queue,) if log_queue else ()
    ) as executor:
        answers = list(tqdm(executor.map(proxy,
                                         models_to_test,
                                         itertools.repeat(notebooks),
                                         itertools.repeat(temperature),
                                         itertools.repeat(use_cot)),
                            total=len(models_to_test)))

    logger.info("Prompts complete")
    logger.info("Saving files in output folder...\n")

    for m, answer in answers:
        generate_files_from_answers(answer, output_dir, m)

    logger.info("Generation complete")


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
        "--cot",
        action="store_true",
        help="Use Chain of Thought (dspy.ChainOfThought) instead of standard Predict."
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Run the scoring pipeline after generating the model outputs."
    )
    args = parser.parse_args()

    file_handler = logging.FileHandler("benchmark.log")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, stream_handler]
    )

    m = multiprocessing.Manager()
    log_queue = m.Queue()
    queue_listener = logging.handlers.QueueListener(log_queue, file_handler, stream_handler)
    queue_listener.start()

    BASE_DIR = Path(__file__).resolve().parent

    config_path = BASE_DIR / args.config
    try:
        with open(config_path) as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Config file {config_path} not found.")
        sys.exit(1)

    models_to_test = config["models"]
    if args.model:
        models_to_test = [m for m in models_to_test if m["model"] == args.model]
        if not models_to_test:
            logger.error(f"Model {args.model} not found in config.")
            sys.exit(1)

    temperature = config.get("settings", {}).get("temperature", 0.1)
    complexity = args.complexity

    try:
        prompt_file_path = BASE_DIR / "configs" / "prompt.yml"
        with open(prompt_file_path, "r", encoding="utf-8") as prompt_file:
            prompts = yaml.safe_load(prompt_file)
        if complexity in prompts:
            dspy_config.set_prompt(prompts[complexity])
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
    logger.info(f"Running models: {[m.get('model', m.get('name')) for m in models_to_test]} against {[nb[0] for nb in notebooks]}")
    run_benchmark(notebooks=notebooks, models_to_test=models_to_test, temperature=temperature, output_dir=output_dir, use_cot=args.cot, log_queue=log_queue)

    if args.score:
        logger.info("Initializing scoring pipeline...")
        score_pipeline(output_dir, data_dir)
        
    queue_listener.stop()
