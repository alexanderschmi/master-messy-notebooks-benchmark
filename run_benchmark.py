import argparse
from envyaml import EnvYAML
import logging
import logging.handlers
import multiprocessing
import sys
import concurrent.futures
import itertools
from pathlib import Path
from tqdm import tqdm

from src.core import dspy_config
from src.core.notebook_parser import load_notebooks
from src.runners.llm_runner import get_model_response
from src.runners.openhands_runner import get_openhands_response
from src.scoring.pipeline import score_pipeline

logger = logging.getLogger(__name__)

def worker_init(q):
    """Initialize logging in worker processes."""
    qh = logging.handlers.QueueHandler(q)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(qh)

def proxy(params, notebook, temperature, use_cot, complexity, save_history, output_dir, run_id):
    provider = params.get("provider", "openai")
    model_name = params.get("model", params.get("name", "unknown_model"))
    runner = params.get("runner", "simple")
    nb_id, _ = notebook

    base_path = Path(output_dir) / f"{nb_id}/{runner}/{model_name}_complexity_{complexity}_run_{run_id}"
    if base_path.exists():
        logger.info(f"Skipping already generated: {base_path}")
    
    try:
        if runner == "agentic":
            get_openhands_response(
                model=model_name,
                provider=provider,
                api_key=params.get("api_key", ""),
                notebook=notebook,
                prompt=params.get("prompt"),
                api_base=params.get("api_base"),
                output_dir=output_dir,
                complexity=complexity,
                run=run_id,
            )
        else:
            get_model_response(
                notebook=notebook,
                temperature=temperature,
                api_base=params.get("api_base"),
                use_cot=use_cot,
                model=model_name,
                provider=provider,
                api_key=params.get("api_key", ""),
                save_history=save_history,
                output_dir=output_dir,
                complexity=complexity,
                run=run_id,
            )

    except Exception as e:
        error_msg = f"Failed on {model_name}: {type(e).__name__} - {str(e)}"
        logger.warning(error_msg)


def run_benchmark(notebooks, models_to_test, temperature, output_dir, use_cot, complexity, save_history, runs, log_queue=None):
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
                             itertools.repeat(use_cot), 
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
        "--cot",
        action="store_true",
        help="Use Chain of Thought (dspy.ChainOfThought) instead of standard Predict."
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
        choices=["simple", "agentic"],
        default="simple",
        help="The runner engine to use."
    )
    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Save the prompt and response history for each notebook to a history.json file."
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
        run_benchmark(notebooks=notebooks, models_to_test=models_to_test, temperature=temperature, output_dir=output_dir, use_cot=args.cot, complexity=complexity, save_history=save_history, runs=args.runs, log_queue=log_queue)
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

