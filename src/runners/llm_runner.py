import dspy
import logging
from typing import List, Optional
from src.core import dspy_config
from src.core.file_io import generate_files_from_answer

logger = logging.getLogger(__name__)


def get_model_response(
    model: str,
    provider: str,
    api_key: str,
    notebook: tuple[str, str],
    temperature: float,
    api_base: Optional[str] = None,
    use_cot: bool = False,
    save_history: bool = False,
    output_dir: str = "/output",
    complexity: int = 4,
    run: int = 1,
) -> dict:
    """Query an LLM for a single notebook and return the predictions."""
    nb_id, nb_content = notebook
    logger.info(f"[{model}] Processing notebook {nb_id}")
    lm_kwargs = dict(
        model=provider + "/" + model,
        api_key=api_key,
        temperature=temperature,
        num_retries=0,
        timeout=180,
    )
    if api_base:
        lm_kwargs["api_base"] = api_base

    lm = dspy.LM(**lm_kwargs)

    with dspy.context(lm=lm):
        if use_cot:
            module = dspy.ChainOfThought(dspy_config.CodeGenSignature)
        else:
            module = dspy.Predict(dspy_config.CodeGenSignature)

        history_dict = None
        try:
            answer = module(notebook=nb_content)
            logger.info(f"[{model}] Successfully processed notebook {nb_id}")
            if save_history and hasattr(lm, "history") and len(lm.history) > 0:
                history_dict = lm.history[-1]
        except Exception as e:
            logger.error(f"[{model}] Failed processing notebook {nb_id}: {e}")
            return
    
    generate_files_from_answer((nb_id, answer, history_dict), output_dir, model, complexity, runner="simple", run=run)
