import dspy
import logging
from typing import List, Optional
from src import dspy_config

logger = logging.getLogger(__name__)


def get_model_response(
    model: str,
    api_key: str,
    notebooks: List[tuple[str, str]],
    temperature: float,
    api_base: Optional[str] = None,
    use_cot: bool = False,
) -> dict:
    """Query an LLM for each notebook and return the predictions."""
    logger.info(f"[{model}] Starting to process {len(notebooks)} notebooks")
    lm_kwargs = dict(
        model=model,
        api_key=api_key,
        temperature=temperature,
        num_retries=0,
        timeout=120,
        max_tokens=8192,
    )
    if api_base:
        lm_kwargs["api_base"] = api_base

    lm = dspy.LM(**lm_kwargs)

    with dspy.context(lm=lm):
        if use_cot:
            module = dspy.ChainOfThought(dspy_config.CodeGenSignature)
        else:
            module = dspy.Predict(dspy_config.CodeGenSignature)

        tasks = []
        for nb_id, nb_content in notebooks:
            logger.info(f"[{model}] Processing notebook {nb_id}")
            try:
                answer = module(notebook=nb_content)
                logger.info(f"[{model}] Successfully processed notebook {nb_id}")
            except Exception as e:
                logger.error(f"[{model}] Failed processing notebook {nb_id}: {e}")
                answer = str(e)

            tasks.append((nb_id, answer))

    logger.info(f"[{model}] Finished processing all notebooks")

    return {"result": tasks}
