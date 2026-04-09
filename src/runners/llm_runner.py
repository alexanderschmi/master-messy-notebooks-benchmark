import dspy
import logging
from src.core import dspy_config
from src.core.file_io import generate_files_from_answer
from src.runners.base import BaseRunner

logger = logging.getLogger(__name__)


class _BaseLLMRunner(BaseRunner):
    """Shared LLM execution logic — subclasses define the dspy module to use."""

    RUNNER_NAME: str

    def _build_module(self):
        raise NotImplementedError

    def run(
        self,
        params: dict,
        notebook: tuple[str, str],
        temperature: float,
        save_history: bool,
        output_dir: str,
        complexity: int,
        run: int,
    ) -> None:
        model = params.get("model", params.get("name", "unknown_model"))
        provider = params.get("provider", "openai")
        api_key = params.get("api_key", "")
        api_base = params.get("api_base")

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
            module = self._build_module()
            history_dict = None
            try:
                answer = module(notebook=nb_content)
                logger.info(f"[{model}] Successfully processed notebook {nb_id}")
                if save_history and hasattr(lm, "history") and len(lm.history) > 0:
                    history_dict = lm.history[-1]
            except Exception as e:
                logger.error(f"[{model}] Failed processing notebook {nb_id}: {e}")
                return

        generate_files_from_answer((nb_id, answer, history_dict), output_dir, model, complexity, runner=self.RUNNER_NAME, run=run)


class LLMRunner(_BaseLLMRunner):
    RUNNER_NAME = "simple"

    def _build_module(self):
        return dspy.Predict(dspy_config.CodeGenSignature)


class CoTRunner(_BaseLLMRunner):
    RUNNER_NAME = "cot"

    def _build_module(self):
        return dspy.ChainOfThought(dspy_config.CodeGenSignature)
