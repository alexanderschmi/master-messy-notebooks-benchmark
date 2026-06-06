import dspy
import logging
import json
import re
from types import SimpleNamespace

import litellm

from src.core import dspy_config
from src.core.file_io import generate_files_from_answer
from src.runners.base import BaseRunner
import time

logger = logging.getLogger(__name__)


def _extract_json_payload(raw_text: str) -> dict | None:
    """Best-effort extraction of a JSON object from model text output."""
    if not raw_text:
        return None

    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    # First attempt: parse as-is.
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    # Second attempt: locate the first complete JSON object.
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidate = text[first : last + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    return None


def _run_text_fallback(
    *,
    provider: str,
    model: str,
    api_key: str,
    api_base: str | None,
    temperature: float,
    notebook_text: str,
    prompt_text: str | None,
) -> tuple[SimpleNamespace, dict | None, dict]:
    """
    Fallback completion path for OpenAI-compatible endpoints that reject
    response_format=json_object.
    """
    system_prompt = prompt_text or dspy_config.CodeGenSignature.__doc__
    user_prompt = (
        "Return ONLY a valid JSON object with exactly these keys: "
        '"train", "inference", "requirements". '\
        "Each value must be a string containing full file contents. "
        "Do not include markdown code fences.\n\n"
        f"Notebook:\n{notebook_text}"
    )

    completion_kwargs = {
        "model": f"{provider}/{model}",
        "api_key": api_key,
        "temperature": temperature,
        "num_retries": 0,
        "timeout": 180,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "text"},
    }

    if api_base:
        completion_kwargs["api_base"] = api_base

    raw_resp = litellm.completion(**completion_kwargs)
    content = raw_resp.choices[0].message.content if raw_resp.choices else ""
    payload = _extract_json_payload(content)

    if payload is None:
        raise ValueError("Fallback completion did not return parseable JSON.")

    answer = SimpleNamespace(
        train=str(payload.get("train", "")),
        inference=str(payload.get("inference", "")),
        requirements=str(payload.get("requirements", "")),
    )

    history = {
        "outputs": [content],
        "fallback": "litellm_text_response_format",
    }
    usage = raw_resp.get("usage") if isinstance(raw_resp, dict) else getattr(raw_resp, "usage", None)
    return answer, usage, history


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
        notebook_order: str,
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

        lm = dspy.LM(cache=False, **lm_kwargs)

        with dspy.context(lm=lm, track_usage=True):
            module = self._build_module()
            history_dict = None
            fallback_usage = None
            try:
                start_time = time.time()
                answer = module(notebook=nb_content)
                end_time = time.time()
                logger.info(f"[{model}] Successfully processed notebook {nb_id}")
                if save_history and hasattr(lm, "history") and len(lm.history) > 0:
                    history_dict = lm.history[-1]
            except Exception as e:
                error_str = str(e)
                if "response_format.type" in error_str and "json_schema' or 'text" in error_str:
                    logger.warning(
                        f"[{model}] Endpoint rejected default structured output format; retrying with text fallback."
                    )
                    try:
                        start_time = time.time()
                        answer, fallback_usage, fallback_history = _run_text_fallback(
                            provider=provider,
                            model=model,
                            api_key=api_key,
                            api_base=api_base,
                            temperature=temperature,
                            notebook_text=nb_content,
                            prompt_text=params.get("prompt"),
                        )
                        end_time = time.time()
                        logger.info(f"[{model}] Successfully processed notebook {nb_id} with fallback path")
                        if save_history:
                            history_dict = fallback_history
                    except Exception as fallback_error:
                        logger.error(
                            f"[{model}] Fallback also failed for notebook {nb_id}: {fallback_error}"
                        )
                        return
                else:
                    logger.error(f"[{model}] Failed processing notebook {nb_id}: {e}")
                    return

        time_taken = end_time - start_time
        usage = answer.get_lm_usage() if hasattr(answer, "get_lm_usage") else fallback_usage

        generate_files_from_answer(
            (nb_id, answer, history_dict),
            output_dir,
            model,
            complexity,
            time_taken,
            usage,
            runner=self.RUNNER_NAME,
            run=run,
            notebook_order=notebook_order,
        )


class LLMRunner(_BaseLLMRunner):
    RUNNER_NAME = "simple"

    def _build_module(self):
        return dspy.Predict(dspy_config.CodeGenSignature)


class CoTRunner(_BaseLLMRunner):
    RUNNER_NAME = "cot"

    def _build_module(self):
        return dspy.ChainOfThought(dspy_config.CodeGenSignature)
