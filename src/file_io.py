import os
import re
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def sanitize_json_output(output_str):
    """Cleans up LLM output to ensure valid JSON parsing."""
    cleaned = re.sub(r"^```json", "", output_str.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"^```python", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^```", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def generate_files_from_answers(answers, output_dir: str, model: str):
    if "error" in answers:
        logger.error(f"Error: {answers['error']}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        for nb_id, answer in answers["result"]:
            if type(answer) is str:
                logger.error(f"Error: {answer}")
                continue
            answer = answer.output
            base_path = Path(output_dir) / f"{nb_id}/{model}"
            base_path.mkdir(parents=True, exist_ok=True)

            with open(
                base_path / "requirements.txt", "w", encoding="utf-8", errors="replace"
            ) as f:
                f.write(sanitize_json_output(answer.requirements))

            with open(
                base_path / "train.py", "w", encoding="utf-8", errors="replace"
            ) as f:
                f.write(sanitize_json_output(answer.train))

            with open(
                base_path / "inference.py", "w", encoding="utf-8", errors="replace"
            ) as f:
                f.write(sanitize_json_output(answer.inference))

            # Rely on anchoring the input path mapping from data root to model path
            src_input = Path(output_dir).parent / "data" / nb_id / "input"
            if src_input.exists():
                shutil.copytree(src_input, base_path / "input", dirs_exist_ok=True)
