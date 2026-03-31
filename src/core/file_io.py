import os
import json
import re
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def sanitize_json_output(output_str):
    """Cleans up LLM output to ensure valid JSON parsing."""
    if not output_str:
        return ""
    cleaned = re.sub(r"^```json", "", output_str.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"^```python", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^```", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[\x08\x0b\x0c]", "", cleaned)
    cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
    return cleaned.strip()

def extract_file_content(field_content, filename, history_dict=None):
    """
    Extracts the file content from DSPy's parsed field, or falls back to
    searching the full raw LLM output if the field is empty or just a reference.
    """
    is_suspicious = False
    if not field_content or len(field_content.strip()) < 100 or "code as above" in field_content.lower():
        is_suspicious = True
        
    if not is_suspicious:
        return sanitize_json_output(field_content)
        
    if history_dict and "outputs" in history_dict and len(history_dict["outputs"]) > 0:
        raw_output = history_dict["outputs"][0]
        
        if isinstance(raw_output, dict):
            raw_output = raw_output.get("text") or raw_output.get("content") or str(raw_output)
            
        blocks_pattern = r"```(?:\w+)?\s*\n(.*?)```"
        blocks = re.findall(blocks_pattern, raw_output, flags=re.DOTALL)
        
        for block in blocks:
            if filename.lower() in block.lower()[:200]:
                return block.strip()

    return sanitize_json_output(field_content)


def generate_files_from_answer(answer, output_dir: str, model: str, complexity: int, runner: str = "simple", run: int = 1):
    """Generates files from answers."""
    os.makedirs(output_dir, exist_ok=True)
    nb_id, answer, history_tuple = answer
    logger.info(f"Saving files for model {model} and notebook {nb_id}...")
    if type(answer) is str:
        logger.error(f"Error: {answer}")
        return
    base_path = Path(output_dir) / f"{nb_id}/{runner}/{model}_complexity_{complexity}_run_{run}"
    base_path.mkdir(parents=True, exist_ok=True)
        
    if history_tuple is not None:
        with open(
            base_path / "history.json", "w", encoding="utf-8"
        ) as f:
            # history_tuple is a dict for the dspy LM interaction
            json.dump(history_tuple, f, indent=2, default=str)
            
    with open(
        base_path / "requirements.txt", "w", encoding="utf-8", errors="replace"
    ) as f:
        f.write(extract_file_content(answer.requirements, "requirements.txt", history_tuple))

    with open(
        base_path / "train.py", "w", encoding="utf-8", errors="replace"
    ) as f:
        f.write(extract_file_content(answer.train, "train.py", history_tuple))

    with open(
        base_path / "inference.py", "w", encoding="utf-8", errors="replace"
    ) as f:
        f.write(extract_file_content(answer.inference, "inference.py", history_tuple))

    # Rely on anchoring the input path mapping from data root to model path
    src_input = Path(output_dir).parent / "data" / nb_id / "input"
    if src_input.exists():
        shutil.copytree(src_input, base_path / "input", dirs_exist_ok=True)
    logger.info(f"Files saved for model {model} and notebook {nb_id}")
