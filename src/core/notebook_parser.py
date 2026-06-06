import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def load_notebooks(path) -> List[tuple[str, str]]:
    """Loads all notebooks from a directory."""
    nbs = []
    for filepath in Path(path).rglob("*.ipynb"):
        nb_content = parse_notebook(filepath)
        nb_id = "unknown"
        for part in filepath.parts:
            if part.startswith("nb") and part[2:].isdigit():
                nb_id = part
                break
        nbs.append((nb_id, nb_content))

    logger.info(f"Loaded {len(nbs)} notebooks.")
    return nbs


def parse_notebook(file_path):
    """Reads a .ipynb file and converts it to a readable string format."""
    with open(file_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    extracted_text = []
    for cell in nb_data.get("cells", []):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        if cell_type == "code":
            extracted_text.append(f"--- CODE CELL ---\n{source}\n")
        elif cell_type == "markdown":
            extracted_text.append(f"--- MARKDOWN CELL ---\n{source}\n")

    return "\n".join(extracted_text)
