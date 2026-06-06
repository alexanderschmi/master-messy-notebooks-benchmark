import hashlib
import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _extract_nb_id(filepath: Path) -> str:
    for part in filepath.parts:
        if part.startswith("nb") and part[2:].isdigit():
            return part
    return "unknown"


def _should_swap_pair(*, nb_id: str, left_index: int) -> bool:
    seed = f"{nb_id}:{left_index}".encode("utf-8")
    digest = hashlib.sha256(seed).digest()
    return digest[0] % 3 == 0


def _reorder_cells(cells: list[dict], *, nb_id: str, order_mode: str, run_id: int) -> list[dict]:
    if order_mode == "original":
        return cells
    if order_mode != "adjacent-swap":
        raise ValueError(f"Unsupported notebook order mode: {order_mode}")

    reordered = list(cells)
    index = 0
    while index + 1 < len(reordered):
        if _should_swap_pair(nb_id=nb_id, left_index=index):
            reordered[index], reordered[index + 1] = reordered[index + 1], reordered[index]
            index += 2
            continue
        index += 1
    return reordered


def _render_cells(cells: list[dict]) -> str:
    extracted_text = []
    for cell in cells:
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        if cell_type == "code":
            extracted_text.append(f"--- CODE CELL ---\n{source}\n")
        elif cell_type == "markdown":
            extracted_text.append(f"--- MARKDOWN CELL ---\n{source}\n")

    return "\n".join(extracted_text)


def load_notebooks(path, order_mode: str = "original", run_id: int = 1) -> List[tuple[str, str]]:
    """Loads all notebooks from a directory."""
    nbs = []
    for filepath in Path(path).rglob("*.ipynb"):
        nb_id = _extract_nb_id(filepath)
        nb_content = parse_notebook(filepath, order_mode=order_mode, run_id=run_id, nb_id=nb_id)
        nbs.append((nb_id, nb_content))

    logger.info(f"Loaded {len(nbs)} notebooks with order mode '{order_mode}' for run {run_id}.")
    return nbs


def parse_notebook(file_path, order_mode: str = "original", run_id: int = 1, nb_id: str | None = None):
    """Reads a .ipynb file and converts it to a readable string format."""
    with open(file_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    resolved_nb_id = nb_id or _extract_nb_id(Path(file_path))
    cells = _reorder_cells(
        nb_data.get("cells", []),
        nb_id=resolved_nb_id,
        order_mode=order_mode,
        run_id=run_id,
    )
    return _render_cells(cells)
