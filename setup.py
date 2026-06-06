import argparse
import logging
import subprocess
import sys
from pathlib import Path

from src.core.logger import setup_logger

logger = logging.getLogger(__name__)


def setup_notebooks(data_dir: Path, notebook: str = None):
    """
    For each notebook in data_dir, checks whether data/nbX/output/ exists and is
    non-empty. If not, executes the notebook in-place with nbconvert so that
    outputs are generated.
    """
    nb_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("nb")])
    if not nb_dirs:
        logger.warning("No notebook directories found in data dir.")
        return

    if notebook:
        nb_dirs = [d for d in nb_dirs if d.name == notebook]
        if not nb_dirs:
            logger.error(f"Notebook {notebook} not found in data dir.")
            return

    for nb_dir in nb_dirs:
        output_dir = nb_dir / "output"
        if output_dir.exists() and any(output_dir.rglob("*")):
            logger.info(f"{nb_dir.name}: output already exists, skipping.")
            continue

        nb_files = list(nb_dir.glob("*.ipynb"))
        if not nb_files:
            logger.warning(f"{nb_dir.name}: no .ipynb file found, skipping.")
            continue

        nb_path = nb_files[0]
        logger.info(f"{nb_dir.name}: running {nb_path.name} ...")
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "nbconvert",
                    "--to", "notebook",
                    "--execute",
                    "--inplace",
                    str(nb_path),
                ],
                capture_output=True,
                text=True,
                timeout=1800,
            )
            if result.returncode == 0:
                logger.info(f"{nb_dir.name}: execution complete.")
            else:
                logger.warning(f"{nb_dir.name}: execution failed.\n{result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning(f"{nb_dir.name}: execution timed out.")
        except Exception as e:
            logger.warning(f"{nb_dir.name}: error running notebook: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data notebooks to generate reference outputs.")
    parser.add_argument(
        "--notebook",
        type=str,
        default=None,
        help="Specific notebook ID to run (e.g. nb1). If not set, runs all.",
    )
    args = parser.parse_args()

    log_queue, queue_listener = setup_logger(log_file="setup.log", use_multiprocessing=False)
    queue_listener.start()

    BASE_DIR = Path(__file__).resolve().parent
    data_dir = BASE_DIR / "data"

    setup_notebooks(data_dir, notebook=args.notebook)

    queue_listener.stop()
