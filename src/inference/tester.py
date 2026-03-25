import logging
from pathlib import Path

from src.inference.strategies import InferenceStrategy
import src.inference.notebooks as notebooks

logger = logging.getLogger(__name__)


# 3. Write your unified testing function
def test_artifact(script_instance: InferenceStrategy, path: str | Path = None):
    prompt = script_instance.get_prompt()
    if path:
        result = script_instance.testInference(path, prompt)
    else:
        result = script_instance.testInference("dummy_path", prompt)
    logger.info(f"Generated: {result}")
    return result


# Execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    for i in range(1, 11):
        nb_name = f"nb{i}"
        nb_dir = data_dir / nb_name / "output"

        inference_class_name = f"InferenceNb{i}"

        if hasattr(notebooks, inference_class_name):
            inference_class = getattr(notebooks, inference_class_name)

            if nb_dir.exists():
                logger.info(f"Testing {nb_name} from {nb_dir}")
                try:
                    test_artifact(inference_class(), nb_dir)
                except Exception as e:
                    logger.error(f"Error testing {nb_name}: {e}")
            else:
                logger.warning(f"Output directory not found for {nb_name}: {nb_dir}")
        else:
            logger.warning(f"Inference class {inference_class_name} not found.")
