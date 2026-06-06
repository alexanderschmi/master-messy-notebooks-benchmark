from src.runners.base import BaseRunner
from src.runners.llm_runner import LLMRunner, CoTRunner
from src.runners.openhands_runner import OpenHandsRunner

_REGISTRY: dict[str, type[BaseRunner]] = {
    LLMRunner.RUNNER_NAME: LLMRunner,
    CoTRunner.RUNNER_NAME: CoTRunner,
    OpenHandsRunner.RUNNER_NAME: OpenHandsRunner,
}


def validate_runner(name: str) -> None:
    """Raise ValueError if the runner name is not registered."""
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY)
        raise ValueError(f"Unknown runner '{name}'. Available runners: {available}")


def get_runner(name: str) -> BaseRunner:
    """Return an instantiated runner by its name (e.g. 'simple', 'agentic')."""
    try:
        return _REGISTRY[name]()
    except KeyError:
        available = ", ".join(_REGISTRY)
        raise ValueError(f"Unknown runner '{name}'. Available runners: {available}")


def register_runner(name: str, runner_cls: type[BaseRunner]) -> None:
    """Register a custom runner under the given name."""
    _REGISTRY[name] = runner_cls
