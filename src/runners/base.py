from abc import ABC, abstractmethod
from typing import Optional


class BaseRunner(ABC):
    @abstractmethod
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
        """Execute inference for a single notebook and write output files."""
        ...
