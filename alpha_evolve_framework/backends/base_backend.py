"""
Base backend abstraction for evolution workflows.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from ..core_types import Program, StageOutput, LLMSettings


class EvolutionBackend(ABC):
    """Abstract base class for evolution backends."""

    def __init__(self, api_key: str):
        """Initialize backend with API key."""
        self.api_key = api_key

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available/installed."""
        pass

    @abstractmethod
    def run_stage(
        self,
        stage_config: Dict[str, Any],
        initial_codebase_code: str,
        evaluator_fn,
        evaluator_config: Dict[str, Any],
        initial_population: Optional[List[Program]] = None,
    ) -> StageOutput:
        """Run a single evolution stage."""
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Get the backend name for logging/identification."""
        pass

    def validate_stage_config(self, stage_config: Dict[str, Any]) -> None:
        """Validate stage configuration. Override in subclasses if needed."""
        required_fields = ["name", "max_generations", "llm_ensemble"]
        for field in required_fields:
            if field not in stage_config:
                raise ValueError(f"Missing required field '{field}' in stage config")

        if not isinstance(stage_config["llm_ensemble"], list):
            raise ValueError("llm_ensemble must be a list of LLMSettings")

        if stage_config["max_generations"] <= 0:
            raise ValueError("max_generations must be positive")
