import dataclasses
from typing import List, Optional, Dict, Any
from .core_types import LLMSettings

@dataclasses.dataclass
class RunConfiguration:
    """Holds configuration parameters for an evolutionary run."""
    max_generations: int = 10
    population_size: int = 5
    candidates_per_ask: int = 1

    # --- NEW: Evaluation Timeout Setting ---
    evaluation_timeout_seconds: int = 60

    # LLM & Evolution Strategy Settings
    target_evolve_block_names: Optional[List[str]] = None
    use_diff_format_probability: float = 0.0
    llm_ensemble: Optional[List[LLMSettings]] = None
    llm_selection_strategy: str = "weighted_random"
    self_refine_attempts: int = 0
    allow_full_rewrites: bool = False

    # Island Model Settings
    use_island_model: bool = False
    num_islands: int = 1
    island_generations_per_epoch: int = 5
    migration_num_emigrants: int = 1
    migration_strategy: str = "ring" # 'ring' or 'broadcast_best_to_all'

    # Cascade Evaluation Settings
    cascade_evaluation: bool = False
    cascade_thresholds: List[float] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.self_refine_attempts < 0:
            raise ValueError("self_refine_attempts cannot be negative.")
        if self.evaluation_timeout_seconds <= 0:
            raise ValueError("evaluation_timeout_seconds must be a positive integer.")
