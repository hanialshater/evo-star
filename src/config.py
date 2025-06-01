import dataclasses
from typing import List, Optional, Dict, Any
# Assuming LLMSettings is in core_types.py and core_types.py is in the same directory
# For proper package structure, use relative import if core_types is a sibling module
from .core_types import LLMSettings

@dataclasses.dataclass
class RunConfiguration:
    """
    Holds configuration parameters for an evolutionary run.
    """
    max_generations: int = 10 # Total generations for the orchestrator
    population_size: int = 5
    target_evolve_block_names: Optional[List[str]] = None
    parent_selection_strategy: str = "best"
    use_diff_format_probability: float = 0.0
    log_level: str = "INFO"
    llm_ensemble: Optional[List[LLMSettings]] = None
    llm_selection_strategy: str = "weighted_random"

    map_elites_parent_selection_strategy: str = "random_elite"
    num_map_elites_context: int = 2
    map_elites_context_strategy: str = "diverse_elites_from_map"

    candidates_per_ask: int = 1

    # --- Island Model Settings ---
    use_island_model: bool = False
    num_islands: int = 3
    island_generations_per_epoch: int = 5
    migration_num_emigrants: int = 1
    migration_strategy: str = "ring"
    immigrant_acceptance_strategy: str = "add_to_database"

    def __post_init__(self):
        if self.max_generations <= 0: raise ValueError("max_generations must be positive.")
        if self.population_size <= 0: raise ValueError("population_size must be positive.")
        if not (0.0 <= self.use_diff_format_probability <= 1.0): raise ValueError("use_diff_format_probability error.")
        if self.llm_ensemble and not all(llm.selection_weight >= 0 for llm in self.llm_ensemble):
            raise ValueError("LLM selection_weights must be non-negative.")

        effective_island_gens_per_epoch = "N/A"
        effective_num_islands = "N/A"
        if self.use_island_model:
            if self.num_islands <= 0: # Changed from <=1 to allow single island testing if desired, though typical use is >1
                 print("Warning: num_islands <= 0 with use_island_model=True. Island model effects disabled or will error.")
                 # Consider raising ValueError if num_islands == 1 for actual island dynamics.
            if self.island_generations_per_epoch <=0: raise ValueError("island_generations_per_epoch must be > 0.")
            if self.migration_num_emigrants < 0: raise ValueError("migration_num_emigrants cannot be negative.")
            effective_island_gens_per_epoch = self.island_generations_per_epoch
            effective_num_islands = self.num_islands

        print(f"RunConfiguration: MaxGens={self.max_generations}, Islands Used={self.use_island_model}, NumIslands={effective_num_islands}, IslandEpochGens={effective_island_gens_per_epoch}")

print("alpha_evolve_framework/config.py (re)written with corrected Island Model settings logic in __post_init__.")
