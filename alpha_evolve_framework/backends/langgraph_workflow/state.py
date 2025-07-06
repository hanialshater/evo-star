"""
State management for LangGraph evolutionary workflows.
"""

from typing import Dict, List, Optional, Any
from ...core_types import Program, LLMSettings


class EvolutionState(dict):
    """State schema for LangGraph evolution workflow."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize default values if not provided
        self.setdefault("generation", 0)
        self.setdefault("max_generations", 10)
        self.setdefault("stage_name", "")
        self.setdefault("current_population", [])
        self.setdefault("candidate_programs", [])
        self.setdefault("evaluated_programs", [])
        self.setdefault("best_program", None)
        self.setdefault("island_populations", {})
        self.setdefault("num_islands", 1)
        self.setdefault("current_island", 0)
        self.setdefault("llm_settings", [])
        self.setdefault("stage_config", {})
        self.setdefault("run_config", {})
        self.setdefault("api_key", "")
        self.setdefault("initial_codebase_code", "")
        self.setdefault("task_description", "")
        self.setdefault("evaluation_batch", [])
        self.setdefault("evaluation_results", [])
        self.setdefault("migration_counter", 0)
        self.setdefault("convergence_metrics", {})
        self.setdefault("should_terminate", False)
        self.setdefault("error_message", None)
        self.setdefault("artifacts", {})
        self.setdefault("generation_logs", [])


def create_initial_state(
    stage_name: str,
    max_generations: int,
    llm_settings: List[LLMSettings],
    stage_config: Dict[str, Any],
    run_config: Dict[str, Any],
    api_key: str,
    initial_codebase_code: str,
    task_description: str,
    initial_population: Optional[List[Program]] = None,
) -> EvolutionState:
    """Create initial state for evolution workflow."""

    num_islands = run_config.get("num_islands", 1)

    return EvolutionState(
        # Core evolution state
        generation=0,
        max_generations=max_generations,
        stage_name=stage_name,
        # Population management
        current_population=initial_population or [],
        candidate_programs=[],
        evaluated_programs=[],
        best_program=None,
        # Multi-island support
        island_populations={i: [] for i in range(num_islands)},
        num_islands=num_islands,
        current_island=0,
        # Configuration
        llm_settings=llm_settings,
        stage_config=stage_config,
        run_config=run_config,
        # Execution context
        api_key=api_key,
        initial_codebase_code=initial_codebase_code,
        task_description=task_description,
        # Evaluation results
        evaluation_batch=[],
        evaluation_results=[],
        # Migration and convergence
        migration_counter=0,
        convergence_metrics={},
        # Workflow control
        should_terminate=False,
        error_message=None,
        # Artifacts and logging
        artifacts={},
        generation_logs=[],
    )


def update_state_with_evaluation(
    state: EvolutionState, evaluated_programs: List[Program]
) -> EvolutionState:
    """Update state with evaluation results."""

    # Update evaluated programs
    state["evaluated_programs"] = evaluated_programs

    # Update best program
    current_best = state["best_program"]
    current_best_score = (
        current_best.scores.get("main_score", -float("inf"))
        if current_best
        else -float("inf")
    )

    for program in evaluated_programs:
        program_score = program.scores.get("main_score", -float("inf"))
        if program_score > current_best_score:
            state["best_program"] = program
            current_best_score = program_score

    # Update convergence metrics
    if evaluated_programs:
        scores = [p.scores.get("main_score", -float("inf")) for p in evaluated_programs]
        state["convergence_metrics"] = {
            "mean_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "score_std": _calculate_std(scores) if len(scores) > 1 else 0.0,
        }

    return state


def _calculate_std(values: List[float]) -> float:
    """Calculate standard deviation of values."""
    if len(values) <= 1:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance**0.5
