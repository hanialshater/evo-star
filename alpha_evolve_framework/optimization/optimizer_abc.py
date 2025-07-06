import abc
from typing import List, Optional, Any, Dict
from ..core_types import Program  # Relative import
from ..config import RunConfiguration  # Relative import
from ..evaluators import BaseEvaluator  # Relative import


class BaseOptimizer(abc.ABC):
    def __init__(
        self,
        problem_name: str,
        run_config: RunConfiguration,
        evaluator: BaseEvaluator,
        island_id: Optional[int] = None,
        **kwargs,
    ):
        self.problem_name = problem_name
        self.run_config = run_config
        self.evaluator = evaluator
        self.current_generation = 0
        self.overall_best_solution: Optional[Program] = None
        self.island_id = island_id if island_id is not None else 0
        print(
            f"BaseOptimizer (Island {self.island_id}) initialized for problem: {self.problem_name}"
        )

    @abc.abstractmethod
    def initialize_population(self):
        pass

    @abc.abstractmethod
    def ask(self) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def tell(self, evaluated_programs: List[Program]):
        pass

    @abc.abstractmethod
    def get_best_solution(self) -> Optional[Program]:
        pass

    @abc.abstractmethod
    def get_emigrants(self, num_emigrants: int) -> List[Program]:
        pass

    @abc.abstractmethod
    def receive_immigrants(self, immigrants: List[Program]):
        pass

    def should_terminate(self) -> bool:
        # This method determines if THIS optimizer instance (island) should stop its local epoch.
        # The MainLoopOrchestrator handles overall termination (max_generations for the entire run).
        if self.run_config.use_island_model:
            # Terminate local epoch if island_generations_per_epoch is reached
            return (
                self.current_generation >= self.run_config.island_generations_per_epoch
            )
        else:
            # If not using island model (i.e., this optimizer is the only one),
            # it doesn't terminate itself early; orchestrator's global max_generations controls.
            return False

    def increment_generation(self):
        self.current_generation += 1

    def reset_local_generations(self):
        self.current_generation = 0
