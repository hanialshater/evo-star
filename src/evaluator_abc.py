import abc
from typing import Tuple, Dict, Any

class BaseEvaluator(abc.ABC):
    def __init__(self, problem_name: str):
        self.problem_name = problem_name
        print(f"BaseEvaluator initialized for problem: {self.problem_name}")

    @abc.abstractmethod
    def evaluate(self, program_id: str, full_code_to_evaluate: str,
                 program_generation: int, stage: int = 0 # Added stage for cascading
                ) -> Tuple[Dict[str, float], Dict[str, Any]]: # Scores, EvalDetails
        pass

print("alpha_evolve_framework/evaluator_abc.py written")
