import abc
from typing import Tuple, Dict, Any

class BaseEvaluator(abc.ABC):
    def __init__(self, problem_name: str):
        self.problem_name = problem_name

    @abc.abstractmethod
    def evaluate(self,
                 program_id: str,
                 full_code_to_evaluate: str,
                 program_generation: int,
                 timeout_seconds: int, # <-- Add timeout to the abstract method
                 stage: int = 0
                 ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        pass
