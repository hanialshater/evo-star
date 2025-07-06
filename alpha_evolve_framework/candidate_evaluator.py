"""
Shared candidate evaluation logic for both LocalPython and LangGraph backends.
Ensures functional parity between different execution backends.
"""

import copy
import logging
from typing import Dict, List, Any, Optional, Callable

from .core_types import Program
from .codebase import Codebase
from .evaluator_abc import BaseEvaluator
from .llm_judge import LLMJudge

logger = logging.getLogger(__name__)


class CandidateEvaluator:
    """
    Shared candidate evaluation logic that ensures functional parity
    between different execution backends.
    """

    def __init__(
        self,
        initial_codebase: Codebase,
        evaluator: BaseEvaluator,
        evaluation_timeout_seconds: int = 60,
        llm_judge: Optional[LLMJudge] = None,
        task_description: str = "",
        evaluation_step_counter: Optional[List[int]] = None,
        all_evaluated_programs: Optional[List[Program]] = None,
    ):
        """
        Initialize the candidate evaluator.

        Args:
            initial_codebase: The initial codebase to use for evaluation
            evaluator: The evaluator to use for scoring programs
            evaluation_timeout_seconds: Timeout for evaluation in seconds
            llm_judge: Optional LLM judge for program feedback
            task_description: Description of the task for LLM judge
            evaluation_step_counter: Optional counter for tracking evaluations
            all_evaluated_programs: Optional list to store all evaluated programs
        """
        self.initial_codebase = initial_codebase
        self.evaluator = evaluator
        self.evaluation_timeout_seconds = evaluation_timeout_seconds
        self.llm_judge = llm_judge
        self.task_description = task_description
        self.evaluation_step_counter = evaluation_step_counter or [0]
        self.all_evaluated_programs = all_evaluated_programs or []

    def evaluate_candidates(
        self,
        candidate_suggestions: List[Dict[str, Any]],
        generation: int,
    ) -> List[Program]:
        """
        Evaluate a list of candidate programs.

        This method implements the exact same logic as the original working MainLoopOrchestrator
        to ensure functional parity between backends.

        Args:
            candidate_suggestions: List of candidate program dictionaries
            generation: Current generation number

        Returns:
            List of evaluated Program objects
        """
        evaluated_programs = []

        for candidate_info in candidate_suggestions:
            candidate_id = candidate_info["candidate_id"]
            block_name = candidate_info["block_name"]
            parent_id = candidate_info.get("parent_id", None)
            code_str = candidate_info["code_str"]  # Use raw code_str directly

            # Create temporary codebase for evaluation
            temp_codebase = copy.deepcopy(self.initial_codebase)
            temp_codebase.update_block_code(block_name, code_str)
            full_code = temp_codebase.reconstruct_full_code()

            # Evaluate the program
            logger.info(f"Evaluating candidate {candidate_id}...")
            scores, eval_details = self.evaluator.evaluate(
                candidate_id,
                full_code,
                generation,
                timeout_seconds=self.evaluation_timeout_seconds,
            )

            # Create Program object
            program = Program(
                id=candidate_id,
                code_str=code_str,  # Store the raw LLM output directly
                block_name=block_name,
                parent_id=parent_id,
                scores=scores,
                eval_details=eval_details,
                generation=generation,
            )

            # Judge the program if LLM judge is available
            if self.llm_judge:
                judge_feedback = self.llm_judge.judge_program(
                    program, self.task_description
                )
                if judge_feedback:
                    program.judge_feedback = judge_feedback

            evaluated_programs.append(program)
            self.all_evaluated_programs.append(program)
            self.evaluation_step_counter[0] += 1

            logger.info(
                f"Evaluated {candidate_id}: Score = {scores.get('main_score', -float('inf')):.4f}"
            )

        return evaluated_programs
