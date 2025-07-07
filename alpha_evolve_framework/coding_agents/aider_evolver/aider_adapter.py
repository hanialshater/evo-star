"""
Adapter to make AiderEvolver compatible with ask/tell pattern used by backends.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Callable
import logging

from .aider_evolver import AiderEvolver
from ..llm_block_evolver.codebase import Codebase
from ...core_types import Program, ProgramCandidate
from ...databases.database_abc import BaseProgramDatabase
from ...evaluators.evaluator_abc import BaseEvaluator

logger = logging.getLogger(__name__)


class AiderEvolverAdapter:
    """
    Adapter to make AiderEvolver compatible with ask/tell pattern.

    This adapter bridges the async AiderEvolver with the sync ask/tell pattern
    used by the existing backend infrastructure.
    """

    def __init__(
        self,
        initial_codebase: Codebase,
        evaluator: BaseEvaluator,
        program_database: BaseProgramDatabase,
        model: str = "gemini-1.5-flash",
        working_dir: str = "temp_evolution",
        **kwargs,
    ):
        """Initialize the adapter."""
        self.initial_codebase = initial_codebase
        self.evaluator = evaluator
        self.program_database = program_database
        self.model = model
        self.working_dir = working_dir
        self.kwargs = kwargs

        # Create the underlying AiderEvolver
        self.aider_evolver = AiderEvolver(
            model=model,
            initial_codebase=initial_codebase,
            evaluator=evaluator,
            working_dir=working_dir,
            **kwargs,
        )

        # Track state
        self.generation = 0
        self.candidates_generated = 0

    def initialize_population(self, initial_population: Optional[List[Program]] = None):
        """Initialize the population in the database."""
        if initial_population:
            # Use provided population
            for program in initial_population:
                self.program_database.add_program(program)
        else:
            # Create initial program from codebase
            initial_program = Program(
                id=str(uuid.uuid4()),
                code_str=self.initial_codebase.reconstruct_full_code(),
                block_name="main",
                parent_id=None,
                generation=0,
                scores={},
                evaluator_feedback={},
            )
            self.program_database.add_program(initial_program)

        logger.info(
            f"AiderEvolver initialized with {len(self.program_database.get_all_programs())} programs"
        )

    def ask(self) -> List[Dict[str, Any]]:
        """
        Generate candidate suggestions using AiderEvolver.

        Returns:
            List of candidate suggestions for evaluation
        """
        try:
            # Get best program from database
            programs = self.program_database.get_all_programs()
            if not programs:
                logger.warning("No programs in database for ask()")
                return []

            # Get the best program as base for evolution
            best_program = max(
                programs, key=lambda p: p.scores.get("main_score", -float("inf"))
            )

            # Create candidate for evolution
            candidate = ProgramCandidate(
                id=f"candidate_{self.candidates_generated}",
                code_str=best_program.code_str,
                block_name=best_program.block_name or "main",
                parent_id=best_program.id,
                generation=self.generation,
            )

            # Create task description based on current performance
            current_score = best_program.scores.get("main_score", 0)
            task_description = f"Improve the code to achieve better performance than current score: {current_score}"

            # Run AiderEvolver (sync wrapper for async call)
            try:
                evolved_candidate = asyncio.run(
                    self.aider_evolver.evolve(
                        candidate, task_description, context=self.kwargs
                    )
                )

                # Convert to candidate suggestion format
                suggestion = {
                    "id": evolved_candidate.id,
                    "code_str": evolved_candidate.code_str,
                    "block_name": evolved_candidate.block_name,
                    "parent_id": evolved_candidate.parent_id,
                    "generation": evolved_candidate.generation,
                    "mutation_type": "aider_evolution",
                    "evolved_from": best_program.id,
                }

                self.candidates_generated += 1
                return [suggestion]

            except Exception as e:
                logger.error(f"AiderEvolver evolution failed: {e}")
                # Return original as fallback
                fallback_suggestion = {
                    "id": f"fallback_{self.candidates_generated}",
                    "code_str": best_program.code_str,
                    "block_name": best_program.block_name or "main",
                    "parent_id": best_program.id,
                    "generation": self.generation,
                    "mutation_type": "no_change",
                    "evolved_from": best_program.id,
                }
                self.candidates_generated += 1
                return [fallback_suggestion]

        except Exception as e:
            logger.error(f"Error in ask(): {e}")
            return []

    def tell(self, evaluated_programs: List[Program]):
        """
        Handle evaluated programs by adding them to the database.

        Args:
            evaluated_programs: List of evaluated programs
        """
        for program in evaluated_programs:
            self.program_database.add_program(program)

        logger.info(f"Added {len(evaluated_programs)} evaluated programs to database")

    def increment_generation(self):
        """Increment the generation counter."""
        self.generation += 1
        logger.info(f"AiderEvolver generation incremented to {self.generation}")

    def get_best_program(self) -> Optional[Program]:
        """Get the best program from the database."""
        programs = self.program_database.get_all_programs()
        if not programs:
            return None
        return max(programs, key=lambda p: p.scores.get("main_score", -float("inf")))

    def get_population(self) -> List[Program]:
        """Get current population."""
        return self.program_database.get_all_programs()
