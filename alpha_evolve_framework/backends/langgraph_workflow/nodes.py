"""
LangGraph nodes for evolutionary workflow.
Uses the same core components as LocalPython backend for feature parity.
"""

import copy
import uuid
from typing import Dict, List, Any
import logging

from .state import EvolutionState, update_state_with_evaluation
from ...core_types import Program
from ...codebase import Codebase
from ...llm_manager import LLMManager
from ...prompt_engine import PromptEngine
from ...functional_evaluator import FunctionalEvaluator
from ...simple_program_database import SimpleProgramDatabase
from ...map_elites_database import MAPElitesDatabase
from ...llm_block_evolver import LLMBlockEvolver
from ...optimizer_abc import BaseOptimizer
from ...candidate_evaluator import CandidateEvaluator

logger = logging.getLogger(__name__)


class EvolutionNodes:
    """Collection of LangGraph nodes for evolutionary workflow."""

    def __init__(self, evaluator_fn=None):
        """Initialize with evaluator function."""
        self.evaluator_fn = evaluator_fn
        self.optimizers = []  # Will hold actual optimizers like LocalPython backend
        self.candidate_evaluator = None  # Will hold shared CandidateEvaluator

    def initialize_population(self, state: EvolutionState) -> EvolutionState:
        """Initialize optimizers and population using same components as LocalPython backend."""
        logger.info(f"Initializing population for stage: {state['stage_name']}")

        try:
            # Create the same components as LocalPython backend
            stage_config = state["stage_config"]

            # Create RunConfiguration from stage config
            from ...config import RunConfiguration
            import dataclasses

            run_config_fields = {f.name for f in dataclasses.fields(RunConfiguration)}
            params_for_run_config = {
                key: stage_config[key]
                for key in stage_config
                if key in run_config_fields
            }
            run_config = RunConfiguration(**params_for_run_config)

            # Initialize LLM manager
            llm_manager = LLMManager(
                default_api_key=state["api_key"],
                llm_settings_list=state["llm_settings"],
                min_inter_request_delay_sec=stage_config.get(
                    "min_inter_request_delay_sec", 1.1
                ),
            )

            # Create evaluator
            evaluator_config = stage_config.get("evaluator_config", {})
            evaluator = FunctionalEvaluator(
                state["stage_name"], self.evaluator_fn, evaluator_config
            )

            # Create database (same logic as LocalPython backend)
            if stage_config.get("feature_definitions"):
                database = MAPElitesDatabase(
                    feature_definitions=stage_config["feature_definitions"],
                    context_program_capacity=stage_config.get("population_size", 10),
                )
            else:
                database = SimpleProgramDatabase(
                    population_size=stage_config.get("population_size", 10)
                )

            # Create codebase
            codebase = Codebase(state["initial_codebase_code"])

            # Create prompt engine
            prompt_engine = PromptEngine(
                task_description=state["task_description"],
                problem_specific_instructions=stage_config.get(
                    "problem_specific_instructions"
                ),
            )

            # Create LLMBlockEvolver optimizer (same as LocalPython backend)
            optimizer = LLMBlockEvolver(
                initial_codebase=codebase,
                llm_manager=llm_manager,
                program_database=database,
                prompt_engine=prompt_engine,
                evaluator=evaluator,
                run_config=run_config,
                feature_definitions=stage_config.get("feature_definitions"),
                feature_extractor_fn=stage_config.get("feature_extractor_fn"),
                problem_specific_feature_configs=evaluator_config,
                island_id=0,
            )

            # Store optimizer for use in other nodes
            self.optimizers = [optimizer]

            # Create shared candidate evaluator (same as MainLoopOrchestrator)
            self.candidate_evaluator = CandidateEvaluator(
                initial_codebase=codebase,
                evaluator=evaluator,
                evaluation_timeout_seconds=run_config.evaluation_timeout_seconds,
                llm_judge=None,  # Can be added later if needed
                task_description=state["task_description"],
                evaluation_step_counter=[0],
                all_evaluated_programs=[],
            )

            # Initialize optimizer (same as LocalPython backend)
            optimizer.initialize_population(
                initial_population=state.get("current_population"),
            )

            # Get initial population from optimizer
            initial_population = optimizer.program_database.get_all_programs()
            state["current_population"] = initial_population

            logger.info(
                f"Created initial population with {len(initial_population)} program(s)"
            )

        except Exception as e:
            logger.error(f"Error initializing population: {e}")
            state["error_message"] = f"Failed to initialize population: {e}"
            state["should_terminate"] = True

        return state

    def generate_candidates(self, state: EvolutionState) -> EvolutionState:
        """Generate candidate programs using optimizer's ask/tell pattern (same as LocalPython backend)."""
        logger.info(f"Generating candidates for generation {state['generation']}")

        try:
            if not self.optimizers:
                raise RuntimeError("No optimizers initialized")

            optimizer = self.optimizers[0]

            # Use optimizer's ask method to get candidate suggestions
            candidate_suggestions = optimizer.ask()

            if not candidate_suggestions:
                logger.warning(
                    "No candidate suggestions generated. Skipping generation."
                )
                state["generation"] += 1
                return state

            # Evaluate candidates (same pattern as MainLoopOrchestrator)
            evaluated_programs = self._evaluate_candidates(candidate_suggestions, state)

            # Tell optimizer about the results
            optimizer.tell(evaluated_programs)
            optimizer.increment_generation()

            # Get updated population from optimizer
            state["current_population"] = optimizer.program_database.get_all_programs()

            # Update best program
            if state["current_population"]:
                best_program = max(
                    state["current_population"],
                    key=lambda p: p.scores.get("main_score", -float("inf")),
                )
                state["best_program"] = best_program

                # Update convergence metrics
                scores = [
                    p.scores.get("main_score", -float("inf"))
                    for p in state["current_population"]
                ]
                import statistics

                state["convergence_metrics"] = {
                    "max_score": max(scores),
                    "mean_score": statistics.mean(scores),
                    "score_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                }

            # Store evaluated programs for reference
            state["evaluated_programs"] = evaluated_programs

            logger.info(f"Generation {state['generation']} completed")

        except Exception as e:
            logger.error(f"Error generating candidates: {e}")
            state["error_message"] = f"Failed to generate candidates: {e}"
            state["should_terminate"] = True

        return state

    def _evaluate_candidates(
        self, candidate_suggestions: List[Dict[str, Any]], state: EvolutionState
    ) -> List[Program]:
        """Evaluate a list of candidate programs using shared CandidateEvaluator."""
        if not self.candidate_evaluator:
            raise RuntimeError("CandidateEvaluator not initialized")

        # Use shared evaluator (same as MainLoopOrchestrator)
        evaluated_programs = self.candidate_evaluator.evaluate_candidates(
            candidate_suggestions, state["generation"]
        )

        return evaluated_programs

    def check_termination(self, state: EvolutionState) -> EvolutionState:
        """Check if evolution should terminate."""
        logger.info(
            f"Checking termination conditions (gen {state['generation']}/{state['max_generations']})"
        )

        try:
            # Check generation limit
            if state["generation"] >= state["max_generations"]:
                logger.info("Reached maximum generations")
                state["should_terminate"] = True
                return state

            # Check for errors
            if state["error_message"]:
                logger.warning(f"Terminating due to error: {state['error_message']}")
                state["should_terminate"] = True
                return state

            # Simple convergence check (optional for POC)
            convergence_threshold = state["stage_config"].get("convergence_threshold")
            if convergence_threshold and state["convergence_metrics"]:
                std_score = state["convergence_metrics"].get("score_std", float("inf"))
                if std_score < convergence_threshold:
                    logger.info(
                        f"Converged (std: {std_score} < {convergence_threshold})"
                    )
                    state["should_terminate"] = True
                    return state

            # Continue evolution
            state["generation"] += 1
            logger.info(f"Continuing to generation {state['generation']}")

        except Exception as e:
            logger.error(f"Error checking termination: {e}")
            state["error_message"] = f"Failed to check termination: {e}"
            state["should_terminate"] = True

        return state


# Note: Code extraction functions removed as they're now handled by the shared CandidateEvaluator
