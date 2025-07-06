# -*- coding: utf-8 -*-
import copy
import time
from typing import List, Dict, Any, Optional, Tuple, Callable

from ..core_types import Program, StageOutput
from ..codebase import Codebase
from ..config import RunConfiguration
from ..optimizer_abc import BaseOptimizer
from ..evaluator_abc import BaseEvaluator
from ..llm_judge import LLMJudge
from ..candidate_evaluator import CandidateEvaluator
from ..logging_utils import logger


class MainLoopOrchestrator:
    def __init__(
        self,
        optimizers: List[BaseOptimizer],
        evaluator: BaseEvaluator,
        run_config: RunConfiguration,
        initial_codebase: Codebase,
        llm_judge: Optional[LLMJudge] = None,
        task_description: str = "",
        stage_name: str = "DefaultStage",
        generation_logger: Optional[Callable] = None,
        judge_visual_generator: Optional[Callable] = None,
        judge_feedback_combiner: Optional[Callable] = None,
    ):
        self.optimizers = optimizers
        self.generation_logger = generation_logger
        self.llm_judge = llm_judge
        self.judge_visual_generator = judge_visual_generator
        self.judge_feedback_combiner = judge_feedback_combiner
        self.task_description = task_description
        self.stage_name = stage_name
        self.evaluator = evaluator
        self.run_config = run_config
        self.initial_codebase = initial_codebase
        self.llm_judge = llm_judge
        self.task_description = task_description
        self.global_generation = 0
        self.all_evaluated_programs: List[Program] = []
        self.overall_best_program: Optional[Program] = None
        self.evaluation_step_counter = 0

        # Create shared candidate evaluator
        self.candidate_evaluator = CandidateEvaluator(
            initial_codebase=initial_codebase,
            evaluator=evaluator,
            evaluation_timeout_seconds=run_config.evaluation_timeout_seconds,
            llm_judge=llm_judge,
            task_description=task_description,
            evaluation_step_counter=[self.evaluation_step_counter],
            all_evaluated_programs=self.all_evaluated_programs,
        )

        logger.info(
            f"MainLoopOrchestrator initialized with {len(self.optimizers)} optimizer(s)."
        )
        if self.run_config.use_island_model:
            logger.info(
                f"Island model enabled: {self.run_config.num_islands} islands, {self.run_config.island_generations_per_epoch} generations per epoch."
            )
        else:
            logger.info("Single optimizer mode.")

    def evolve(
        self,
        initial_population: Optional[List[Program]] = None,
        max_runtime_minutes: float = 60.0,
    ) -> StageOutput:
        start_time = time.time()

        # Initialize all optimizers
        for i, optimizer in enumerate(self.optimizers):
            logger.info(f"Initializing optimizer {i} (Island {optimizer.island_id})...")
            optimizer.initialize_population(initial_population)

        try:
            # Main evolution loop
            while self.global_generation < self.run_config.max_generations:
                logger.info(f"\n=== GLOBAL GENERATION {self.global_generation} ===")

                # Check runtime limit
                elapsed_minutes = (time.time() - start_time) / 60.0
                if elapsed_minutes > max_runtime_minutes:
                    logger.warning(
                        f"Runtime limit ({max_runtime_minutes} minutes) exceeded. Stopping evolution."
                    )
                    break

                if self.run_config.use_island_model:
                    self._run_island_epoch()
                else:
                    self._run_single_generation()

                # Update global generation counter
                self.global_generation += 1

                # Log progress
                if self.overall_best_program:
                    logger.info(
                        f"Current best program: {self.overall_best_program.id} with score {self.overall_best_program.scores.get('main_score', -float('inf')):.4f}"
                    )

            logger.info(f"\n=== EVOLUTION COMPLETE ===")
            logger.info(f"Total generations: {self.global_generation}")
            logger.info(f"Total evaluations: {self.evaluation_step_counter}")

            total_time = time.time() - start_time
            logger.info(
                f"--- Orchestrator Stage '{self.stage_name}' Finished in {total_time:.2f}s ---"
            )
            status = "COMPLETED"
            message = f"Stage '{self.stage_name}' completed successfully after {self.global_generation} generations."

        except Exception as e:
            logger.error(
                f"Orchestrator stage '{self.stage_name}' failed with an exception: {e}",
                exc_info=True,
            )
            status = "FAILED"
            message = f"Stage '{self.stage_name}' failed with exception: {e}"

        # Collect final population
        final_population = []
        for opt in self.optimizers:
            if hasattr(opt, "program_database"):
                final_population.extend(opt.program_database.get_all_programs())

        unique_programs = list({prog.id: prog for prog in final_population}.values())
        sorted_programs = sorted(
            unique_programs,
            key=lambda p: p.scores.get("main_score", -float("inf")),
            reverse=True,
        )

        logger.info(
            f"Orchestrator returning {len(sorted_programs)} unique programs to the pipeline."
        )

        return StageOutput(
            stage_name=self.stage_name,
            status=status,
            message=message,
            best_program=self.overall_best_program,
            final_population=sorted_programs,
            artifacts={},
        )

    def _run_single_generation(self):
        """Run a single generation with one optimizer."""
        optimizer = self.optimizers[0]

        # Ask for candidate programs
        candidate_suggestions = optimizer.ask()
        if not candidate_suggestions:
            logger.warning("No candidate suggestions generated. Skipping generation.")
            return

        # Evaluate candidates
        evaluated_programs = self._evaluate_candidates(candidate_suggestions)

        # Tell optimizer about results
        optimizer.tell(evaluated_programs)
        optimizer.increment_generation()

        # Update overall best
        self._update_overall_best()

    def _run_island_epoch(self):
        """Run one epoch of island model evolution."""
        # Run local generations on each island
        for optimizer in self.optimizers:
            optimizer.reset_local_generations()

            while not optimizer.should_terminate():
                # Ask for candidates
                candidate_suggestions = optimizer.ask()
                if not candidate_suggestions:
                    logger.warning(
                        f"Island {optimizer.island_id}: No candidates generated."
                    )
                    optimizer.increment_generation()
                    continue

                # Evaluate candidates
                evaluated_programs = self._evaluate_candidates(candidate_suggestions)

                # Tell optimizer about results
                optimizer.tell(evaluated_programs)
                optimizer.increment_generation()

        # Migration phase
        self._perform_migration()

        # Update overall best
        self._update_overall_best()

    def _evaluate_candidates(
        self, candidate_suggestions: List[Dict[str, Any]]
    ) -> List[Program]:
        """Evaluate a list of candidate programs using shared CandidateEvaluator."""
        evaluated_programs = self.candidate_evaluator.evaluate_candidates(
            candidate_suggestions, self.global_generation
        )

        # Update local counter from shared evaluator
        self.evaluation_step_counter = self.candidate_evaluator.evaluation_step_counter[
            0
        ]

        return evaluated_programs

    def _perform_migration(self):
        """Handle migration between islands."""
        if len(self.optimizers) <= 1:
            return

        logger.info("Performing migration between islands...")

        if self.run_config.migration_strategy == "ring":
            # Ring topology: each island sends to the next
            for i, optimizer in enumerate(self.optimizers):
                next_island_idx = (i + 1) % len(self.optimizers)
                emigrants = optimizer.get_emigrants(
                    self.run_config.migration_num_emigrants
                )
                if emigrants:
                    self.optimizers[next_island_idx].receive_immigrants(emigrants)
                    logger.info(
                        f"Island {optimizer.island_id} -> Island {self.optimizers[next_island_idx].island_id}: {len(emigrants)} emigrants"
                    )

        elif self.run_config.migration_strategy == "broadcast_best_to_all":
            # Find best optimizer and broadcast its best to all others
            best_optimizer = max(
                self.optimizers,
                key=lambda opt: (
                    opt.get_best_solution().scores.get("main_score", -float("inf"))
                    if opt.get_best_solution()
                    else -float("inf")
                ),
            )
            emigrants = best_optimizer.get_emigrants(
                self.run_config.migration_num_emigrants
            )

            if emigrants:
                for optimizer in self.optimizers:
                    if optimizer != best_optimizer:
                        optimizer.receive_immigrants(copy.deepcopy(emigrants))
                        logger.info(
                            f"Island {best_optimizer.island_id} -> Island {optimizer.island_id}: {len(emigrants)} emigrants"
                        )

    def _update_overall_best(self):
        """Update the overall best program across all optimizers."""
        current_best = self.overall_best_program
        current_best_score = (
            current_best.scores.get("main_score", -float("inf"))
            if current_best
            else -float("inf")
        )

        for optimizer in self.optimizers:
            optimizer_best = optimizer.get_best_solution()
            if optimizer_best:
                optimizer_best_score = optimizer_best.scores.get(
                    "main_score", -float("inf")
                )
                if optimizer_best_score > current_best_score:
                    self.overall_best_program = copy.deepcopy(optimizer_best)
                    current_best_score = optimizer_best_score

    def get_final_codebase(self) -> Codebase:
        """Get the final codebase with the best program integrated."""
        if not self.overall_best_program:
            return self.initial_codebase

        final_codebase = copy.deepcopy(self.initial_codebase)
        if self.overall_best_program.block_name:
            final_codebase.update_block_code(
                self.overall_best_program.block_name, self.overall_best_program.code_str
            )
        return final_codebase

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get a summary of the evolution process."""
        return {
            "total_generations": self.global_generation,
            "total_evaluations": self.evaluation_step_counter,
            "best_program_id": (
                self.overall_best_program.id if self.overall_best_program else None
            ),
            "best_score": (
                self.overall_best_program.scores.get("main_score", -float("inf"))
                if self.overall_best_program
                else -float("inf")
            ),
            "num_optimizers": len(self.optimizers),
            "island_model_used": self.run_config.use_island_model,
            "all_evaluated_programs": len(self.all_evaluated_programs),
        }
