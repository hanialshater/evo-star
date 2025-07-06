# -*- coding: utf-8 -*-
import copy
import numpy as np
import ast
from typing import Callable, Tuple, List, Optional, Dict, Any

from ...core_types import Program
from .codebase import Codebase
from ...llm.llm_manager import LLMManager
from ...llm.prompt_engine import PromptEngine
from ...databases.database_abc import BaseProgramDatabase
from ...optimization.optimizer_abc import BaseOptimizer
from ...evaluators.evaluator_abc import BaseEvaluator
from ...config import RunConfiguration
from ...utils.logging_utils import logger


class LLMBlockEvolver(BaseOptimizer):
    def __init__(
        self,
        initial_codebase: Codebase,
        llm_manager: LLMManager,
        program_database: BaseProgramDatabase,
        prompt_engine: PromptEngine,
        evaluator: BaseEvaluator,
        run_config: RunConfiguration,
        feature_definitions: Optional[List[Dict[str, Any]]] = None,
        feature_extractor_fn: Optional[Callable] = None,
        problem_specific_feature_configs: Optional[Dict[str, Any]] = None,
        island_id: Optional[int] = None,
    ):
        super().__init__(
            problem_name=evaluator.problem_name,
            run_config=run_config,
            evaluator=evaluator,
            island_id=island_id,
        )
        self.working_codebase = copy.deepcopy(initial_codebase)
        self.llm_manager = llm_manager
        self.program_database = program_database
        self.prompt_engine = prompt_engine
        self.feature_definitions = feature_definitions
        self.feature_extractor_fn = feature_extractor_fn
        self.problem_specific_feature_configs = problem_specific_feature_configs or {}
        self.program_id_counter = 0
        self._initialized = False

        # --- Added Debug Prints ---
        print(f"\n--- Debug: LLMBlockEvolver __init__ (Island {self.island_id}) ---")
        print(f"Working Codebase: {self.working_codebase}")
        print(f"Code Template Parts: {self.working_codebase.code_template_parts}")
        print(f"Evolve Blocks: {self.working_codebase.evolve_blocks}")
        print("-------------------------------------------------\n")
        # --- End Debug Prints ---

    def initialize_population(self, initial_population: Optional[List[Program]] = None):
        if self._initialized:
            return
        if initial_population:
            logger.info(
                f"Island {self.island_id} seeding population from previous stage."
            )
            self.tell(initial_population)
        else:
            logger.info(
                f"Island {self.island_id} initializing new population with seed program."
            )
            initial_full_code = self.working_codebase.reconstruct_full_code()

            # --- THIS IS THE CORRECTED CALL ---
            # It now correctly passes the timeout value from the run configuration.
            scores, eval_details = self.evaluator.evaluate(
                "seed",
                initial_full_code,
                0,
                timeout_seconds=self.run_config.evaluation_timeout_seconds,
            )
            # --- END OF CORRECTION ---

            block_name = self.working_codebase.get_block_names()[0]
            initial_program = Program(
                id="seed",
                code_str=self.working_codebase.get_block(block_name).initial_code,
                block_name=block_name,
                scores=scores,
                eval_details=eval_details,
            )
            self.tell([initial_program])
        self._initialized = True

    def ask(self) -> List[Dict[str, Any]]:
        suggestions_for_orchestrator: List[Dict[str, Any]] = []
        for _ in range(self.run_config.candidates_per_ask):
            parent_program = (
                self.program_database.select_parent_program()
                or self.overall_best_solution
            )
            if not parent_program:
                logger.warning(
                    f"Island {self.island_id}: Could not select a parent program to evolve."
                )
                continue

            target_block_name = (
                parent_program.block_name or self.working_codebase.get_block_names()[0]
            )
            context_programs = self.program_database.get_context_programs(
                2, parent_program.id
            )

            llm_suggestion_content = None
            refinement_feedback = None

            temp_codebase_for_prompting = copy.deepcopy(self.working_codebase)
            temp_codebase_for_prompting.get_block(target_block_name).update_code(
                parent_program.code_str
            )

            for attempt in range(self.run_config.self_refine_attempts + 1):
                prompt = self.prompt_engine.build_evolution_prompt(
                    parent_program,
                    target_block_name,
                    temp_codebase_for_prompting,
                    allow_full_rewrites=self.run_config.allow_full_rewrites,
                    refinement_feedback=refinement_feedback,
                    context_programs=context_programs,
                )

                generated_code = self.llm_manager.generate_code_modification(prompt)
                if not generated_code or not generated_code.strip():
                    refinement_feedback = "The generated code was empty. Please provide a valid Python code block."
                    continue

                try:
                    ast.parse(generated_code)
                    logger.info(
                        f"Self-Refinement: Code for parent {parent_program.id} is syntactically valid on attempt {attempt + 1}."
                    )
                    llm_suggestion_content = generated_code
                    break
                except SyntaxError as e:
                    logger.warning(
                        f"Self-Refinement: Attempt {attempt + 1} for parent {parent_program.id} failed syntax check. Error: {e}"
                    )
                    refinement_feedback = (
                        f"The code you provided has a syntax error: {e}. Please fix it."
                    )
                    temp_codebase_for_prompting.get_block(
                        target_block_name
                    ).update_code(generated_code)

            if llm_suggestion_content is None:
                llm_suggestion_content = generated_code
                logger.warning(
                    f"Self-Refinement failed after {self.run_config.self_refine_attempts + 1} attempts. Proceeding with last attempt."
                )

            if not llm_suggestion_content:
                continue

            candidate_info = {
                "candidate_id": f"cand_g{self.current_generation}_i{self.island_id}_{self.program_id_counter}",
                "block_name": target_block_name,
                "parent_id": parent_program.id,
                "code_str": llm_suggestion_content,
                "output_format_used": "full_code",
                "base_code_for_diff": None,
            }
            suggestions_for_orchestrator.append(candidate_info)
        return suggestions_for_orchestrator

    def tell(self, evaluated_programs: List[Program]):
        for prog in evaluated_programs:
            if self.feature_extractor_fn and prog.features is None:
                prog.features = self.feature_extractor_fn(
                    prog.eval_details,
                    self.problem_specific_feature_configs,
                    self.feature_definitions,
                )
            self.program_database.add_program(prog)

            current_best_score = (
                self.overall_best_solution.scores.get("main_score", -float("inf"))
                if self.overall_best_solution
                else -float("inf")
            )
            if prog.scores.get("main_score", -float("inf")) > current_best_score:
                self.overall_best_solution = copy.deepcopy(prog)
                if prog.block_name:
                    self.working_codebase.update_block_code(
                        prog.block_name, prog.code_str
                    )

    def get_best_solution(self):
        return self.program_database.get_best_program()

    def get_emigrants(self, num_emigrants: int):
        all_progs = sorted(
            self.program_database.get_all_programs(),
            key=lambda p: p.scores.get("main_score", -float("inf")),
            reverse=True,
        )
        return [copy.deepcopy(p) for p in all_progs[:num_emigrants]]

    def receive_immigrants(self, immigrants: List[Program]):
        self.tell(immigrants)
