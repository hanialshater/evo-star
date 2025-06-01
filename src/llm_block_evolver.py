import copy
import numpy as np
from typing import Callable, Tuple, List, Optional, Dict, Any

# Relative imports assuming file structure
from .core_types import Program
from .codebase import Codebase
from .llm_manager import LLMManager
from .prompt_engine import PromptEngine
from .database_abc import BaseProgramDatabase
from .map_elites_database import MAPElitesDatabase # For isinstance check
from .optimizer_abc import BaseOptimizer
from .evaluator_abc import BaseEvaluator
from .config import RunConfiguration

class LLMBlockEvolver(BaseOptimizer):
    def __init__(self,
                 initial_codebase: Codebase,
                 llm_manager: LLMManager,
                 program_database: BaseProgramDatabase,
                 prompt_engine: PromptEngine,
                 evaluator: BaseEvaluator,
                 run_config: RunConfiguration,
                 feature_definitions: Optional[List[Dict[str, Any]]] = None,
                 feature_extractor_fn: Optional[Callable[[Dict[str, Any], Dict[str,Any], List[Dict[str, Any]]], Optional[Tuple[float, ...]]]] = None,
                 problem_specific_feature_configs: Optional[Dict[str, Any]] = None,
                 island_id: Optional[int] = None):
        super().__init__(problem_name=evaluator.problem_name, run_config=run_config,
                         evaluator=evaluator, island_id=island_id)
        self.working_codebase = copy.deepcopy(initial_codebase)
        self.llm_manager = llm_manager
        self.program_database = program_database
        self.prompt_engine = prompt_engine
        self.feature_definitions = feature_definitions
        self.feature_extractor_fn = feature_extractor_fn
        self.problem_specific_feature_configs = problem_specific_feature_configs or {}
        self.program_id_counter = 0
        print(f"LLMBlockEvolver (Island {self.island_id}) initialized.")

    def _generate_candidate_temp_id(self) -> str:
        self.program_id_counter += 1
        return f"cand_g{self.current_generation}_i{self.island_id}_evo_id{self.program_id_counter}"

    def get_current_codebase_for_skeleton(self) -> Codebase: return self.working_codebase
    def get_final_codebase(self) -> Codebase: return self.working_codebase

    def initialize_population(self):
        print(f"LLMBlockEvolver (Island {self.island_id}): Initializing population...")
        evolve_block_names = self.working_codebase.get_block_names()
        if not evolve_block_names:
            print(f"LLMBlockEvolver (Island {self.island_id}) Warn: No evolve blocks in codebase.")
            return

        initial_full_code = self.working_codebase.reconstruct_full_code()
        initial_program_id = f"orch_g0_i{self.island_id}_seed"

        print(f"LLMBlockEvolver (Island {self.island_id}): Evaluating initial state (Prog ID {initial_program_id})...")
        scores, eval_details_from_evaluator = self.evaluator.evaluate(
            initial_program_id, initial_full_code, 0, stage=0
        )

        initial_block_name = evolve_block_names[0] if evolve_block_names else None
        initial_block_obj = self.working_codebase.get_block(initial_block_name) if initial_block_name else None

        initial_program = Program(
            id=initial_program_id,
            code_str=initial_block_obj.initial_code if initial_block_obj else "N/A_InitialCodebase",
            block_name=initial_block_name,
            scores=scores,
            generation=0,
            parent_id=None,
            eval_details=eval_details_from_evaluator
        )

        if self.feature_extractor_fn and self.feature_definitions:
            # print(f"LLMBlockEvolver (Island {self.island_id}) DEBUG INITIALIZE - EvalDetails for {initial_program.id}: "
            #       f"shape_ok={initial_program.eval_details.get('shape_ok')}, "
            #       f"is_fully_valid={initial_program.eval_details.get('is_fully_valid_packing')}, "
            #       f"SumR={initial_program.eval_details.get('sum_radii_achieved', 'N/A')}, "
            #       f"Keys: {list(initial_program.eval_details.keys())}")
            features = self.feature_extractor_fn(
                initial_program.eval_details,
                self.problem_specific_feature_configs, # Contains {'n_circles': N}
                self.feature_definitions
            )
            if features:
                initial_program.features = features
                print(f"LLMBlockEvolver (Island {self.island_id}) initial features for {initial_program.id}: {features}")
            else:
                print(f"LLMBlockEvolver (Island {self.island_id}) Warn: No features extracted for initial program {initial_program.id}.")

        self.program_database.add_program(initial_program)
        self.overall_best_solution = copy.deepcopy(initial_program)
        print(f"LLMBlockEvolver (Island {self.island_id}): Initial program {initial_program.id} processed. Score: {scores.get('main_score',-float('inf')):.4f}")

    def ask(self) -> List[Dict[str, Any]]:
        print(f"LLMBlockEvolver (Island {self.island_id}): ask() for local_gen {self.current_generation}")
        suggestions_for_orchestrator: List[Dict[str, Any]] = []
        num_candidates_to_generate = getattr(self.run_config, 'candidates_per_ask', 1)

        for _ in range(num_candidates_to_generate):
            parent_selection_strategy = self.run_config.parent_selection_strategy
            if isinstance(self.program_database, MAPElitesDatabase):
                parent_selection_strategy = getattr(self.run_config, 'map_elites_parent_selection_strategy', "random_elite")

            parent_program = self.program_database.select_parent_program(strategy=parent_selection_strategy)

            if not parent_program:
                parent_program = self.overall_best_solution
                if parent_program:
                    print(f"LLMBlockEvolver (Island {self.island_id}): No parent from DB ({parent_selection_strategy}). Using overall best solution: {parent_program.id}")
                else:
                    print(f"LLMBlockEvolver (Island {self.island_id}): No parent program available in ask(). Cannot generate candidate.")
                    continue

            target_block_name = parent_program.block_name
            if not target_block_name:
                if self.run_config.target_evolve_block_names:
                    block_idx = (self.current_generation - 1) % len(self.run_config.target_evolve_block_names)
                    target_block_name = self.run_config.target_evolve_block_names[block_idx]
                else:
                    all_blocks = self.working_codebase.get_block_names()
                    if all_blocks:
                        block_idx = (self.current_generation - 1) % len(all_blocks)
                        target_block_name = all_blocks[block_idx]
                    else:
                        print(f"LLMBlockEvolver (Island {self.island_id}): Cannot determine target block.")
                        continue

            print(f"LLMBlockEvolver (Island {self.island_id}): Parent {parent_program.id} selected, targeting block '{target_block_name}'.")

            num_ctx = getattr(self.run_config, 'num_map_elites_context', 2)
            ctx_strat = "best_alternate"
            if isinstance(self.program_database, MAPElitesDatabase):
                 ctx_strat = getattr(self.run_config, 'map_elites_context_strategy', "diverse_elites_from_map")

            context_programs = self.program_database.get_context_programs(num_ctx, parent_program.id if parent_program else None, ctx_strat)

            output_format_request = "full_code"
            if self.run_config.use_diff_format_probability > 0 and np.random.rand() < self.run_config.use_diff_format_probability:
                output_format_request = "diff"

            temp_codebase_for_prompting = copy.deepcopy(self.working_codebase)
            current_code_of_block_to_evolve = ""
            target_block_in_temp_cb = temp_codebase_for_prompting.get_block(target_block_name)

            if parent_program.block_name == target_block_name and parent_program.code_str is not None:
                current_code_of_block_to_evolve = parent_program.code_str
                if target_block_in_temp_cb:
                    target_block_in_temp_cb.update_code(parent_program.code_str)
            elif target_block_in_temp_cb:
                current_code_of_block_to_evolve = target_block_in_temp_cb.current_code
            else:
                initial_block_for_prompt = self.working_codebase.get_block(target_block_name)
                if initial_block_for_prompt:
                    current_code_of_block_to_evolve = initial_block_for_prompt.initial_code
                    if temp_codebase_for_prompting.get_block(target_block_name):
                        temp_codebase_for_prompting.update_block_code(target_block_name, current_code_of_block_to_evolve)
                else:
                    print(f"LLMBlockEvolver (Island {self.island_id}) Critical Warning: Target block '{target_block_name}' not found for prompting. Using empty string.")
                    current_code_of_block_to_evolve = ""

            prompt = self.prompt_engine.build_evolution_prompt(
                parent_program, target_block_name, temp_codebase_for_prompting,
                output_format_request, context_programs
            )
            llm_suggestion_content = self.llm_manager.generate_code_modification(prompt)

            if not llm_suggestion_content or not llm_suggestion_content.strip():
                print(f"LLMBlockEvolver (Island {self.island_id}): LLM returned no valid suggestion for block '{target_block_name}'.")
                continue

            candidate_info = {
                'candidate_id': self._generate_candidate_temp_id(),
                'block_name': target_block_name,
                'parent_id': parent_program.id,
                'output_format_used': output_format_request,
                'code_str': llm_suggestion_content, # Corrected key
                'base_code_for_diff': current_code_of_block_to_evolve if output_format_request == "diff" else None
            }
            suggestions_for_orchestrator.append(candidate_info)
            print(f"LLMBlockEvolver (Island {self.island_id}): Suggestion {candidate_info['candidate_id']} for block '{target_block_name}' prepared.")
        return suggestions_for_orchestrator

    def tell(self, evaluated_programs: List[Program]):
        print(f"LLMBlockEvolver (Island {self.island_id}): tell() with {len(evaluated_programs)} programs for local_gen {self.current_generation}.")
        if not evaluated_programs: return
        for prog in evaluated_programs:
            if self.feature_extractor_fn and self.feature_definitions and prog.features is None:
                # print(f"LLMBlockEvolver (Island {self.island_id}) DEBUG TELL - EvalDetails for {prog.id}: "
                #       f"shape_ok={prog.eval_details.get('shape_ok')}, "
                #       f"is_fully_valid={prog.eval_details.get('is_fully_valid_packing')}, "
                #       f"SumR={prog.eval_details.get('sum_radii_achieved', 'N/A')}, "
                #       f"Keys: {list(prog.eval_details.keys())}")
                features = self.feature_extractor_fn(
                    prog.eval_details,
                    self.problem_specific_feature_configs, # Contains {'n_circles': N}
                    self.feature_definitions
                )
                if features:
                    prog.features = features
                    # print(f"LLMBlockEvolver (Island {self.island_id}): Features for {prog.id}: {features}") # Can be verbose
                # else:
                    # print(f"LLMBlockEvolver (Island {self.island_id}) Warn: No features extracted for {prog.id}.")

            self.program_database.add_program(prog)

            current_best_score = -float('inf')
            if self.overall_best_solution and self.overall_best_solution.scores:
                 current_best_score = self.overall_best_solution.scores.get('main_score', -float('inf'))
            new_prog_score = prog.scores.get('main_score', -float('inf'))

            if new_prog_score > current_best_score:
                self.overall_best_solution = copy.deepcopy(prog)
                if self.overall_best_solution.block_name and self.overall_best_solution.code_str:
                    self.working_codebase.update_block_code(
                        self.overall_best_solution.block_name, self.overall_best_solution.code_str
                    )
                    print(f"LLMBlockEvolver (Island {self.island_id}): New internal best {self.overall_best_solution.id}, Score: {new_prog_score:.4f}. Updated working_codebase.")

    def get_best_solution(self) -> Optional[Program]:
        # This method returns the single best program this optimizer is aware of.
        # It considers its internally tracked overall_best_solution and the
        # best discoverable program in its program_database.

        db_best_program = self.program_database.get_best_program()
        optimizer_internal_best = self.overall_best_solution

        contenders: List[Program] = []
        if optimizer_internal_best:
            contenders.append(optimizer_internal_best)
        if db_best_program:
            # Avoid adding the same object twice if they happen to be the same by ID
            # (assuming IDs are globally unique or unique enough for this comparison)
            if not optimizer_internal_best or db_best_program.id != optimizer_internal_best.id:
                contenders.append(db_best_program)

        if not contenders:
            # print(f"LLMBlockEvolver (Island {self.island_id}) get_best_solution: No contenders found.")
            return None

        # Return the one with the highest main_score among these contenders
        best_overall_for_optimizer = max(contenders, key=lambda p: p.scores.get('main_score', -float('inf')), default=None)

        # if best_overall_for_optimizer:
        #     print(f"LLMBlockEvolver (Island {self.island_id}) get_best_solution: Returning {best_overall_for_optimizer.id} "
        #           f"Score: {best_overall_for_optimizer.scores.get('main_score', -float('inf')):.4f}")
        return best_overall_for_optimizer

    # --- Island Model support methods ---
    def get_emigrants(self, num_emigrants: int) -> List[Program]:
        if num_emigrants <= 0: return []
        all_progs_on_island = []
        if isinstance(self.program_database, MAPElitesDatabase):
             # Get unique programs from map elites and recent good programs
             temp_progs_dict = {p.id: p for p in self.program_database.map_elites.values()}
             for p in self.program_database.recent_good_programs:
                 if p.id not in temp_progs_dict:
                     temp_progs_dict[p.id] = p
             all_progs_on_island = list(temp_progs_dict.values())
        else:
            all_progs_on_island = self.program_database.get_all_programs()

        if not all_progs_on_island: return []

        all_progs_on_island.sort(key=lambda p: p.scores.get('main_score', -float('inf')), reverse=True)

        emigrants = [copy.deepcopy(p) for p in all_progs_on_island[:num_emigrants]]
        # print(f"LLMBlockEvolver (Island {self.island_id}): Providing {len(emigrants)} emigrants.")
        return emigrants

    def receive_immigrants(self, immigrants: List[Program]):
        if not immigrants: return
        print(f"LLMBlockEvolver (Island {self.island_id}): Receiving {len(immigrants)} immigrants.")
        for immigrant_prog_original in immigrants:
            # Create a new Program object for the immigrant to avoid ID clashes if it's added to multiple islands,
            # and to reset its generation if island model tracks local generations for immigrants.
            # For now, we'll use its existing ID but ensure it's a copy.
            immigrant_prog = copy.deepcopy(immigrant_prog_original)

            # Optional: Mark as immigrant or adjust generation? For now, add as is.
            # immigrant_prog.generation = self.current_generation # Or orchestrator's global generation

            print(f"  Island {self.island_id} considering immigrant {immigrant_prog.id} (Original Gen: {immigrant_prog.generation}, Score: {immigrant_prog.scores.get('main_score'):.4f}, Feats: {immigrant_prog.features})")
            self.program_database.add_program(immigrant_prog)

            imm_score = immigrant_prog.scores.get('main_score', -float('inf'))
            island_best_score = -float('inf')
            if self.overall_best_solution and self.overall_best_solution.scores:
                island_best_score = self.overall_best_solution.scores.get('main_score', -float('inf'))

            if imm_score > island_best_score:
                print(f"  Immigrant {immigrant_prog.id} is new best on Island {self.island_id}!")
                self.overall_best_solution = copy.deepcopy(immigrant_prog)
                if self.overall_best_solution.block_name and self.overall_best_solution.code_str:
                     self.working_codebase.update_block_code(
                        self.overall_best_solution.block_name,
                        self.overall_best_solution.code_str
                    )
print("alpha_evolve_framework/llm_block_evolver.py (re)written with refined get_best_solution and debug prints.")
