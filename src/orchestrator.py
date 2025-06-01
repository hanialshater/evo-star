import time
import copy
import numpy as np # For migration strategies if needed
from typing import List, Optional, Any, Dict, Union

# Assuming these are correctly placed for relative imports
from .core_types import Program
from .config import RunConfiguration
from .optimizer_abc import BaseOptimizer
from .evaluator_abc import BaseEvaluator
from .codebase import Codebase
from .llm_block_evolver import LLMBlockEvolver # For isinstance check and specific methods

class MainLoopOrchestrator:
    def __init__(self,
                 optimizers: Union[BaseOptimizer, List[BaseOptimizer]], # Can be single or list for islands
                 evaluator: BaseEvaluator,
                 run_config: RunConfiguration,
                 initial_codebase_skeleton: Optional[Codebase] = None): # Used if optimizers are LLMBlockEvolver

        if run_config.use_island_model:
            if not isinstance(optimizers, list) or not optimizers:
                raise ValueError("If use_island_model is True, 'optimizers' must be a non-empty list.")
            if len(optimizers) != run_config.num_islands:
                print(f"Warning: run_config.num_islands ({run_config.num_islands}) "
                      f"does not match len(optimizers) ({len(optimizers)}). Using len(optimizers).")
                run_config.num_islands = len(optimizers) # Adjust config to match reality
            self.optimizers_list: List[BaseOptimizer] = optimizers
            print(f"MainLoopOrchestrator initialized for Island Model with {len(self.optimizers_list)} islands.")
        else:
            if isinstance(optimizers, list):
                if len(optimizers) == 1:
                    self.optimizers_list = optimizers
                else:
                    raise ValueError("If use_island_model is False, 'optimizers' must be a single BaseOptimizer instance, not a list with multiple items.")
            else: # It's a single optimizer instance
                self.optimizers_list = [optimizers]
            print("MainLoopOrchestrator initialized for Single Optimizer Model.")

        self.evaluator = evaluator
        self.run_config = run_config
        self.overall_best_program_orchestrator: Optional[Program] = None
        self.current_global_generation = 0 # Renamed for clarity
        self.program_id_counter_orchestrator = 0

        self.current_codebase_skeleton = None # Used by LLMBlockEvolver instances for reconstruction
        if initial_codebase_skeleton:
             self.current_codebase_skeleton = copy.deepcopy(initial_codebase_skeleton)
        elif isinstance(self.optimizers_list[0], LLMBlockEvolver):
             self.current_codebase_skeleton = copy.deepcopy(self.optimizers_list[0].get_current_codebase_for_skeleton())
             print("MainLoopOrchestrator: Using codebase skeleton from the first LLMBlockEvolver island.")

        if not self.current_codebase_skeleton and any(isinstance(opt, LLMBlockEvolver) for opt in self.optimizers_list):
             print("Warning: Orchestrator with LLMBlockEvolver(s) but no initial_codebase_skeleton. Evaluation might fail.")

        print("MainLoopOrchestrator component setup complete.")


    def _generate_global_program_id(self, temp_candidate_id: str, island_id: int) -> str:
        self.program_id_counter_orchestrator +=1
        clean_temp_id = temp_candidate_id.replace('cand_','').replace('evo_','').replace(f"g{self.optimizers_list[island_id].current_generation}_",'').replace(f"i{island_id}_",'')
        return f"orch_g{self.current_global_generation}_i{island_id}_uid{self.program_id_counter_orchestrator}_{clean_temp_id}"

    def _initialize_all_optimizer_populations(self):
        for i, optimizer_instance in enumerate(self.optimizers_list):
            print(f"\n--- Orchestrator: Initializing population for Optimizer/Island {optimizer_instance.island_id} ---")
            optimizer_instance.initialize_population()
            island_initial_best = optimizer_instance.get_best_solution()
            if island_initial_best:
                current_orch_best_score = -float('inf')
                if self.overall_best_program_orchestrator and self.overall_best_program_orchestrator.scores:
                    current_orch_best_score = self.overall_best_program_orchestrator.scores.get('main_score', -float('inf'))

                island_initial_best_score = island_initial_best.scores.get('main_score', -float('inf')) if island_initial_best.scores else -float('inf')

                if island_initial_best_score > current_orch_best_score:
                    self.overall_best_program_orchestrator = copy.deepcopy(island_initial_best)
                    print(f"Orchestrator: Updated initial global best from Island {optimizer_instance.island_id}: {self.overall_best_program_orchestrator.id}, Score: {island_initial_best_score:.4f}")
        if self.overall_best_program_orchestrator:
            print(f"\nOrchestrator: Overall initial best after all islands initialized: {self.overall_best_program_orchestrator.id}, Score: {self.overall_best_program_orchestrator.scores.get('main_score', -float('inf')):.4f}")


    def _perform_migration(self):
        if not self.run_config.use_island_model or self.run_config.num_islands <= 1 or self.run_config.migration_num_emigrants <= 0:
            return # No migration needed

        print(f"\n--- Orchestrator: Performing Migration (Strategy: {self.run_config.migration_strategy}) ---")
        all_emigrants_from_all_islands: List[List[Program]] = []
        for island_idx, source_optimizer in enumerate(self.optimizers_list):
            emigrants = source_optimizer.get_emigrants(self.run_config.migration_num_emigrants)
            all_emigrants_from_all_islands.append(emigrants)
            # print(f"  Island {source_optimizer.island_id} selected {len(emigrants)} emigrants.")

        num_islands = len(self.optimizers_list)
        for source_island_idx, source_emigrants in enumerate(all_emigrants_from_all_islands):
            if not source_emigrants: continue

            if self.run_config.migration_strategy == "ring":
                target_island_idx = (source_island_idx + 1) % num_islands
                print(f"  Migrating {len(source_emigrants)} from Island {self.optimizers_list[source_island_idx].island_id} to Island {self.optimizers_list[target_island_idx].island_id}")
                self.optimizers_list[target_island_idx].receive_immigrants(source_emigrants)

            elif self.run_config.migration_strategy == "broadcast_best_to_all":
                # Simplification: each island broadcasts its best. A true broadcast_best would find THE best and send it.
                # This implementation sends each island's best to all *other* islands.
                for target_island_idx in range(num_islands):
                    if target_island_idx == source_island_idx: continue
                    print(f"  Broadcasting {len(source_emigrants)} from Island {self.optimizers_list[source_island_idx].island_id} to Island {self.optimizers_list[target_island_idx].island_id}")
                    self.optimizers_list[target_island_idx].receive_immigrants(source_emigrants)
            # Add more migration strategies here ("random_pairs_one_way", "all_to_all")
            else:
                print(f"Warning: Unknown migration strategy '{self.run_config.migration_strategy}'. Skipping migration for Island {source_island_idx}.")
        print("--- Orchestrator: Migration Phase Completed ---")


    def run(self):
        print(f"\n--- Orchestrator Starting Evolution with {len(self.optimizers_list)} Optimizer(s)/Island(s) ---")
        start_time = time.time()

        self._initialize_all_optimizer_populations()

        # The main loop iterates for total generations specified for the orchestrator
        for gen_idx in range(self.run_config.max_generations):
            self.current_global_generation = gen_idx + 1
            print(f"\n--- Orchestrator: Global Generation {self.current_global_generation}/{self.run_config.max_generations} ---")
            self.program_id_counter_orchestrator = 0 # Reset per global generation for cleaner global IDs

            # Epoch: Run local generations on each island
            for island_idx, current_optimizer in enumerate(self.optimizers_list):
                print(f"  --- Orchestrator: Processing Island {current_optimizer.island_id} (Optimizer: {current_optimizer.__class__.__name__}) ---")
                current_optimizer.reset_local_generations() # Reset for this epoch

                for local_gen_idx in range(self.run_config.island_generations_per_epoch if self.run_config.use_island_model else 1):
                    current_optimizer.increment_generation() # This is the optimizer's local generation

                    # The should_terminate here refers to the optimizer's local epoch termination.
                    # Global termination is handled by the orchestrator's max_generations loop.
                    if current_optimizer.should_terminate() and self.run_config.use_island_model:
                        # print(f"  Orchestrator: Island {current_optimizer.island_id} finished its local epoch generations.")
                        break

                    # print(f"    Island {current_optimizer.island_id} - Local Gen {current_optimizer.current_generation}/{self.run_config.island_generations_per_epoch if self.run_config.use_island_model else 1}")

                    candidate_suggestions = current_optimizer.ask()
                    if not candidate_suggestions:
                        print(f"    Island {current_optimizer.island_id}: Optimizer returned no candidates for local gen {current_optimizer.current_generation}.")
                        continue

                    evaluated_programs_for_tell: List[Program] = []
                    for i, suggestion in enumerate(candidate_suggestions):
                        temp_id = suggestion.get('candidate_id', f"sugg{i}")
                        program_id = self._generate_global_program_id(temp_id, current_optimizer.island_id)

                        full_code_to_eval = ""
                        suggested_block_code = suggestion.get('code_str')
                        block_name_to_evolve = suggestion.get('block_name')

                        if isinstance(current_optimizer, LLMBlockEvolver):
                            if not self.current_codebase_skeleton: raise RuntimeError("Orchestrator: Codebase skeleton needed for LLMBlockEvolver.")
                            if not block_name_to_evolve or suggested_block_code is None:
                                print(f"    Orchestrator Warn (Island {current_optimizer.island_id}): Invalid suggestion (no block/code): {suggestion}"); continue

                            eval_cb = copy.deepcopy(self.current_codebase_skeleton)
                            output_format_used = suggestion.get('output_format_used')
                            base_code_for_diff = suggestion.get('base_code_for_diff')
                            block_to_update_in_eval_cb = eval_cb.get_block(block_name_to_evolve)
                            if not block_to_update_in_eval_cb:
                                print(f"    Orchestrator Err (Island {current_optimizer.island_id}): Block '{block_name_to_evolve}' not in skeleton. Suggestion: {suggestion}"); continue

                            if output_format_used == "diff":
                                if base_code_for_diff is None: print(f"    Orchestrator Err (Island {current_optimizer.island_id}): Diff suggested but no base_code. Suggestion: {suggestion}"); continue
                                block_to_update_in_eval_cb.update_code(base_code_for_diff)
                                applied_diff = block_to_update_in_eval_cb.apply_structured_diff(suggested_block_code)
                                if not applied_diff: print(f"    Orchestrator Warn (Island {current_optimizer.island_id}): Failed to apply diff for {temp_id}."); continue
                            else: # full_code
                                block_to_update_in_eval_cb.update_code(suggested_block_code)
                            full_code_to_eval = eval_cb.reconstruct_full_code()
                        else: # Other optimizer types
                            if suggested_block_code is not None: full_code_to_eval = suggested_block_code
                            else: print(f"    Orchestrator Warn (Island {current_optimizer.island_id}): Suggestion without 'code_str': {suggestion}"); continue

                        if not full_code_to_eval: print(f"    Orchestrator Warn (Island {current_optimizer.island_id}): No code to eval for {program_id}."); continue

                        # print(f"    Orchestrator: Evaluating {program_id} (Island {current_optimizer.island_id}, block: '{block_name_to_evolve}')...")
                        scores, eval_details_from_evaluator = self.evaluator.evaluate(
                            program_id, full_code_to_eval, self.current_global_generation
                        )

                        final_block_code_for_program = ""
                        if isinstance(current_optimizer, LLMBlockEvolver):
                            final_block_code_for_program = block_to_update_in_eval_cb.current_code if output_format_used == "diff" else (suggested_block_code or "") # type: ignore
                        else: final_block_code_for_program = suggested_block_code or ""

                        program_for_tell = Program(
                            id=program_id, code_str=final_block_code_for_program, block_name=block_name_to_evolve,
                            scores=scores, generation=self.current_global_generation,
                            parent_id=suggestion.get('parent_id'), eval_details=eval_details_from_evaluator
                        )
                        evaluated_programs_for_tell.append(program_for_tell)

                        cand_score = scores.get('main_score', -float('inf'))
                        orch_best_score = -float('inf')
                        if self.overall_best_program_orchestrator and self.overall_best_program_orchestrator.scores:
                            orch_best_score = self.overall_best_program_orchestrator.scores.get('main_score', -float('inf'))
                        if cand_score > orch_best_score:
                            self.overall_best_program_orchestrator = copy.deepcopy(program_for_tell)
                            print(f"🏆 Orchestrator New Global Best (from Island {current_optimizer.island_id}): {program_for_tell.id}, Score: {cand_score:.4f}")

                    if evaluated_programs_for_tell:
                        current_optimizer.tell(evaluated_programs_for_tell)

            # --- Migration Phase (after all islands completed their local epoch gens) ---
            if self.run_config.use_island_model and self.run_config.num_islands > 1 : #and \
               #(self.current_global_generation % self.run_config.migration_interval_epochs == 0): # Or simply after each full epoch round
                self._perform_migration()
                # After migration, it's good to check if any immigrant became the new global best
                for opt in self.optimizers_list:
                    island_best = opt.get_best_solution()
                    if island_best and island_best.scores:
                        island_best_score = island_best.scores.get('main_score', -float('inf'))
                        orch_best_score = -float('inf')
                        if self.overall_best_program_orchestrator and self.overall_best_program_orchestrator.scores:
                             orch_best_score = self.overall_best_program_orchestrator.scores.get('main_score',-float('inf'))
                        if island_best_score > orch_best_score:
                            self.overall_best_program_orchestrator = copy.deepcopy(island_best)
                            print(f"🏆 Orchestrator New Global Best (post-migration from Island {opt.island_id}): {island_best.id}, Score: {island_best_score:.4f}")


            db_size_info_list = []
            if self.run_config.use_island_model:
                for i, opt in enumerate(self.optimizers_list):
                    db_size_info_list.append(f"Island{opt.island_id} DB: {len(opt.program_database) if hasattr(opt, 'program_database') and opt.program_database else 'N/A'}") # type: ignore
                db_size_info = ", ".join(db_size_info_list)
            elif hasattr(self.optimizers_list[0], 'program_database') and self.optimizers_list[0].program_database: # type: ignore
                 db_size_info = f"Opt DB: {len(self.optimizers_list[0].program_database)}" # type: ignore

            print(f"End Orchestrator Global Gen {self.current_global_generation}. {db_size_info}")
            if self.overall_best_program_orchestrator and self.overall_best_program_orchestrator.scores:
                print(f"Orchestrator Overall Best: {self.overall_best_program_orchestrator.scores.get('main_score', -float('inf')):.4f} (ID: {self.overall_best_program_orchestrator.id})")
            else: print("Orchestrator: overall_best_program_orchestrator is None or has no scores.")

        total_time = time.time() - start_time
        print(f"\n--- Orchestrator Finished in {total_time:.2f}s ---")

        # Final consolidation of the best program from all optimizers
        final_best_candidates = [opt.get_best_solution() for opt in self.optimizers_list]
        if self.overall_best_program_orchestrator: # Add orchestrator's tracked best as a candidate
            final_best_candidates.append(self.overall_best_program_orchestrator)

        true_final_best: Optional[Program] = None
        for prog_cand in final_best_candidates:
            if prog_cand and prog_cand.scores:
                cand_s = prog_cand.scores.get('main_score', -float('inf'))
                if true_final_best is None or (true_final_best.scores and cand_s > true_final_best.scores.get('main_score', -float('inf'))):
                    true_final_best = copy.deepcopy(prog_cand)

        self.overall_best_program_orchestrator = true_final_best

        if self.overall_best_program_orchestrator:
            print(f"Orchestrator Final Best Program ID: {self.overall_best_program_orchestrator.id}, Global Gen: {self.overall_best_program_orchestrator.generation}, Scores: {self.overall_best_program_orchestrator.scores}, Feats: {self.overall_best_program_orchestrator.features}")
            print(f"  Best Program Final Eval Details: {self.overall_best_program_orchestrator.eval_details}")
            print(f"  Represented Block Code:\n```python\n{self.overall_best_program_orchestrator.code_str.strip()}\n```")
            # If the best program came from an LLMBlockEvolver, print its codebase
            # This requires knowing which optimizer produced the best, or assuming they all share/update one main codebase.
            # For now, if the primary optimizer (first one) is LLMBlockEvolver:
            if isinstance(self.optimizers_list[0], LLMBlockEvolver): # type: ignore
                # To get the *actual* final codebase that corresponds to overall_best_program_orchestrator,
                # we need to know which island it came from and get that island's working_codebase.
                # This is complex if the best came from an immigrant.
                # A simpler approach for now is to show the codebase from island 0,
                # or from the island that produced the overall_best_program_orchestrator IF we track that.
                # For now, let's assume the first optimizer's codebase is representative or was updated.
                final_ref_optimizer = self.optimizers_list[0] # Default to first
                for opt in self.optimizers_list: # Find the optimizer that holds the actual best program object
                    if opt.overall_best_solution and self.overall_best_program_orchestrator and opt.overall_best_solution.id == self.overall_best_program_orchestrator.id:
                        final_ref_optimizer = opt
                        break
                if isinstance(final_ref_optimizer, LLMBlockEvolver): # type: ignore
                    final_full_codebase = final_ref_optimizer.get_final_codebase()
                    if final_full_codebase:
                        print(f"\nFinal state of the full codebase (from Island {final_ref_optimizer.island_id}'s perspective):") # type: ignore
                        print("```python")
                        final_code_print = final_full_codebase.reconstruct_full_code().strip()
                        print(f"{final_code_print[:1000]}...\n...{final_code_print[-1000:]}" if len(final_code_print) > 2000 else final_code_print)
                        print("```")
        else: print("Orchestrator: No best program was found.")
        return self.overall_best_program_orchestrator

print("alpha_evolve_framework/orchestrator.py (re)written for Island Model structure.")
