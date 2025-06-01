import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from typing import List, Dict, Any, Optional, Callable, Tuple # For type hints

# Ensure the framework directory is in the Python path
# This is often needed if running a script directly that uses a sibling package.
# For Colab, after %%writefile, sometimes a restart or direct path manipulation is needed
# if the files are not in the root or Colab doesn't pick them up immediately.
# If 'alpha_evolve_framework' is in the same root as 'examples', this might not be strictly necessary
# when Colab's working directory is that root.
# However, adding it defensively can help.
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Not reliable in all notebook environments
# PARENT_DIR = os.path.join(SCRIPT_DIR, '..', '..') # Go up two levels to project root
# if PARENT_DIR not in sys.path:
#    sys.path.insert(0, PARENT_DIR)
if '' not in sys.path: # In Colab, '' (current dir) is usually in sys.path
    sys.path.insert(0, '')


# Framework imports
from alpha_evolve_framework.core_types import Program, LLMSettings
from alpha_evolve_framework.codebase import Codebase
from alpha_evolve_framework.llm_manager import LLMManager
from alpha_evolve_framework.prompt_engine import PromptEngine
from alpha_evolve_framework.map_elites_database import MAPElitesDatabase # Using MAP-Elites per island
from alpha_evolve_framework.config import RunConfiguration
from alpha_evolve_framework.optimizer_abc import BaseOptimizer # For type hinting list of optimizers
from alpha_evolve_framework.llm_block_evolver import LLMBlockEvolver
from alpha_evolve_framework.orchestrator import MainLoopOrchestrator

# Example-specific imports
from examples.circle_packing.packer_constants import (
    N_CIRCLES, TARGET_SUM_RADII_REF, FLOAT_PRECISION_TOLERANCE, PACKER_FEATURE_DEFINITIONS
)
# packer_helpers are used by packer_evaluator, which imports them.
# We might need them for visualization's re-evaluation, so direct import can be useful.
from examples.circle_packing.packer_helpers import helper_validate_packing, helper_compute_max_radii
from examples.circle_packing.packer_features import extract_circle_packing_features
from examples.circle_packing.packer_evaluator import CirclePackingEvaluator
from examples.circle_packing.packer_initial_code import get_packer_initial_code_n26_str


def run_circle_packing_evolution(gemini_api_key: str):
    print("\n--- Setting up N=26 MAP-Elites Circle Packing Example with Island Model via Orchestrator ---")

    initial_packer_code_str = get_packer_initial_code_n26_str()
    # This initial codebase structure will be copied for each island's optimizer
    base_initial_codebase = Codebase(initial_full_code=initial_packer_code_str)

    # Shared components among islands (or could be island-specific if needed)
    llm_ensemble_config = [
        LLMSettings(model_name="gemini-1.5-flash-latest", selection_weight=0.7, generation_params={"temperature": 0.7}),
        LLMSettings(model_name="gemini-1.5-flash-latest", selection_weight=0.3, generation_params={"temperature": 0.9}) # Simulate another model with different settings
    ]
    shared_llm_manager = LLMManager(
        default_api_key=gemini_api_key,
        llm_settings_list=llm_ensemble_config,
        selection_strategy="weighted_random"
    )

    task_desc = f"Optimize center placement for {N_CIRCLES} circles in a unit square to maximize sum of radii. Target sum ~{TARGET_SUM_RADII_REF:.3f} (SOTA ~2.635)."
    problem_instr = "Focus on deterministic geometric constructions. Insights: hexagonal local structures, edge effects, layered arrangements. Initial code places 25 circles in rings; optimize all 26."
    shared_prompt_engine = PromptEngine(task_description=task_desc, problem_specific_instructions=problem_instr)

    shared_evaluator = CirclePackingEvaluator(
        n_circles=N_CIRCLES,
        target_sum_radii_ref=TARGET_SUM_RADII_REF,
        float_precision_tolerance=FLOAT_PRECISION_TOLERANCE
    )

    # Configuration for the run
    # Note: N_CIRCLES and PACKER_FEATURE_DEFINITIONS are imported from packer_constants
    run_config_islands = RunConfiguration(
        max_generations=10,                 # Total orchestrator generations/epochs
        use_island_model=True,              # Enable island model
        num_islands=2,                      # Number of islands
        island_generations_per_epoch=5,     # Local generations on each island per orchestrator epoch
        migration_num_emigrants=1,          # How many best individuals migrate
        migration_strategy="ring",          # e.g., "ring", "random_pairs_one_way", "broadcast_best_to_all"

        population_size=7, # Used for MAPElitesDatabase context_program_capacity & SimpleDB
        target_evolve_block_names=["packing_core_logic"],
        use_diff_format_probability=0.4,
        llm_ensemble=llm_ensemble_config,
        llm_selection_strategy="weighted_random",
        map_elites_parent_selection_strategy="random_elite", # Strategy for parent selection within each island's MAP
        num_map_elites_context=2,
        candidates_per_ask=1
    )

    # Create optimizer instances for each island
    island_optimizers: List[BaseOptimizer] = []
    for i in range(run_config_islands.num_islands):
        print(f"\n--- Initializing Optimizer for Island {i} ---")
        island_map_db = MAPElitesDatabase(
            feature_definitions=PACKER_FEATURE_DEFINITIONS,
            context_program_capacity=run_config_islands.population_size # Or a specific config for this
        )
        island_optimizer = LLMBlockEvolver(
            initial_codebase=copy.deepcopy(base_initial_codebase), # Each island starts with a fresh copy
            llm_manager=shared_llm_manager,         # Shared
            program_database=island_map_db,         # Island-specific database
            prompt_engine=shared_prompt_engine,     # Shared
            evaluator=shared_evaluator,             # Shared (used for its initialize_population)
            run_config=run_config_islands,          # Shared config
            feature_definitions=PACKER_FEATURE_DEFINITIONS, # Shared
            feature_extractor_fn=extract_circle_packing_features, # Shared
            problem_specific_feature_configs={'n_circles': N_CIRCLES}, # Shared
            island_id=i # Assign unique ID to island optimizer
        )
        island_optimizers.append(island_optimizer)

    print("\n--- Initializing MainLoopOrchestrator for Island Model ---")
    orchestrator = MainLoopOrchestrator(
        optimizers=island_optimizers,         # Pass the list of island optimizers
        evaluator=shared_evaluator,           # Orchestrator uses this for actual evaluation
        run_config=run_config_islands,
        initial_codebase_skeleton=base_initial_codebase # Skeleton for reconstructing code
    )

    # --- Run Evolution ---
    start_run_time = time.time()
    best_program_found = orchestrator.run()
    end_run_time = time.time()
    print(f"\nN={N_CIRCLES} Island Model evolution run via Orchestrator took {end_run_time - start_run_time:.2f} seconds.")

    # --- Visualization ---
    if best_program_found and best_program_found.eval_details and \
       best_program_found.eval_details.get('is_fully_valid_packing', False):
        print(f"\nVisualizing best program from Island run: ID {best_program_found.id}")
        try:
            vis_codebase = copy.deepcopy(base_initial_codebase)
            if best_program_found.block_name and best_program_found.code_str:
                 vis_codebase.update_block_code(best_program_found.block_name, best_program_found.code_str)
            full_code_for_vis = vis_codebase.reconstruct_full_code()

            vis_exec_globals = {
                'np': np,
                'helper_validate_packing': helper_validate_packing,
                'helper_compute_max_radii': helper_compute_max_radii,
                'FLOAT_PRECISION_TOLERANCE_FOR_HELPERS': FLOAT_PRECISION_TOLERANCE
            }
            exec(full_code_for_vis, vis_exec_globals)
            harness_func_vis = vis_exec_globals.get('run_evaluation_harness')

            if harness_func_vis and callable(harness_func_vis):
                # Need to ensure N_CIRCLES is available for the harness
                centers_vis, radii_vis, sum_r_vis, error_vis = harness_func_vis(N_CIRCLES)
                if error_vis:
                    print(f"Error during visualization re-evaluation: {error_vis}")
                elif isinstance(centers_vis, np.ndarray) and isinstance(radii_vis, np.ndarray) and \
                     centers_vis.shape == (N_CIRCLES, 2) and radii_vis.shape == (N_CIRCLES,):

                    fig, ax = plt.subplots(1, figsize=(8, 8))
                    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
                    ax.set_aspect('equal', adjustable='box')
                    title_str = (f"Best Island Model Packing (N={N_CIRCLES})\nID: {best_program_found.id}, "
                                 f"Score: {best_program_found.scores.get('main_score',0):.3f}, SumR: {sum_r_vis:.4f}\n"
                                 f"Features: {best_program_found.features}")
                    ax.set_title(title_str, fontsize=10)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    for i in range(N_CIRCLES):
                        if radii_vis[i] > FLOAT_PRECISION_TOLERANCE / 10:
                            circle = plt.Circle((centers_vis[i,0], centers_vis[i,1]), radii_vis[i],
                                                fill=True, alpha=0.6, edgecolor='black', lw=0.3)
                            ax.add_artist(circle)
                    plt.show()
                else:
                    print(f"Vis err: Could not retrieve valid centers/radii. Centers shape: {getattr(centers_vis,'shape','N/A')}, Radii shape: {getattr(radii_vis,'shape','N/A')}")
            else:
                print("Vis err: `run_evaluation_harness` not found in best program's executed code.")
        except Exception as e_vis:
            print(f"Error during N={N_CIRCLES} Island Model visualization: {e_vis}")
            import traceback
            traceback.print_exc()
    elif best_program_found:
        print(f"\nBest N={N_CIRCLES} Island Model program (ID: {best_program_found.id}) not fully valid or details missing. No vis. Eval Details: {best_program_found.eval_details}")
    else:
        print(f"\nNo best program found for N={N_CIRCLES} (Island Model) after evolution.")
    print(f"\n--- N={N_CIRCLES} Circle Packing Run (Island Model via Orchestrator) Finished ---")

if __name__ == "__main__":
    # This block is for when you save this as a .py file and run from terminal.
    # In Colab, you would call run_circle_packing_evolution(GEMINI_API_KEY) in a cell.
    print("Script 'run_packer_map_elites.py' executed directly.")
    print("To run the evolution, ensure GEMINI_API_KEY is set globally (e.g., from Colab secrets or env var),")
    print("and then call: run_circle_packing_evolution(your_api_key_variable)")

    # Example for trying to run if key is an environment variable (won't work directly in Colab like this usually)
    # key = os.environ.get("GEMINI_API_KEY")
    # if key:
    #    print("Found GEMINI_API_KEY in environment, attempting run...")
    #    run_circle_packing_evolution(gemini_api_key=key)
    # else:
    #    print("GEMINI_API_KEY environment variable not found.")

print("examples/circle_packing/run_packer_map_elites.py written for Island Model test.")
