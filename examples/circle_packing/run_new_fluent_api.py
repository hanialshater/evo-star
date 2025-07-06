# -*- coding: utf-8 -*-
"""
This script demonstrates the use of the EvoAgent framework with the fluent API,
including the LLM Judge, for the circle packing problem.
"""
import sys
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import traceback
import json
import re

# Ensure the project root is in the Python path
if '' not in sys.path:
    sys.path.insert(0, '')

from alpha_evolve_framework.fluent_api import EvoAgent
from alpha_evolve_framework.core_types import LLMSettings, Program
from alpha_evolve_framework.codebase import Codebase
from examples.circle_packing.problem_functions import (
    get_packer_initial_code,
    circle_packer_evaluator,
    extract_packer_features,
    get_packer_feature_definitions
)

# --- Visualization & Judge Input Functions ---

def visualize_best_packing(generation: int, best_program: Program, full_codebase: Codebase):
    """
    A logger function that visualizes the best circle packing solution at the end of a generation.
    This function is for display purposes during the run.
    """
    if not best_program or not best_program.eval_details.get('is_valid', False):
        print(f"  Logger (Gen {generation}): Skipping visualization for invalid or missing program.")
        return

    n_circles = best_program.eval_details['centers_for_features'].shape[0]
    centers = best_program.eval_details['centers_for_features']
    radii = best_program.eval_details['radii_for_features']

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal', adjustable='box')

    title_str = (f"Best of Generation {generation}\n"
                 f"Score: {best_program.scores.get('main_score', 0):.4f}, "
                 f"SumR: {np.sum(radii):.4f}")
    ax.set_title(title_str, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    for i in range(n_circles):
        circle = plt.Circle((centers[i, 0], centers[i, 1]), radii[i], fill=True, alpha=0.7)
        ax.add_artist(circle)

    plt.show()

def packing_image_generator(eval_details: dict) -> Image.Image:
    """
    Generates a PIL Image from the packing data for the LLM Judge, without displaying it.
    """
    if not eval_details.get('is_valid', False) or 'centers_for_features' not in eval_details:
        return None

    centers = eval_details['centers_for_features']
    radii = eval_details['radii_for_features']
    n_circles = centers.shape[0]

    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)

    for i in range(n_circles):
        circle = plt.Circle((centers[i, 0], centers[i, 1]), radii[i], fill=True, alpha=0.7)
        ax.add_artist(circle)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)  # Important to prevent display in the notebook
    buf.seek(0)
    return Image.open(buf)

def simple_feedback_combiner(scores: dict, eval_details: dict, judge_feedback: dict) -> (dict, dict):
    """
    Combines the hard-coded script score and the judge's score.
    This example uses a weighted average to calculate a new 'main_score'.
    """
    original_sum_radii = scores.get('sum_radii', 0.0)
    judge_score_normalized = judge_feedback.get('judge_score', 0.5)
    new_main_score = (original_sum_radii * 1.0) + (original_sum_radii * judge_score_normalized * 0.0)
    scores['main_score'] = new_main_score
    scores['judge_score'] = judge_score_normalized
    return scores, eval_details

# --- Main Execution ---

def run_new_evolution(api_key: str):
    """Configures and runs the evolutionary process using the fluent API."""
    agent = EvoAgent(api_key=api_key)

    # 1. Define the core problem
    agent.define_problem(
        initial_code_fn=get_packer_initial_code,
        evaluator_fn=circle_packer_evaluator,
        evaluator_config={'n_circles': 26},
        feature_extractor_fn=extract_packer_features,
        feature_definitions_fn=get_packer_feature_definitions
    )

    # 2. Configure the LLM Judge for qualitative feedback
    agent.with_llm_judge(
        judge_llm_settings=LLMSettings(
            model_name="gemini/gemini-1.5-flash",
            generation_params={"temperature": 0.2}
        ),
        visual_generator_fn=packing_image_generator,
        feedback_combiner_fn=simple_feedback_combiner
    )

    # 3. Set a logger to visualize progress
    agent.with_generation_logger(visualize_best_packing)

    # 4. Add evolutionary stages
    # Stage 1: Broad exploration
    agent.add_stage(
        name="Explore",
        max_generations=20, # Reduced for quick testing
        evaluation_timeout_seconds=60,
        llm_settings=[LLMSettings(model_name="gemini/gemini-1.5-flash", selection_weight=1.0, generation_params={"temperature": 0.9})],
        task_description="Evolve a Python function to pack 26 non-overlapping circles into a unit square, maximizing the sum of their radii.",
        allow_full_rewrites=True,
        use_map_elites=True,
        self_refine_attempts=5,
        population_size=10,
        candidates_per_ask=3
    )

    # Stage 2: Refinement
    agent.add_stage(
        name="Refine",
        max_generations=30, # Reduced for quick testing
        evaluation_timeout_seconds=90,
        llm_settings=[LLMSettings(model_name="gemini/gemini-1.5-flash", selection_weight=1.0, generation_params={"temperature": 0.4})],
        task_description="Evolve a Python function to pack 26 non-overlapping circles into a unit square, maximizing the sum of their radii.",
        self_refine_attempts=5,
        population_size=10,
        candidates_per_ask=3
    )

    # 5. Run the entire pipeline and handle the new output
    final_output = agent.run()

    # --- MODIFIED HANDLING OF THE FINAL OUTPUT ---
    if final_output:
        print("\n--- PIPELINE EXECUTION SUMMARY ---")
        print(f"Final Stage: '{final_output.stage_name}'")
        print(f"Status: {final_output.status}")
        print(f"Message: {final_output.message}")
        if final_output.best_program:
            best_prog = final_output.best_program
            print("\n--- OVERALL BEST PROGRAM ---")
            print(f"  ID: {best_prog.id}")
            print(f"  Generation: {best_prog.generation}")
            print(f"  Scores: {best_prog.scores}")
            print(f"\n  Final Code for block '{best_prog.block_name}':")
            print("-" * 30)
            print(best_prog.code_str)
            print("-" * 30)
        else:
            print("\nNo single best program was found in the final stage.")
    else:
        print("\nPipeline did not produce a final output.")

if __name__ == "__main__":
    try:
        # Read API key from environment variable for security
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("Please set the GEMINI_API_KEY environment variable")
        print("GEMINI_API_KEY loaded successfully from environment.")
        run_new_evolution(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

print("examples/circle_packing/run_new_fluent_api.py updated to handle StageOutput.")
