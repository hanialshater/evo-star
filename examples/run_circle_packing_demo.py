#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circle Packing Evolution Demo - Standalone Script

This script demonstrates the EvoAgent framework by evolving a Python function 
to pack 26 circles into a unit square, maximizing the sum of their radii.

Run this script directly with: python examples/run_circle_packing_demo.py
"""

import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import traceback
import json
import re
import os

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpha_evolve_framework.fluent_api import EvoAgent
from alpha_evolve_framework.core_types import LLMSettings, Program
from alpha_evolve_framework.codebase import Codebase
from examples.circle_packing.problem_functions import (
    get_packer_initial_code,
    circle_packer_evaluator,
    extract_packer_features,
    get_packer_feature_definitions
)

# Configuration
import os

# Read API key from environment variable for security
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set the GEMINI_API_KEY environment variable")

MODEL_NAME = "gemini/gemini-1.5-flash"

# Evolution Configuration
N_CIRCLES = 26
EXPLORE_GENERATIONS = 15  # Reduced for demo
REFINE_GENERATIONS = 20   # Reduced for demo
POPULATION_SIZE = 8       # Reduced for demo
CANDIDATES_PER_ASK = 2    # Reduced for demo

def visualize_best_packing(generation: int, best_program: Program, full_codebase: Codebase):
    """
    Visualizes the best circle packing solution at the end of a generation.
    """
    if not best_program or not best_program.eval_details.get('is_valid', False):
        print(f"  📊 Gen {generation}: Skipping visualization for invalid program.")
        return

    n_circles = best_program.eval_details['centers_for_features'].shape[0]
    centers = best_program.eval_details['centers_for_features']
    radii = best_program.eval_details['radii_for_features']

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal', adjustable='box')

    # Create a colormap for the circles
    colors = plt.cm.viridis(np.linspace(0, 1, n_circles))
    
    title_str = (f"🏆 Best of Generation {generation}\n"
                 f"Score: {best_program.scores.get('main_score', 0):.4f} | "
                 f"Sum of Radii: {np.sum(radii):.4f} | "
                 f"Program: {best_program.id}")
    ax.set_title(title_str, fontsize=12, pad=20)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Draw unit square boundary
    boundary = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(boundary)

    # Draw circles
    for i in range(n_circles):
        circle = plt.Circle((centers[i, 0], centers[i, 1]), radii[i], 
                          fill=True, alpha=0.7, color=colors[i],
                          edgecolor='black', linewidth=0.5)
        ax.add_artist(circle)
        
        # Add circle number
        if radii[i] > 0.02:  # Only show numbers for larger circles
            ax.text(centers[i, 0], centers[i, 1], str(i+1), 
                   ha='center', va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'generation_{generation}_best.png', dpi=150, bbox_inches='tight')
    plt.show()

def packing_image_generator(eval_details: dict) -> Image.Image:
    """
    Generates a PIL Image from the packing data for the LLM Judge.
    """
    if not eval_details.get('is_valid', False) or 'centers_for_features' not in eval_details:
        return None

    centers = eval_details['centers_for_features']
    radii = eval_details['radii_for_features']
    n_circles = centers.shape[0]

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)

    colors = plt.cm.viridis(np.linspace(0, 1, n_circles))
    
    for i in range(n_circles):
        circle = plt.Circle((centers[i, 0], centers[i, 1]), radii[i], 
                          fill=True, alpha=0.8, color=colors[i])
        ax.add_artist(circle)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def simple_feedback_combiner(scores: dict, eval_details: dict, judge_feedback: dict) -> tuple:
    """
    Combines the algorithmic score and the judge's qualitative score.
    """
    original_sum_radii = scores.get('sum_radii', 0.0)
    judge_score_normalized = judge_feedback.get('judge_score', 0.5)
    
    # Weighted combination: 80% algorithmic, 20% judge
    new_main_score = (original_sum_radii * 0.8) + (original_sum_radii * judge_score_normalized * 0.2)
    scores['main_score'] = new_main_score
    scores['judge_score'] = judge_score_normalized
    
    return scores, eval_details

def run_circle_packing_evolution():
    """Configures and runs the evolutionary process using the fluent API."""
    print("🚀 Starting Circle Packing Evolution...\n")
    print(f"🔧 Configuration:")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Circles to pack: {N_CIRCLES}")
    print(f"   Total generations: {EXPLORE_GENERATIONS + REFINE_GENERATIONS}")
    print(f"   Population size: {POPULATION_SIZE}")
    print()
    
    # Create the EvoAgent
    agent = EvoAgent(api_key=GEMINI_API_KEY)

    # 1. Define the core problem
    agent.define_problem(
        initial_code_fn=get_packer_initial_code,
        evaluator_fn=circle_packer_evaluator,
        evaluator_config={'n_circles': N_CIRCLES},
        feature_extractor_fn=extract_packer_features,
        feature_definitions_fn=get_packer_feature_definitions
    )

    # 2. Configure the LLM Judge for qualitative feedback
    agent.with_llm_judge(
        judge_llm_settings=LLMSettings(
            model_name=MODEL_NAME,
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
        max_generations=EXPLORE_GENERATIONS,
        evaluation_timeout_seconds=60,
        llm_settings=[LLMSettings(
            model_name=MODEL_NAME, 
            selection_weight=1.0, 
            generation_params={"temperature": 0.9}
        )],
        task_description=f"Evolve a Python function to pack {N_CIRCLES} non-overlapping circles into a unit square, maximizing the sum of their radii.",
        allow_full_rewrites=True,
        use_map_elites=True,
        self_refine_attempts=3,
        population_size=POPULATION_SIZE,
        candidates_per_ask=CANDIDATES_PER_ASK
    )

    # Stage 2: Refinement
    agent.add_stage(
        name="Refine",
        max_generations=REFINE_GENERATIONS,
        evaluation_timeout_seconds=90,
        llm_settings=[LLMSettings(
            model_name=MODEL_NAME, 
            selection_weight=1.0, 
            generation_params={"temperature": 0.4}
        )],
        task_description=f"Refine the Python function to pack {N_CIRCLES} non-overlapping circles into a unit square, maximizing the sum of their radii.",
        self_refine_attempts=5,
        population_size=POPULATION_SIZE,
        candidates_per_ask=CANDIDATES_PER_ASK
    )

    # 5. Run the entire pipeline
    final_output = agent.run()
    
    return final_output

def analyze_results(final_result):
    """Analyze and display the final results."""
    if not final_result:
        print("❌ No results to analyze - evolution may have failed.")
        return

    print("\n📊 PIPELINE EXECUTION SUMMARY")
    print("=" * 50)
    print(f"Final Stage: '{final_result.stage_name}'")
    print(f"Status: {final_result.status}")
    print(f"Message: {final_result.message}")
    
    if final_result.best_program:
        best_prog = final_result.best_program
        
        print("\n🏆 OVERALL BEST PROGRAM")
        print("=" * 50)
        print(f"  🆔 Program ID: {best_prog.id}")
        print(f"  🧬 Generation: {best_prog.generation}")
        print(f"  📈 Main Score: {best_prog.scores.get('main_score', 'N/A'):.4f}")
        
        if 'sum_radii' in best_prog.scores:
            print(f"  🔵 Sum of Radii: {best_prog.scores['sum_radii']:.4f}")
        
        if 'judge_score' in best_prog.scores:
            print(f"  🤖 Judge Score: {best_prog.scores['judge_score']:.4f}")
        
        if best_prog.features:
            feature_names = [f['name'] for f in get_packer_feature_definitions()]
            print(f"  🎯 Features: {dict(zip(feature_names, best_prog.features))}")
        
        print(f"\n💻 EVOLVED CODE for block '{best_prog.block_name}':")
        print("-" * 50)
        print(best_prog.code_str)
        print("-" * 50)
        
        # Visualize the final best solution
        print("\n🎨 Creating Final Visualization...")
        if best_prog.eval_details.get('is_valid', False):
            centers = best_prog.eval_details['centers_for_features']
            radii = best_prog.eval_details['radii_for_features']
            
            fig, ax = plt.subplots(1, figsize=(12, 12))
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal', adjustable='box')

            colors = plt.cm.viridis(np.linspace(0, 1, len(centers)))
            
            title_str = (f"🏆 FINAL BEST SOLUTION\n"
                        f"Program: {best_prog.id} | Score: {best_prog.scores.get('main_score', 0):.4f} | "
                        f"Sum of Radii: {np.sum(radii):.4f}")
            ax.set_title(title_str, fontsize=16, pad=20)
            ax.grid(True, linestyle='--', alpha=0.3)

            # Draw unit square boundary
            boundary = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='red', linewidth=3)
            ax.add_patch(boundary)

            # Draw circles
            for i, (center, radius) in enumerate(zip(centers, radii)):
                circle = plt.Circle(center, radius, 
                                  fill=True, alpha=0.7, color=colors[i],
                                  edgecolor='black', linewidth=0.8)
                ax.add_artist(circle)
                
                # Add circle number for larger circles
                if radius > 0.02:
                    ax.text(center[0], center[1], str(i+1), 
                           ha='center', va='center', fontsize=10, fontweight='bold')

            plt.tight_layout()
            plt.savefig('final_best_solution.png', dpi=200, bbox_inches='tight')
            plt.show()
            
            # Print some statistics
            print(f"\n📊 STATISTICS:")
            print(f"   Total circles packed: {len(centers)}")
            print(f"   Average radius: {np.mean(radii):.4f}")
            print(f"   Largest radius: {np.max(radii):.4f}")
            print(f"   Smallest radius: {np.min(radii):.4f}")
            print(f"   Standard deviation of radii: {np.std(radii):.4f}")
            
            # Calculate coverage
            total_area = np.sum(np.pi * radii**2)
            coverage = total_area * 100
            print(f"   Area coverage: {coverage:.2f}% of unit square")
            
        else:
            print("   ❌ Final solution is invalid - no visualization available")
    
    else:
        print("\n❌ No best program found in the final stage.")
        
    print(f"\n📈 POPULATION SUMMARY:")
    print(f"   Total programs in final population: {len(final_result.final_population)}")
    
    if final_result.final_population:
        scores = [p.scores.get('main_score', -999) for p in final_result.final_population]
        valid_scores = [s for s in scores if s > -999]
        
        if valid_scores:
            print(f"   Best score in population: {max(valid_scores):.4f}")
            print(f"   Average score: {np.mean(valid_scores):.4f}")
            print(f"   Score range: [{min(valid_scores):.4f}, {max(valid_scores):.4f}]")
            print(f"   Valid programs: {len(valid_scores)}/{len(final_result.final_population)}")

def test_evolved_solution(final_result):
    """Test the evolved solution independently."""
    if not final_result or not final_result.best_program:
        print("⚠️  No evolved solution available to test.")
        return

    print("\n🧪 TESTING EVOLVED SOLUTION")
    print("=" * 40)
    
    try:
        # Create a complete codebase with the evolved solution
        initial_code = get_packer_initial_code()
        codebase = Codebase(initial_code)
        
        # Update with the evolved code
        best_prog = final_result.best_program
        codebase.update_block_code(best_prog.block_name, best_prog.code_str)
        
        # Get the complete code
        complete_code = codebase.reconstruct_full_code()
        
        print("✅ Executing evolved solution...")
        
        # Execute the evolved code
        exec_globals = {}
        exec(complete_code, exec_globals)
        
        # Test the harness function
        harness_fn = exec_globals.get("run_evaluation_harness")
        if callable(harness_fn):
            centers, error_msg = harness_fn(N_CIRCLES)
            
            if error_msg:
                print(f"❌ Test failed: {error_msg}")
            else:
                print(f"✅ Test successful!")
                print(f"   Generated {len(centers)} circle centers")
                print(f"   Centers shape: {centers.shape}")
                print(f"   Center coordinates range: [{centers.min():.3f}, {centers.max():.3f}]")
                
                # Quick validation
                scores, details = circle_packer_evaluator(complete_code, {'n_circles': N_CIRCLES})
                
                print(f"\n📊 Independent evaluation:")
                print(f"   Score: {scores.get('main_score', 'N/A'):.4f}")
                print(f"   Sum of radii: {scores.get('sum_radii', 'N/A'):.4f}")
                print(f"   Valid solution: {details.get('is_valid', False)}")
                
        else:
            print("❌ Could not find harness function in evolved code")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        traceback.print_exc()

def main():
    """Main execution function."""
    print("🎯 Circle Packing Evolution Demo")
    print("=" * 60)
    print()
    
    # Show initial problem setup
    print("📝 Initial code to be evolved:")
    print("=" * 30)
    initial_code = get_packer_initial_code()
    print(initial_code[:500] + "..." if len(initial_code) > 500 else initial_code)
    print("=" * 30)
    
    # Show feature definitions
    feature_defs = get_packer_feature_definitions()
    print(f"\n🎯 MAP-Elites feature definitions:")
    for i, feat in enumerate(feature_defs):
        print(f"   {i+1}. {feat['name']}: [{feat['min_val']:.2f}, {feat['max_val']:.2f}] with {feat['bins']} bins")
    
    print("\n" + "=" * 60)
    
    try:
        print("🎬 Starting Evolution Run...")
        final_result = run_circle_packing_evolution()
        
        print("\n🏁 Evolution Complete!")
        analyze_results(final_result)
        test_evolved_solution(final_result)
        
        print(f"\n✅ Demo completed! Check generated image files:")
        print(f"   - generation_*.png (progress visualizations)")
        print(f"   - final_best_solution.png (final result)")
        
    except Exception as e:
        print(f"❌ Error during evolution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
