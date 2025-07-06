# -*- coding: utf-8 -*-
"""
This script demonstrates the use of the EvoAgent framework with the fluent API
for the HTML city page generation problem, consolidated into a single file.
"""
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import traceback
import json
import os
import asyncio
from typing import Dict, Any, Tuple, Optional, List, Callable
from tempfile import NamedTemporaryFile
import re

# Ensure the project root is in the Python path for framework imports
if '' not in sys.path:
    sys.path.insert(0, '')

from alpha_evolve_framework.fluent_api import EvoAgent
from alpha_evolve_framework.core_types import LLMSettings, Program
from alpha_evolve_framework.codebase import Codebase

# --- Problem Function 1: Initial Code ---
def get_city_initial_html_code() -> str:
    """Provides the initial HTML code string for the city page problem."""
    html_code = """
<!-- EVOLVE-BLOCK-START city_content -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>City Page</title>
    <style>
        body { font-family: sans-serif; margin: 0; padding: 0;}
        img { max-width: 100%; height: auto; display: block; margin: 10px auto; }
        .container { max-width: 800px; margin: 20px auto; padding: 0 20px; }
        h1, h2 { text-align: center; }
    </style>
</head>
<body>

    <div class="container">
        <div>
            <h1>City Name Placeholder</h1>
            <img src="https://via.placeholder.com/600x400" alt="City Image Placeholder">
            <p>This is a placeholder for the city description. The goal is to replace this with interesting facts and details about a specific city, maybe add more sections like attractions or history.</p>
            <h2>Attractions</h2>
            <ul>
                <li>Placeholder attraction 1</li>
                <li>Placeholder attraction 2</li>
            </ul>
        </div>
    </div>

</body>
</html>
<!-- EVOLVE-BLOCK-END -->
"""
    return html_code

# --- Problem Function 2: Evaluator ---
def city_html_evaluator(full_code_to_evaluate: str, config: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Evaluates an HTML city page solution using basic text analysis.
    This is a simplified version that doesn't require Playwright.
    """
    scores: Dict[str, float] = {'main_score': -1000.0}
    details: Dict[str, Any] = {'error_message': 'Evaluation failed to start', 'is_valid': False}
    
    try:
        # Basic HTML content analysis
        html_lower = full_code_to_evaluate.lower()
        
        # Count HTML elements
        h1_count = html_lower.count('<h1')
        h2_count = html_lower.count('<h2')
        h3_count = html_lower.count('<h3')
        p_count = html_lower.count('<p')
        img_count = html_lower.count('<img')
        div_count = html_lower.count('<div')
        ul_count = html_lower.count('<ul')
        ol_count = html_lower.count('<ol')
        a_count = html_lower.count('<a ')
        
        # Check for meaningful content (not just placeholders)
        meaningful_words = len(re.findall(r'\b(?!city|name|placeholder|this|is|a|for|the|and|or|like|more|add|section|sections|attraction|attractions|history|details)\w{4,}\b', full_code_to_evaluate, re.IGNORECASE))
        
        details.update({
            'h1_count': h1_count,
            'h2_count': h2_count,
            'h3_count': h3_count,
            'p_count': p_count,
            'img_count': img_count,
            'div_count': div_count,
            'ul_count': ul_count,
            'ol_count': ol_count,
            'link_count': a_count,
            'list_count': ul_count + ol_count,
            'meaningful_words': meaningful_words
        })
        
        # Calculate score based on content richness
        score = 0.0
        feedback = []
        
        # Reward for essential elements
        score += min(h1_count, 1) * 3.0  # Reward for at least one h1
        score += min(h2_count + h3_count, 3) * 1.0  # Reward for subheadings
        score += min(p_count, 5) * 0.5  # Reward for paragraphs
        score += min(img_count, 3) * 1.5  # Reward for images
        score += min(ul_count + ol_count, 3) * 1.0  # Reward for lists
        score += min(a_count, 5) * 0.5  # Reward for links
        
        # Reward for content richness
        score += min(meaningful_words * 0.02, 5.0)  # Reward for meaningful text
        
        # Penalize placeholder content
        if 'placeholder' in html_lower:
            feedback.append("Placeholder text still present.")
            score -= 2.0
        
        # Bonus for well-structured content
        if h1_count >= 1 and (h2_count + h3_count) >= 2 and p_count >= 3:
            score += 2.0
            feedback.append("Well-structured content.")
        
        scores['main_score'] = score
        details['evaluation_feedback'] = " | ".join(feedback) if feedback else "Basic evaluation passed."
        details['error_message'] = None
        details['is_valid'] = True
        
    except Exception as e:
        scores['main_score'] = -1000.0
        details['error_message'] = f'Evaluation failed: {e}'
        details['is_valid'] = False
    
    return scores, details

# --- Problem Function 3: Feature Extractor ---
def extract_city_features(eval_details: Dict[str, Any], config: Dict[str, Any]) -> Optional[Tuple[float, ...]]:
    """
    Extracts features from evaluation details for MAP-Elites.
    """
    if not eval_details.get('is_valid', False):
        return None
    
    h1_count = eval_details.get('h1_count', 0)
    p_count = eval_details.get('p_count', 0)
    img_count = eval_details.get('img_count', 0)
    list_count = eval_details.get('list_count', 0)
    link_count = eval_details.get('link_count', 0)
    
    return (float(h1_count), float(p_count), float(img_count), float(list_count), float(link_count))

# --- Problem Function 4: Feature Definitions ---
def get_city_feature_definitions() -> List[Dict[str, Any]]:
    """
    Returns feature definitions for MAP-Elites for the city page problem.
    """
    return [
        {'name': 'h1_count', 'min_val': 0.0, 'max_val': 3.0, 'bins': 4},
        {'name': 'p_count', 'min_val': 0.0, 'max_val': 10.0, 'bins': 11},
        {'name': 'img_count', 'min_val': 0.0, 'max_val': 5.0, 'bins': 6},
        {'name': 'list_count', 'min_val': 0.0, 'max_val': 5.0, 'bins': 6},
        {'name': 'link_count', 'min_val': 0.0, 'max_val': 10.0, 'bins': 11},
    ]

# --- Visualization Functions ---
def visualize_best_city_page(generation: int, best_program: Program, full_codebase: Codebase):
    """
    A logger function that displays information about the best city page at the end of a generation.
    """
    print(f"\n--- Best Program of Generation {generation} ---")
    if not best_program:
        print("  No best program found for this generation.")
        return
    
    print(f"  ID: {best_program.id}")
    print(f"  Score: {best_program.scores.get('main_score', 0):.4f}")
    print(f"  Feedback: {best_program.eval_details.get('evaluation_feedback', 'N/A')}")
    
    # Show element counts
    details = best_program.eval_details
    print(f"  Elements: H1={details.get('h1_count', 0)}, P={details.get('p_count', 0)}, IMG={details.get('img_count', 0)}, Lists={details.get('list_count', 0)}")
    print(f"  Meaningful words: {details.get('meaningful_words', 0)}")

def city_image_generator(eval_details: dict) -> Optional[Image.Image]:
    """
    Placeholder for image generation - in a real implementation, this could
    take a screenshot of the rendered HTML.
    """
    return None

def simple_city_feedback_combiner(scores: dict, eval_details: dict, judge_feedback: dict) -> Tuple[dict, dict]:
    """
    Combines the script score and the judge's score for the city page.
    """
    judge_score_normalized = judge_feedback.get('judge_score', 0.5)
    original_score = scores.get('main_score', 0.0)
    
    # Boost score based on judge feedback
    scores['main_score'] = original_score * (1.0 + judge_score_normalized * 0.5)
    scores['judge_score'] = judge_score_normalized
    return scores, eval_details

# --- Main Execution ---
def run_city_evolution(api_key: str):
    """Configures and runs the evolutionary process using the fluent API for the city page task."""
    agent = EvoAgent(api_key=api_key)

    # 1. Define the core problem
    agent.define_problem(
        initial_code_fn=get_city_initial_html_code,
        evaluator_fn=city_html_evaluator,
        evaluator_config={},
        feature_extractor_fn=extract_city_features,
        feature_definitions_fn=get_city_feature_definitions
    )

    # 2. Configure the LLM Judge (Optional)
    agent.with_llm_judge(
        judge_llm_settings=LLMSettings(
            model_name="gemini-2.0-flash-exp",
            generation_params={"temperature": 0.2}
        ),
        visual_generator_fn=city_image_generator,
        feedback_combiner_fn=simple_city_feedback_combiner
    )

    # 3. Set a logger to visualize progress
    agent.with_generation_logger(visualize_best_city_page)

    # 4. Add evolutionary stages
    agent.add_stage(
        name="CityPageEvo",
        max_generations=15,
        evaluation_timeout_seconds=60,
        llm_settings=[LLMSettings(model_name="gemini-2.0-flash-exp", selection_weight=1.0, generation_params={"temperature": 0.8})],
        task_description="Evolve an HTML code block to create a visually appealing and informative web page about a city.",
        problem_specific_instructions="""
        Focus on creating rich, engaging content about a specific city. Include:
        1. A compelling title and description
        2. Multiple sections (attractions, history, culture, etc.)
        3. Proper HTML structure with semantic elements
        4. Remove all placeholder content
        5. Add real information about a city of your choice
        6. Include diverse content types (lists, paragraphs, etc.)
        """,
        allow_full_rewrites=True,
        use_map_elites=True,
        self_refine_attempts=3,
        population_size=10,
        candidates_per_ask=2
    )

    # 5. Run the entire pipeline and handle the output
    final_output = agent.run()

    # --- Handling of the Final Output ---
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
        run_city_evolution(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

print("examples/city_page/run_city_evolution.py created with simplified evaluator.")
