from typing import List, Optional, Dict, Any
from .core_types import Program
from .codebase import Codebase

class PromptEngine:
    def __init__(self, task_description: str, problem_specific_instructions: Optional[str] = None):
        self.task_description = task_description
        self.problem_specific_instructions = problem_specific_instructions or ""

    def build_evolution_prompt(self, parent_program: Optional[Program], target_block_name: str,
                               codebase: Codebase, output_format: str = "full_code",
                               context_programs: Optional[List[Program]] = None,
                               allow_full_rewrites: bool = False,
                               refinement_feedback: Optional[str] = None) -> str:

        target_evolve_block = codebase.get_block(target_block_name)
        if not target_evolve_block: raise ValueError(f"Target block '{target_block_name}' not found.")
        current_code_for_prompt = target_evolve_block.current_code.strip()

        parts = ["You are an expert Python programmer and algorithm designer.",
                 f"Your task is to improve a specific block of Python code named '{target_block_name}'."]
        parts.extend(["\n# Overall Task Description:", self.task_description])
        if self.problem_specific_instructions:
            parts.extend(["\n# Specific Instructions & Constraints:", self.problem_specific_instructions])

        # --- NEW: Inject feedback from the LLM Judge ---
        if parent_program and parent_program.judge_feedback:
            parts.append("\n# CRITICAL FEEDBACK FROM EXPERT JUDGE ON YOUR PREVIOUS ATTEMPT:")
            critique = parent_program.judge_feedback.get('critique', 'No critique provided.')
            suggestions = parent_program.judge_feedback.get('suggestions', 'No suggestions provided.')
            parts.append(f"# Judge's Critique: {critique}")
            parts.append(f"# Judge's Suggestions for Improvement: {suggestions}")
            parts.append("# IMPORTANT: You MUST address the judge's feedback in your new version.")

        if refinement_feedback:
            parts.append("\n# FEEDBACK ON PREVIOUS ATTEMPT: Your last generation produced an error.")
            parts.append("Please fix the following error and resubmit your code.")
            parts.append(f"Error details: {refinement_feedback}")
            parts.append(f"\n# Faulty Code Block '{target_block_name}' (to be fixed):")
        elif allow_full_rewrites:
            parts.append("\n# IMPORTANT: You are encouraged to completely rewrite the code for this block from scratch.")
            parts.append("Do not feel bound by the existing implementation if you have a better fundamental approach.")
            parts.append(f"\n# Current Code for Block '{target_block_name}' (for context only, you can ignore it):")
        else:
            parent_info = f"\n# You are improving code from Parent Program (ID: {parent_program.id}, Score: {parent_program.scores.get('main_score', 'N/A'):.4f})."
            parts.append(parent_info)
            parts.append(f"\n# Current Code for Block '{target_block_name}' (this is the code you should modify):")

        parts.extend(["```python", current_code_for_prompt, "```"])

        if context_programs:
            if not allow_full_rewrites:
                parts.append("\n# For inspiration, here are other high-performing examples:")
                for i, ctx_prog in enumerate(context_programs):
                    parts.extend([f"## Context Example {i+1} (Score: {ctx_prog.scores.get('main_score', 'N/A'):.4f}):", "```python", ctx_prog.code_str.strip(), "```"])

        if output_format == "diff":
             parts.extend(["\nPlease provide your modification using the following diff format.", "The SEARCH block must be an *exact verbatim segment* from the 'Current Code for Block' shown above.", "```text\n<<<<<<<< SEARCH\n# Verbatim original lines...\n=========\n# New lines...\n>>>>>>>>> REPLACE\n```"])
        else:
             parts.extend([f"\nPlease provide a new, complete implementation for ONLY the code block '{target_block_name}'.", "Your response should be ONLY the Python code, enclosed in a ```python ... ``` block."])

        return "\n".join(parts)
