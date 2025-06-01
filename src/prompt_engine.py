from typing import List, Optional, Dict, Any
from .core_types import Program # Relative import
from .codebase import Codebase # Relative import

class PromptEngine:
    def __init__(self, task_description: str, problem_specific_instructions: Optional[str] = None):
        self.task_description = task_description
        self.problem_specific_instructions = problem_specific_instructions or ""
        print("PromptEngine initialized.")

    def build_evolution_prompt(self, parent_program: Optional[Program], target_block_name: str,
                               codebase: Codebase, output_format: str = "full_code",
                               context_programs: Optional[List[Program]] = None,
                               previous_attempts_feedback: Optional[List[Dict[str, Any]]] = None) -> str:
        target_evolve_block = codebase.get_block(target_block_name)
        if not target_evolve_block: raise ValueError(f"Target block '{target_block_name}' not in codebase for prompt.")
        current_code_for_prompt = target_evolve_block.current_code.replace('\r\n', '\n').strip()

        parts = ["You are an expert Python programmer and algorithm designer.",
                 f"Your task is to improve a specific block of Python code named '{target_block_name}'."]
        parts.extend(["\n# Overall Task Description:", self.task_description])
        if self.problem_specific_instructions:
            parts.extend(["\n# Specific Instructions & Constraints for this Problem:", self.problem_specific_instructions])

        parent_section = "\n# Context: You are generating an initial version for this block."
        if parent_program:
            parent_section = (f"\n# You are improving code derived from Parent Program (ID: {parent_program.id}, Gen: {parent_program.generation}):"
                             f"\n  Parent's main_score for this block: {parent_program.scores.get('main_score', 'N/A'):.4f}.")
            if parent_program.eval_details:
                feedback = str(parent_program.eval_details.get('error_message', parent_program.eval_details.get('feedback', 'No specific feedback.')))
                parent_section += f"\n  Parent's Evaluation Feedback: {feedback[:200]}{'...' if len(feedback) > 200 else ''}"
        parts.append(parent_section)
        parts.extend([f"\n# Current Code for Block '{target_block_name}' (this is the code you should modify):", "```python", current_code_for_prompt, "```"])

        if context_programs:
            parts.append("\n# Here are some other relevant program examples and their outcomes for inspiration:")
            for i, ctx_prog in enumerate(context_programs):
                ctx_feedback = str(ctx_prog.eval_details.get('error_message', ctx_prog.eval_details.get('feedback', 'N/A')))
                parts.extend([f"\n## Context Example {i+1} (ID: {ctx_prog.id}, Gen: {ctx_prog.generation}, Block: '{ctx_prog.block_name}')",
                              f"   Main Score: {ctx_prog.scores.get('main_score', 'N/A'):.4f}",
                              f"   Feedback: {ctx_feedback[:150]}{'...' if len(ctx_feedback) > 150 else ''}",
                              f"   Code for Block '{ctx_prog.block_name}':", "   ```python",
                              ctx_prog.code_str.replace('\r\n', '\n').strip(), "   ```"])
        if previous_attempts_feedback:
            parts.append("\n# Feedback from recent unsuccessful attempts on this block in the current generation:")
            for i, attempt in enumerate(previous_attempts_feedback):
                feedback = str(attempt.get('feedback', 'No details.'))
                parts.append(f"- Attempt {i+1}: Score {attempt.get('score', 'N/A')}, Feedback: {feedback[:150]}{'...' if len(feedback)>150 else ''}")

        if output_format == "diff":
            parts.extend([f"\nPlease provide your modification for code block '{target_block_name}' using the following diff format.",
                          "The SEARCH block must be an *exact verbatim segment* from the 'Current Code for Block' shown above.",
                          "```text", "<<<<<<<< SEARCH", "# Verbatim original lines...", "=========", "# New lines...", ">>>>>>>>> REPLACE", "```",
                          "Your entire response should be ONLY this diff block, enclosed in a single markdown code block (e.g. ```text ... ``` or ``` ... ```)."])
        else: # "full_code"
            parts.extend([f"\nPlease provide a new, complete implementation for ONLY the code block '{target_block_name}'.",
                          "Your response should be ONLY the Python code for this block, enclosed in:",
                          "```python", "# Your new code for the block here...", "```"])
        parts.extend(["\nIMPORTANT: Do NOT include the '# EVOLVE-BLOCK-START ...' or '# EVOLVE-BLOCK-END' markers in your response.",
                      "Focus on functional improvements for a better evaluation score based on the task description."])
        return "\n".join(parts)

print("alpha_evolve_framework/prompt_engine.py written")
