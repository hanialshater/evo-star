"""
Aider Evolver - AI-powered code evolution using Aider with examples and context.

This module integrates with Aider to evolve code blocks using base programs
and example implementations for inspiration.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from ..base_agent import BaseAgent
from ..llm_block_evolver.codebase import Codebase
from ...core_types import Program, EvolveBlock, ProgramCandidate
from ...utils.env_loader import get_gemini_api_key, get_openai_api_key


logger = logging.getLogger(__name__)


class AiderEvolver(BaseAgent):
    """
    Aider-powered code evolution agent.

    Uses Aider to evolve code blocks with base programs and example implementations
    for inspiration and context.
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        aider_executable: str = "aider",
        auto_commit: bool = True,
        **kwargs,
    ):
        """Initialize the Aider Evolver."""
        super().__init__(**kwargs)

        self.model = model
        self.aider_executable = aider_executable
        self.auto_commit = auto_commit

        self._setup_environment()
        self._verify_aider_installation()

    def _setup_environment(self):
        """Setup environment variables for Aider."""
        try:
            if self.model.startswith("gemini"):
                api_key = get_gemini_api_key()
                os.environ["GEMINI_API_KEY"] = api_key
            elif self.model.startswith("gpt"):
                api_key = get_openai_api_key()
                os.environ["OPENAI_API_KEY"] = api_key
        except ValueError as e:
            logger.warning(f"API key setup failed: {e}")

    def _verify_aider_installation(self):
        """Verify Aider is installed."""
        try:
            result = subprocess.run(
                [self.aider_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Aider verification failed: {result.stderr}")
            logger.info(f"Aider version: {result.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError(
                f"Aider not found: {self.aider_executable}. "
                "Install with: pip install aider-chat"
            )

    def evolve_with_examples(
        self,
        base_codebase: Codebase,
        block_name: str,
        objective: str,
        example_implementations: List[str] = None,
        context_files: List[str] = None,
        **kwargs,
    ) -> Codebase:
        """
        Evolve a block using Aider with example implementations.

        Args:
            base_codebase: The base codebase to evolve
            block_name: Name of the block to evolve
            objective: Evolution objective
            example_implementations: List of example code implementations
            context_files: Additional context files

        Returns:
            Updated codebase with evolved block
        """
        example_implementations = example_implementations or []
        context_files = context_files or []

        # Get the block to evolve
        block = base_codebase.get_block(block_name)
        if not block:
            raise ValueError(f"Block '{block_name}' not found")

        # Create workspace with base program and examples
        workspace = self._create_workspace_with_examples(
            base_codebase, block_name, example_implementations, context_files
        )

        try:
            # Run Aider evolution
            evolved_code = self._run_aider_with_examples(
                workspace, block_name, objective, **kwargs
            )

            # Update the codebase
            updated_codebase = Codebase(base_codebase.original_full_code)
            updated_codebase.evolve_blocks = base_codebase.evolve_blocks.copy()
            updated_codebase.code_template_parts = (
                base_codebase.code_template_parts.copy()
            )
            updated_codebase.update_block_code(block_name, evolved_code)

            return updated_codebase

        finally:
            self._cleanup_workspace(workspace)

    def _create_workspace_with_examples(
        self,
        base_codebase: Codebase,
        block_name: str,
        example_implementations: List[str],
        context_files: List[str],
    ) -> str:
        """Create workspace with base program and examples."""
        workspace = tempfile.mkdtemp(prefix="aider_evolution_")

        # Write base program
        base_file = Path(workspace) / "base_program.py"
        base_file.write_text(base_codebase.reconstruct_full_code())

        # Write example implementations
        for i, example in enumerate(example_implementations):
            example_file = Path(workspace) / f"example_{i+1}.py"
            example_file.write_text(example)

        # Write context files
        for i, context_content in enumerate(context_files):
            context_file = Path(workspace) / f"context_{i+1}.md"
            context_file.write_text(context_content)

        # Write evolution instructions
        instructions_file = Path(workspace) / "EVOLUTION_INSTRUCTIONS.md"
        instructions_content = f"""# Evolution Instructions

## Target Block: {block_name}

The block to evolve is marked in base_program.py with:
```
# EVOLVE-BLOCK-START {block_name}
... current implementation ...
# EVOLVE-BLOCK-END
```

## Available Resources:
- base_program.py: Main program with the block to evolve
- example_*.py: Example implementations for inspiration
- context_*.md: Additional context and requirements

## Rules:
1. Only modify content INSIDE the evolve block markers
2. Keep the markers themselves unchanged
3. Use examples as inspiration but adapt to the specific context
4. Maintain compatibility with the rest of the program
"""
        instructions_file.write_text(instructions_content)

        # Initialize git
        subprocess.run(["git", "init"], cwd=workspace, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "aider@evolution.ai"],
            cwd=workspace,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Aider Evolution"],
            cwd=workspace,
            capture_output=True,
        )
        subprocess.run(["git", "add", "."], cwd=workspace, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"Setup evolution for {block_name}"],
            cwd=workspace,
            capture_output=True,
        )

        return workspace

    def _run_aider_with_examples(
        self, workspace: str, block_name: str, objective: str, **kwargs
    ) -> str:
        """Run Aider evolution with examples as context."""

        # Get all files in workspace for context
        workspace_path = Path(workspace)
        files = [
            "base_program.py",  # Main file to edit
            "EVOLUTION_INSTRUCTIONS.md",  # Instructions
        ]

        # Add example files
        for example_file in workspace_path.glob("example_*.py"):
            files.append(example_file.name)

        # Add context files
        for context_file in workspace_path.glob("context_*.md"):
            files.append(context_file.name)

        # Prepare Aider command
        cmd = [
            self.aider_executable,
            "--model",
            self.model,
            "--no-auto-commits" if not self.auto_commit else "--auto-commits",
            "--yes",
            *files,  # All files for context
        ]

        # Create evolution prompt
        prompt = self._create_evolution_prompt(block_name, objective, **kwargs)

        try:
            # Run Aider
            result = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                cwd=workspace,
                timeout=300,
            )

            if result.returncode != 0:
                logger.error(f"Aider evolution failed: {result.stderr}")
                raise RuntimeError(f"Aider evolution failed: {result.stderr}")

            # Extract evolved block
            evolved_file = Path(workspace) / "base_program.py"
            evolved_full_code = evolved_file.read_text()

            # Parse evolved code to get the block
            evolved_codebase = Codebase(evolved_full_code)
            evolved_block = evolved_codebase.get_block(block_name)

            if evolved_block:
                return evolved_block.current_code
            else:
                raise RuntimeError(f"Block '{block_name}' not found in evolved code")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Aider evolution timed out")

    def _create_evolution_prompt(
        self, block_name: str, objective: str, **kwargs
    ) -> str:
        """Create evolution prompt for Aider."""
        prompt_parts = [
            f"I need to evolve the '{block_name}' block in base_program.py.",
            f"Objective: {objective}",
            "",
            "Please:",
            f"1. Examine the current implementation in the EVOLVE-BLOCK-START {block_name} section",
            "2. Look at the example implementations for inspiration",
            "3. Review the evolution instructions and context files",
            "4. Improve the block implementation to meet the objective",
            "",
            "IMPORTANT:",
            "- Only modify content INSIDE the evolve block markers",
            "- Keep the markers themselves unchanged",
            "- Use examples as inspiration but adapt to this specific context",
            "- Ensure the changes integrate well with the rest of the program",
            "",
        ]

        # Add specific requirements from kwargs
        if kwargs.get("requirements"):
            prompt_parts.append(f"Requirements: {kwargs['requirements']}")

        if kwargs.get("constraints"):
            prompt_parts.append(f"Constraints: {kwargs['constraints']}")

        if kwargs.get("performance_target"):
            prompt_parts.append(f"Performance target: {kwargs['performance_target']}")

        prompt_parts.extend(
            [
                "",
                "Please implement the improvements and explain your changes.",
                "/commit",
            ]
        )

        return "\n".join(prompt_parts)

    def _cleanup_workspace(self, workspace: str):
        """Clean up workspace."""
        import shutil

        try:
            shutil.rmtree(workspace)
        except Exception as e:
            logger.warning(f"Failed to cleanup workspace {workspace}: {e}")

    def crossover_with_aider(
        self,
        parent1_codebase: Codebase,
        parent2_codebase: Codebase,
        block_name: str,
        objective: str = "Combine the best features from both implementations",
    ) -> Codebase:
        """
        Perform crossover between two codebases using Aider.

        Args:
            parent1_codebase: First parent codebase
            parent2_codebase: Second parent codebase
            block_name: Block to crossover
            objective: Crossover objective

        Returns:
            New codebase with crossed-over block
        """
        # Get both implementations
        block1 = parent1_codebase.get_block(block_name)
        block2 = parent2_codebase.get_block(block_name)

        if not block1 or not block2:
            raise ValueError(f"Block '{block_name}' not found in both parents")

        # Use both implementations as examples
        examples = [
            f"# Implementation 1:\n{block1.current_code}",
            f"# Implementation 2:\n{block2.current_code}",
        ]

        # Use parent1 as base and evolve with parent2 as example
        return self.evolve_with_examples(
            parent1_codebase,
            block_name,
            objective,
            examples,
            requirements="Combine the best aspects of both implementations",
        )

    def mutation_with_aider(
        self,
        codebase: Codebase,
        block_name: str,
        mutation_type: str = "improve",
        **kwargs,
    ) -> Codebase:
        """
        Perform mutation using Aider.

        Args:
            codebase: Codebase to mutate
            block_name: Block to mutate
            mutation_type: Type of mutation (improve, optimize, refactor, etc.)
            **kwargs: Additional mutation parameters

        Returns:
            Mutated codebase
        """
        objectives = {
            "improve": "Improve the code quality and readability",
            "optimize": "Optimize for better performance",
            "refactor": "Refactor for better structure and maintainability",
            "robust": "Make the code more robust and handle edge cases",
            "creative": "Add creative improvements or new features",
        }

        objective = objectives.get(mutation_type, mutation_type)

        return self.evolve_with_examples(codebase, block_name, objective, **kwargs)

    async def evolve(
        self,
        candidate: ProgramCandidate,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProgramCandidate:
        """
        Evolve a program candidate using Aider.

        Args:
            candidate: The candidate program to evolve
            task_description: Description of the task/improvement needed
            context: Additional context for the evolution process

        Returns:
            Evolved program candidate
        """
        context = context or {}

        # Create codebase from candidate
        codebase = Codebase(candidate.code_str)

        # Get block name from candidate or use default
        block_name = candidate.block_name or "main"

        # If no blocks exist, create a simple wrapper
        if not codebase.get_block_names():
            # Wrap the entire code as an evolve block
            wrapped_code = f"""
# EVOLVE-BLOCK-START {block_name}
{candidate.code_str}
# EVOLVE-BLOCK-END
"""
            codebase = Codebase(wrapped_code)

        # Get example implementations from context
        examples = context.get("examples", [])
        context_files = context.get("context_files", [])

        # Evolve the codebase
        evolved_codebase = self.evolve_with_examples(
            codebase, block_name, task_description, examples, context_files, **context
        )

        # Create evolved candidate
        evolved_candidate = ProgramCandidate(
            id=f"{candidate.id}_evolved",
            code_str=evolved_codebase.get_block(block_name).current_code,
            block_name=block_name,
            parent_id=candidate.id,
            generation=candidate.generation + 1,
        )

        return evolved_candidate

    def get_capabilities(self) -> List[str]:
        """
        Get the capabilities of this agent.

        Returns:
            List of capabilities this agent supports
        """
        return [
            "code_evolution",
            "block_evolution",
            "example_based_learning",
            "crossover",
            "mutation",
            "performance_optimization",
            "code_refactoring",
            "feature_addition",
            "aider_integration",
            "multi_language_support",
            "context_aware_evolution",
        ]
