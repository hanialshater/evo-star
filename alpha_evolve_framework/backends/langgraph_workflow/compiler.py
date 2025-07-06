"""
Workflow compiler for translating fluent API to LangGraph workflows.
"""

from typing import Dict, Any, List
import logging

from .workflow import create_evolution_workflow
from ...core_types import LLMSettings

logger = logging.getLogger(__name__)


class WorkflowCompiler:
    """Compiles fluent API configurations into LangGraph workflows."""

    def __init__(self):
        self.compiled_workflows = {}

    def compile_stage_config(self, stage_config: Dict[str, Any]) -> Any:
        """
        Compile a single stage configuration into a LangGraph workflow.

        For POC, this is simple - just return the basic workflow.
        In the future, this will generate different workflows based on configuration.
        """
        logger.info(f"Compiling stage: {stage_config.get('name', 'unnamed')}")

        # For POC, we use the same simple workflow for all stages
        # In the future, we'll generate different workflows based on:
        # - Island model configuration
        # - Evaluation strategy
        # - Migration patterns
        # - Convergence criteria

        workflow_key = self._generate_workflow_key(stage_config)

        if workflow_key not in self.compiled_workflows:
            # Create workflow for this configuration
            workflow = create_evolution_workflow()
            self.compiled_workflows[workflow_key] = workflow
            logger.info(f"Created new workflow for key: {workflow_key}")
        else:
            workflow = self.compiled_workflows[workflow_key]
            logger.info(f"Reusing existing workflow for key: {workflow_key}")

        return workflow

    def _generate_workflow_key(self, stage_config: Dict[str, Any]) -> str:
        """Generate a unique key for the workflow based on configuration."""
        # For POC, all stages use the same workflow
        # In the future, this will consider:
        # - Number of islands
        # - Migration strategy
        # - Evaluation type
        # - Special features (MAP-Elites, etc.)

        key_parts = [
            "simple_evolution",  # Base workflow type
            str(stage_config.get("num_islands", 1)),
            stage_config.get("migration_strategy", "none"),
            "map_elites" if stage_config.get("use_map_elites", False) else "simple",
        ]

        return "_".join(key_parts)

    def compile_multi_stage_pipeline(
        self, stages_config: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Compile multiple stages into a pipeline of workflows.

        For POC, this just compiles each stage independently.
        In the future, this could create a single mega-workflow that handles
        stage transitions automatically.
        """
        logger.info(f"Compiling multi-stage pipeline with {len(stages_config)} stages")

        compiled_stages = []
        for i, stage_config in enumerate(stages_config):
            logger.info(
                f"Compiling stage {i+1}/{len(stages_config)}: {stage_config.get('name', f'stage_{i}')}"
            )
            workflow = self.compile_stage_config(stage_config)
            compiled_stages.append(workflow)

        return compiled_stages

    def get_workflow_info(self, workflow_key: str) -> Dict[str, Any]:
        """Get information about a compiled workflow."""
        if workflow_key not in self.compiled_workflows:
            return {}

        # For POC, return basic info
        # In the future, this could return detailed workflow structure,
        # node counts, estimated resource requirements, etc.
        return {
            "workflow_key": workflow_key,
            "type": "simple_evolution",
            "nodes": [
                "initialize_population",
                "generate_candidates",
                "evaluate_candidates",
                "update_population",
                "check_termination",
            ],
            "supports_checkpointing": True,
            "supports_streaming": True,
        }

    def clear_cache(self):
        """Clear the compiled workflow cache."""
        logger.info("Clearing workflow cache")
        self.compiled_workflows.clear()


# Global compiler instance
_compiler = WorkflowCompiler()


def get_compiler() -> WorkflowCompiler:
    """Get the global workflow compiler instance."""
    return _compiler


def compile_fluent_api_to_langgraph(stages_config: List[Dict[str, Any]]) -> List[Any]:
    """
    Main entry point for compiling fluent API configurations to LangGraph workflows.

    This is the function that the enhanced EvoAgent will call to get LangGraph workflows
    from its fluent API configuration.
    """
    compiler = get_compiler()
    return compiler.compile_multi_stage_pipeline(stages_config)


# Future expansion functions (stubs for now)


def compile_island_model_workflow(
    num_islands: int, migration_config: Dict[str, Any]
) -> Any:
    """Compile a workflow with island model support."""
    # TODO: Implement island model workflow generation
    logger.warning("Island model workflow compilation not yet implemented")
    return create_evolution_workflow()


def compile_map_elites_workflow(feature_definitions: List[Dict[str, Any]]) -> Any:
    """Compile a workflow with MAP-Elites support."""
    # TODO: Implement MAP-Elites workflow generation
    logger.warning("MAP-Elites workflow compilation not yet implemented")
    return create_evolution_workflow()


def compile_custom_workflow(workflow_spec: Dict[str, Any]) -> Any:
    """Compile a custom workflow from specification."""
    # TODO: Implement custom workflow generation from specifications
    logger.warning("Custom workflow compilation not yet implemented")
    return create_evolution_workflow()
