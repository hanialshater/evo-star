"""
LangGraph backend for workflow-based evolution.
"""

import logging
from typing import Dict, List, Optional, Any

from .base_backend import EvolutionBackend
from ..core_types import Program, StageOutput, LLMSettings

logger = logging.getLogger(__name__)


class LangGraphBackend(EvolutionBackend):
    """Backend using LangGraph workflow orchestration."""

    def __init__(self, api_key: str):
        """Initialize LangGraph backend."""
        super().__init__(api_key)
        self._workflow_app = None
        self._ensure_langgraph_available()

    def _ensure_langgraph_available(self):
        """Ensure LangGraph is available and create workflow."""
        try:
            from langgraph.graph import StateGraph, END
            from langgraph.checkpoint.memory import MemorySaver
            from .langgraph_workflow.workflow import create_evolution_workflow

            # Store workflow creation function
            self._create_workflow = create_evolution_workflow

        except ImportError as e:
            if "not subscriptable" in str(e):
                raise ImportError(
                    "LangGraph requires Python 3.9+ due to type hint compatibility. "
                    "Please upgrade to Python 3.9+ or use the local_python backend."
                ) from e
            else:
                raise ImportError(
                    "LangGraph is not available. Install with: pip install langgraph"
                ) from e

    def is_available(self) -> bool:
        """Check if LangGraph is available."""
        try:
            self._ensure_langgraph_available()
            return True
        except ImportError:
            return False

    def get_backend_name(self) -> str:
        """Get backend name."""
        return "langgraph"

    def run_stage(
        self,
        stage_config: Dict[str, Any],
        initial_codebase_code: str,
        evaluator_fn,
        evaluator_config: Dict[str, Any],
        initial_population: Optional[List[Program]] = None,
    ) -> StageOutput:
        """Run a single evolution stage using LangGraph workflow."""

        # Validate configuration
        self.validate_stage_config(stage_config)

        logger.info(
            f"Running stage '{stage_config['name']}' with {self.get_backend_name()} backend"
        )

        # Import LangGraph components
        from .langgraph_workflow.state import create_initial_state

        # Create workflow if not already created
        if self._workflow_app is None:
            self._workflow_app = self._create_workflow(evaluator_fn)

        # Create initial state for this stage
        initial_state = create_initial_state(
            stage_name=stage_config["name"],
            max_generations=stage_config["max_generations"],
            llm_settings=stage_config["llm_ensemble"],
            stage_config=stage_config,
            run_config=stage_config,  # Use stage config as run config
            api_key=self.api_key,
            initial_codebase_code=initial_codebase_code,
            task_description=stage_config.get("task_description", ""),
            initial_population=initial_population,
        )

        try:
            # Run the workflow
            config = {"configurable": {"thread_id": f"stage_{stage_config['name']}"}}

            # Execute the workflow
            final_state = None
            for state in self._workflow_app.stream(initial_state, config):
                final_state = state
                # Log progress
                if isinstance(state, dict) and len(state) == 1:
                    node_name = list(state.keys())[0]
                    node_state = list(state.values())[0]
                    gen = node_state.get("generation", 0)
                    best_score = "N/A"
                    if node_state.get("best_program"):
                        best_score = node_state["best_program"].scores.get(
                            "main_score", "N/A"
                        )
                    logger.info(
                        f"  Node: {node_name} | Gen: {gen} | Best: {best_score}"
                    )

            # Extract final state
            if isinstance(final_state, dict) and len(final_state) == 1:
                final_state = list(final_state.values())[0]

            # Create stage output
            status = "FAILED" if final_state.get("error_message") else "COMPLETED"
            message = final_state.get(
                "error_message",
                f"Stage '{stage_config['name']}' completed successfully",
            )

            stage_output = StageOutput(
                stage_name=stage_config["name"],
                status=status,
                message=message,
                best_program=final_state.get("best_program"),
                final_population=final_state.get("current_population", []),
                artifacts=final_state.get("artifacts", {}),
            )

            if status == "FAILED":
                logger.error(f"Stage failed: {message}")
            else:
                logger.info(f"Stage '{stage_config['name']}' completed successfully")

            return stage_output

        except Exception as e:
            logger.error(f"Error running stage '{stage_config['name']}': {e}")
            return StageOutput(
                stage_name=stage_config["name"],
                status="FAILED",
                message=f"Stage failed with exception: {e}",
                best_program=None,
                final_population=[],
                artifacts={},
            )
