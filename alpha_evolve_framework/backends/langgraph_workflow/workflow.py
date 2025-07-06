"""
LangGraph workflow for evolutionary algorithms.
"""

from typing import Dict, Any, List
import logging

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    # Graceful degradation if LangGraph not installed
    StateGraph = None
    END = None
    MemorySaver = None

from .state import EvolutionState
from .nodes import EvolutionNodes
from ...core_types import LLMSettings, StageOutput

logger = logging.getLogger(__name__)


def create_evolution_workflow(evaluator_fn=None):
    """Create a LangGraph workflow for evolution."""

    if StateGraph is None:
        raise ImportError("LangGraph not installed. Run: pip install langgraph")

    # Create the state graph
    workflow = StateGraph(EvolutionState)

    # Create nodes with evaluator function access
    nodes = EvolutionNodes(evaluator_fn)

    # Add nodes (simplified workflow matching LocalPython backend approach)
    workflow.add_node("initialize_population", nodes.initialize_population)
    workflow.add_node(
        "run_generation", nodes.generate_candidates
    )  # Does everything in one step
    workflow.add_node("check_termination", nodes.check_termination)

    # Set entry point
    workflow.set_entry_point("initialize_population")

    # Define the flow
    workflow.add_edge("initialize_population", "run_generation")
    workflow.add_edge("run_generation", "check_termination")

    # Conditional edge: continue or end based on termination condition
    workflow.add_conditional_edges(
        "check_termination",
        _should_continue,
        {"continue": "run_generation", "end": END},
    )

    # Add memory for checkpointing
    memory = MemorySaver()

    # Compile the workflow
    app = workflow.compile(checkpointer=memory)

    return app


def _should_continue(state: EvolutionState) -> str:
    """Determine if evolution should continue or end."""
    if state.get("should_terminate", False):
        return "end"
    return "continue"
