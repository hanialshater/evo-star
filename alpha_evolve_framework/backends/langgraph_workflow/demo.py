"""
Demo for LangGraph backend - Simple mathematical function optimization.
"""

import logging
from typing import Dict, Any

from .workflow import create_evolution_workflow
from ..core_types import LLMSettings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Simple demo problem: optimize a mathematical function
def get_initial_code() -> str:
    """Return initial code for optimization."""
    return '''
def target_function(x):
    """Function to optimize - we want to maximize this."""
    # Initial simple function
    return x * 2

def main():
    result = target_function(5)
    return result
'''


def simple_evaluator(
    program_id: str, full_code: str, generation: int, **kwargs
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """
    Simple evaluator that runs the code and scores it.
    We want to find a function that returns a high value.
    """
    try:
        # Create a local namespace for execution
        namespace = {}

        # Execute the code
        exec(full_code, namespace)

        # Call the main function
        if "main" in namespace:
            result = namespace["main"]()
            score = float(result) if isinstance(result, (int, float)) else 0.0
        else:
            score = -1000.0  # Penalty for missing main function

        return (
            {"main_score": score},
            {
                "result": result if "result" in locals() else None,
                "execution": "success",
            },
        )

    except Exception as e:
        # Penalty for code that doesn't run
        return ({"main_score": -1000.0}, {"error": str(e), "execution": "failed"})


def run_demo(api_key: str) -> None:
    """Run a simple demonstration using LangGraph backend."""
    logger.info("=== Starting LangGraph Demo ===")

    from ..backends import LangGraphBackend

    try:
        # Create LangGraph backend
        backend = LangGraphBackend(api_key)

        # Create stage configuration
        stage_config = {
            "name": "demo_stage",
            "max_generations": 3,
            "llm_ensemble": [
                LLMSettings(
                    model_name="gemini/gemini-1.5-flash",
                    generation_params={"temperature": 0.8, "max_tokens": 1500},
                )
            ],
            "task_description": "Optimize the target_function to return the highest possible value",
            "population_size": 5,
            "evaluation_timeout_seconds": 10,
        }

        # Run evolution
        result = backend.run_stage(
            stage_config=stage_config,
            initial_codebase_code=get_initial_code(),
            evaluator_fn=simple_evaluator,
            evaluator_config={},
        )

        logger.info("=== Demo Results ===")
        logger.info(f"Status: {result.status}")
        logger.info(f"Message: {result.message}")

        if result.best_program:
            best = result.best_program
            logger.info(f"Best Program ID: {best.id}")
            logger.info(f"Best Score: {best.scores.get('main_score', 'N/A')}")
            logger.info(f"Best Code:\n{best.code_str}")
        else:
            logger.warning("No best program found!")

        logger.info(f"Final Population Size: {len(result.final_population)}")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    # Example usage
    from ...utils import get_gemini_api_key

    try:
        API_KEY = get_gemini_api_key()
        run_demo(API_KEY)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your GEMINI_API_KEY in the .env file")
