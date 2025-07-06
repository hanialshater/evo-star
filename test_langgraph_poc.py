"""
Standalone test script for LangGraph POC.
"""

import sys
import os
import logging
from typing import Dict, Any, Tuple

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_evolve_framework.langgraph_backend.poc_workflow import LangGraphEvoAgent, run_simple_poc_evolution
from alpha_evolve_framework.core_types import LLMSettings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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


def simple_evaluator(program_id: str, full_code: str, generation: int, **kwargs) -> Tuple[Dict[str, float], Dict[str, Any]]:
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
        if 'main' in namespace:
            result = namespace['main']()
            score = float(result) if isinstance(result, (int, float)) else 0.0
        else:
            score = -1000.0  # Penalty for missing main function
        
        return (
            {"main_score": score},
            {"result": result if 'result' in locals() else None, "execution": "success"}
        )
        
    except Exception as e:
        # Penalty for code that doesn't run
        return (
            {"main_score": -1000.0},
            {"error": str(e), "execution": "failed"}
        )


def run_poc_demo(api_key: str) -> None:
    """Run a simple POC demonstration."""
    logger.info("=== Starting LangGraph POC Demo ===")
    
    try:
        # Run simple evolution
        result = run_simple_poc_evolution(
            initial_code_fn=get_initial_code,
            evaluator_fn=simple_evaluator,
            api_key=api_key,
            max_generations=2,  # Keep it small for POC
            llm_model="gemini/gemini-1.5-flash"
        )
        
        logger.info("=== POC Demo Results ===")
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
        
        return result
        
    except Exception as e:
        logger.error(f"POC Demo failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # Your provided API key
    API_KEY = "AIzaSyBdiqeV2FSL63Y_rlazDyQEpORimWTt5-M"
    
    logger.info("Testing LangGraph POC...")
    result = run_poc_demo(API_KEY)
    
    if result:
        logger.info("POC Demo completed successfully!")
    else:
        logger.error("POC Demo failed!")
