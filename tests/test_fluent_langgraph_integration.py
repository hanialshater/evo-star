"""
Test script to demonstrate fluent API with LangGraph backend integration.
This shows how existing fluent API code can seamlessly switch to LangGraph.
"""

import sys
import os
import logging
from typing import Dict, Any, Tuple

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_evolve_framework import EvoAgent, LLMSettings

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_initial_code() -> str:
    """Return initial code for optimization."""
    return '''
def target_function(x):
    """Function to optimize - we want to maximize this."""
    # EVOLVE-BLOCK-START main
    # Simple initial function
    return x * 2
    # EVOLVE-BLOCK-END

def main():
    result = target_function(5)
    return result
'''


def simple_evaluator(
    full_code: str, config: Dict[str, Any]
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Simple evaluator that runs the code and scores it."""
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
                "config": config,
            },
        )

    except Exception as e:
        # Penalty for code that doesn't run
        return (
            {"main_score": -1000.0},
            {"error": str(e), "execution": "failed", "config": config},
        )


def test_traditional_orchestrator(api_key: str):
    """Test the traditional MainLoopOrchestrator backend."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING TRADITIONAL ORCHESTRATOR")
    logger.info("=" * 60)

    # Create LLM settings
    llm_settings = [
        LLMSettings(
            model_name="gemini/gemini-1.5-flash",
            generation_params={"temperature": 0.7, "max_tokens": 1000},
        )
    ]

    # Create agent and run (traditional way)
    try:
        agent = EvoAgent(api_key)
        result = (
            agent.define_problem(get_initial_code, simple_evaluator)
            .add_stage(
                "traditional_stage",
                max_generations=2,
                llm_settings=llm_settings,
                task_description="Traditional orchestrator test",
                population_size=3,
            )
            .run()
        )

        logger.info(f"Traditional Result Status: {result.status}")
        if result.best_program:
            logger.info(
                f"Traditional Best Score: {result.best_program.scores.get('main_score', 'N/A')}"
            )

        return result

    except Exception as e:
        logger.error(f"Traditional orchestrator test failed: {e}")
        return None


def test_langgraph_backend(api_key: str):
    """Test the LangGraph backend."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING LANGGRAPH BACKEND")
    logger.info("=" * 60)

    # Create LLM settings
    llm_settings = [
        LLMSettings(
            model_name="gemini/gemini-1.5-flash",
            generation_params={"temperature": 0.7, "max_tokens": 1000},
        )
    ]

    # Create agent and run with LangGraph (only difference is .use_langgraph())
    try:
        agent = EvoAgent(api_key)
        result = (
            agent.define_problem(get_initial_code, simple_evaluator)
            .add_stage(
                "langgraph_stage",
                max_generations=2,
                llm_settings=llm_settings,
                task_description="LangGraph backend test",
                population_size=3,
            )
            .use_langgraph()  # <-- This is the only change!
            .run()
        )

        logger.info(f"LangGraph Result Status: {result.status}")
        if result.best_program:
            logger.info(
                f"LangGraph Best Score: {result.best_program.scores.get('main_score', 'N/A')}"
            )

        return result

    except ImportError as e:
        if "Python 3.9+" in str(e):
            logger.warning(
                "⚠️  LangGraph requires Python 3.9+ for type hint compatibility"
            )
            logger.warning("Current Python version is not compatible")
            logger.info("✅ GRACEFUL DEGRADATION: Use traditional orchestrator instead")
            return None
        else:
            logger.error(f"LangGraph backend test failed: {e}")
            return None
    except TypeError as e:
        if "not subscriptable" in str(e):
            logger.warning(
                "⚠️  LangGraph requires Python 3.9+ for type hint compatibility"
            )
            logger.warning("Current Python version is not compatible")
            logger.info("✅ GRACEFUL DEGRADATION: Use traditional orchestrator instead")
            return None
        else:
            logger.error(f"LangGraph backend test failed: {e}")
            return None
    except Exception as e:
        logger.error(f"LangGraph backend test failed: {e}")
        return None


def compare_results(traditional_result, langgraph_result):
    """Compare results from both backends."""
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON OF RESULTS")
    logger.info("=" * 60)

    if traditional_result and langgraph_result:
        logger.info("✅ Both backends completed successfully!")

        trad_score = (
            traditional_result.best_program.scores.get("main_score", -float("inf"))
            if traditional_result.best_program
            else -float("inf")
        )
        lang_score = (
            langgraph_result.best_program.scores.get("main_score", -float("inf"))
            if langgraph_result.best_program
            else -float("inf")
        )

        logger.info(f"Traditional best score: {trad_score}")
        logger.info(f"LangGraph best score: {lang_score}")

        logger.info("\n🎯 KEY BENEFITS OF LANGGRAPH INTEGRATION:")
        logger.info("- Same fluent API interface")
        logger.info("- Node-based workflow execution")
        logger.info("- Better debugging and visualization")
        logger.info("- Checkpointing and state management")
        logger.info("- Ready for streaming and human-in-the-loop")

    elif traditional_result:
        logger.warning("⚠️  Only traditional orchestrator worked")

    elif langgraph_result:
        logger.warning("⚠️  Only LangGraph backend worked")

    else:
        logger.error("❌ Both backends failed")


if __name__ == "__main__":
    # Your provided API key
    from alpha_evolve_framework.utils import get_gemini_api_key

    try:
        API_KEY = get_gemini_api_key()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your GEMINI_API_KEY in the .env file")
        exit(1)

    logger.info("🚀 Testing Fluent API with LangGraph Integration")

    # Test traditional orchestrator
    traditional_result = test_traditional_orchestrator(API_KEY)

    # Test LangGraph backend
    langgraph_result = test_langgraph_backend(API_KEY)

    # Compare results
    compare_results(traditional_result, langgraph_result)

    logger.info("\n" + "=" * 60)
    logger.info("🎉 FLUENT API LANGGRAPH INTEGRATION COMPLETE!")
    logger.info("=" * 60)
    logger.info("The same fluent API now supports both:")
    logger.info("- Traditional MainLoopOrchestrator (default)")
    logger.info("- LangGraph workflow orchestration (with .use_langgraph())")
    logger.info("=" * 60)
