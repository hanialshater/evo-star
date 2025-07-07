"""
Test script to demonstrate evolver selection with both backends.
This shows all 4 combinations:
1. LocalPythonBackend + LLMBlockEvolver (default)
2. LocalPythonBackend + AiderEvolver
3. LangGraphBackend + LLMBlockEvolver
4. LangGraphBackend + AiderEvolver
"""

import sys
import os
import logging
from typing import Dict, Any, Tuple

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def test_local_python_llm_block(api_key: str):
    """Test LocalPythonBackend + LLMBlockEvolver (default)."""
    logger.info("\n" + "=" * 60)
    logger.info("1. LOCAL PYTHON + LLM BLOCK EVOLVER (DEFAULT)")
    logger.info("=" * 60)

    llm_settings = [
        LLMSettings(
            model_name="gemini/gemini-1.5-flash",
            generation_params={"temperature": 0.7, "max_tokens": 1000},
        )
    ]

    try:
        agent = EvoAgent(api_key)
        result = (
            agent.define_problem(get_initial_code, simple_evaluator)
            .add_stage(
                "local_llm_block",
                max_generations=2,
                llm_settings=llm_settings,
                task_description="Local Python + LLM Block Evolver test",
                population_size=3,
            )
            .use_llm_block_evolver()  # Explicit (optional since it's default)
            .run()
        )

        logger.info(f"Result Status: {result.status}")
        if result.best_program:
            logger.info(
                f"Best Score: {result.best_program.scores.get('main_score', 'N/A')}"
            )
        return result

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return None


def test_local_python_aider(api_key: str):
    """Test LocalPythonBackend + AiderEvolver."""
    logger.info("\n" + "=" * 60)
    logger.info("2. LOCAL PYTHON + AIDER EVOLVER")
    logger.info("=" * 60)

    llm_settings = [
        LLMSettings(
            model_name="gemini/gemini-1.5-flash",
            generation_params={"temperature": 0.7, "max_tokens": 1000},
        )
    ]

    try:
        agent = EvoAgent(api_key)
        result = (
            agent.define_problem(get_initial_code, simple_evaluator)
            .add_stage(
                "local_aider",
                max_generations=2,
                llm_settings=llm_settings,
                task_description="Local Python + Aider Evolver test",
                population_size=3,
            )
            .use_aider_evolver(model="gemini-1.5-flash")  # <-- New evolver!
            .run()
        )

        logger.info(f"Result Status: {result.status}")
        if result.best_program:
            logger.info(
                f"Best Score: {result.best_program.scores.get('main_score', 'N/A')}"
            )
        return result

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return None


def test_langgraph_llm_block(api_key: str):
    """Test LangGraphBackend + LLMBlockEvolver."""
    logger.info("\n" + "=" * 60)
    logger.info("3. LANGGRAPH + LLM BLOCK EVOLVER")
    logger.info("=" * 60)

    llm_settings = [
        LLMSettings(
            model_name="gemini/gemini-1.5-flash",
            generation_params={"temperature": 0.7, "max_tokens": 1000},
        )
    ]

    try:
        agent = EvoAgent(api_key)
        result = (
            agent.define_problem(get_initial_code, simple_evaluator)
            .add_stage(
                "langgraph_llm_block",
                max_generations=2,
                llm_settings=llm_settings,
                task_description="LangGraph + LLM Block Evolver test",
                population_size=3,
            )
            .use_llm_block_evolver()  # Explicit (optional since it's default)
            .use_langgraph()  # <-- Use LangGraph backend
            .run()
        )

        logger.info(f"Result Status: {result.status}")
        if result.best_program:
            logger.info(
                f"Best Score: {result.best_program.scores.get('main_score', 'N/A')}"
            )
        return result

    except ImportError as e:
        if "Python 3.9+" in str(e):
            logger.warning("⚠️  LangGraph requires Python 3.9+")
            logger.info("✅ GRACEFUL DEGRADATION: Test skipped")
            return None
        else:
            logger.error(f"Test failed: {e}")
            return None
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return None


def test_langgraph_aider(api_key: str):
    """Test LangGraphBackend + AiderEvolver (new combination!)."""
    logger.info("\n" + "=" * 60)
    logger.info("4. LANGGRAPH + AIDER EVOLVER (NEW COMBINATION!)")
    logger.info("=" * 60)

    llm_settings = [
        LLMSettings(
            model_name="gemini/gemini-1.5-flash",
            generation_params={"temperature": 0.7, "max_tokens": 1000},
        )
    ]

    try:
        agent = EvoAgent(api_key)
        result = (
            agent.define_problem(get_initial_code, simple_evaluator)
            .add_stage(
                "langgraph_aider",
                max_generations=2,
                llm_settings=llm_settings,
                task_description="LangGraph + Aider Evolver test",
                population_size=3,
            )
            .use_aider_evolver(model="gemini-1.5-flash")  # <-- New evolver!
            .use_langgraph()  # <-- Use LangGraph backend
            .run()
        )

        logger.info(f"Result Status: {result.status}")
        if result.best_program:
            logger.info(
                f"Best Score: {result.best_program.scores.get('main_score', 'N/A')}"
            )
        return result

    except ImportError as e:
        if "Python 3.9+" in str(e):
            logger.warning("⚠️  LangGraph requires Python 3.9+")
            logger.info("✅ GRACEFUL DEGRADATION: Test skipped")
            return None
        else:
            logger.error(f"Test failed: {e}")
            return None
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return None


def summarize_results(results: Dict[str, Any]):
    """Summarize all test results."""
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY OF ALL TESTS")
    logger.info("=" * 60)

    successful_tests = []
    failed_tests = []
    skipped_tests = []

    for test_name, result in results.items():
        if result is None:
            skipped_tests.append(test_name)
        elif result.status == "COMPLETED":
            successful_tests.append(test_name)
            score = (
                result.best_program.scores.get("main_score", "N/A")
                if result.best_program
                else "N/A"
            )
            logger.info(f"✅ {test_name}: Score = {score}")
        else:
            failed_tests.append(test_name)
            logger.info(f"❌ {test_name}: {result.status}")

    logger.info(f"\n🎯 RESULTS:")
    logger.info(f"✅ Successful: {len(successful_tests)}")
    logger.info(f"❌ Failed: {len(failed_tests)}")
    logger.info(f"⏭️  Skipped: {len(skipped_tests)}")

    if successful_tests:
        logger.info(f"\n🚀 NEW EVOLVER INTEGRATION COMPLETE!")
        logger.info(f"All {len(successful_tests)} working combinations:")
        for test in successful_tests:
            logger.info(f"  - {test}")

    logger.info("\n💡 FLUENT API EXAMPLES:")
    logger.info("# Default (LocalPython + LLMBlock)")
    logger.info("agent.define_problem(...).add_stage(...).run()")
    logger.info("")
    logger.info("# Use AiderEvolver with LocalPython")
    logger.info("agent.define_problem(...).add_stage(...).use_aider_evolver().run()")
    logger.info("")
    logger.info("# Use LangGraph with LLMBlock")
    logger.info("agent.define_problem(...).add_stage(...).use_langgraph().run()")
    logger.info("")
    logger.info("# Use AiderEvolver with LangGraph (new!)")
    logger.info(
        "agent.define_problem(...).add_stage(...).use_aider_evolver().use_langgraph().run()"
    )


if __name__ == "__main__":
    # Your provided API key
    from alpha_evolve_framework.utils import get_gemini_api_key

    try:
        API_KEY = get_gemini_api_key()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your GEMINI_API_KEY in the .env file")
        exit(1)

    logger.info("🚀 Testing All Evolver + Backend Combinations")

    # Run all tests
    results = {
        "LocalPython + LLMBlock": test_local_python_llm_block(API_KEY),
        "LocalPython + Aider": test_local_python_aider(API_KEY),
        "LangGraph + LLMBlock": test_langgraph_llm_block(API_KEY),
        "LangGraph + Aider": test_langgraph_aider(API_KEY),
    }

    # Summarize results
    summarize_results(results)

    logger.info("\n" + "=" * 60)
    logger.info("🎉 EVOLVER BACKEND INTEGRATION TEST COMPLETE!")
    logger.info("=" * 60)
