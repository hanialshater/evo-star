"""
Test script to verify the new backend architecture works correctly.
"""

import logging
from alpha_evolve_framework import EvoAgent, LLMSettings

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_initial_code():
    """Return initial code for optimization."""
    return '''
def target_function(x):
    """Function to optimize - we want to maximize this."""
    # EVOLVE-BLOCK-START main
    return x * 2
    # EVOLVE-BLOCK-END

def main():
    result = target_function(5)
    return result
'''


def simple_evaluator(program_id, full_code, generation, **kwargs):
    """Simple evaluator that runs code and returns score."""
    try:
        namespace = {}
        exec(full_code, namespace)

        if "main" in namespace:
            result = namespace["main"]()
            score = float(result) if isinstance(result, (int, float)) else 0.0
        else:
            score = -1000.0

        return (
            {"main_score": score},
            {"result": result if "result" in locals() else None},
        )
    except Exception as e:
        return ({"main_score": -1000.0}, {"error": str(e)})


def test_local_python_backend():
    """Test the LocalPython backend."""
    logger.info("=== Testing LocalPython Backend ===")

    try:
        from alpha_evolve_framework.utils import get_gemini_api_key

        try:
            api_key = get_gemini_api_key()
        except ValueError as e:
            print(f"Error: {e}")
            print("Please set your GEMINI_API_KEY in the .env file")
            return "FAILED"

        agent = EvoAgent(api_key=api_key, log_level=logging.DEBUG)

        result = (
            agent.define_problem(
                initial_code_fn=get_initial_code, evaluator_fn=simple_evaluator
            )
            .add_stage(
                name="test_stage",
                max_generations=2,
                llm_settings=[LLMSettings(model_name="gemini/gemini-1.5-flash")],
                population_size=3,
                task_description="Optimize target_function to return highest value",
            )
            .run()
        )

        logger.info(f"LocalPython Result: {result.status}")
        if result.best_program:
            logger.info(
                f"Best Score: {result.best_program.scores.get('main_score', 'N/A')}"
            )

        return result

    except Exception as e:
        logger.error(f"LocalPython backend test failed: {e}")
        return None


def test_langgraph_backend():
    """Test the LangGraph backend."""
    logger.info("=== Testing LangGraph Backend ===")

    try:
        from alpha_evolve_framework.utils import get_gemini_api_key

        try:
            api_key = get_gemini_api_key()
        except ValueError as e:
            print(f"Error: {e}")
            print("Please set your GEMINI_API_KEY in the .env file")
            return "FAILED"

        agent = EvoAgent(api_key=api_key, log_level=logging.DEBUG)

        result = (
            agent.define_problem(
                initial_code_fn=get_initial_code, evaluator_fn=simple_evaluator
            )
            .add_stage(
                name="test_stage",
                max_generations=2,
                llm_settings=[LLMSettings(model_name="gemini/gemini-1.5-flash")],
                population_size=3,
                task_description="Optimize target_function to return highest value",
            )
            .use_langgraph()
            .run()
        )

        logger.info(f"LangGraph Result: {result.status}")
        if result.best_program:
            logger.info(
                f"Best Score: {result.best_program.scores.get('main_score', 'N/A')}"
            )

        return result

    except Exception as e:
        logger.error(f"LangGraph backend test failed: {e}")
        return None


def main():
    """Run all tests."""
    logger.info("🚀 Testing New Backend Architecture")
    logger.info("=" * 60)

    # Test both backends
    local_result = test_local_python_backend()
    langgraph_result = test_langgraph_backend()

    # Compare results
    logger.info("=" * 60)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 60)

    local_status = local_result.status if local_result else "FAILED"
    langgraph_status = langgraph_result.status if langgraph_result else "FAILED"

    logger.info(f"LocalPython Backend: {local_status}")
    logger.info(f"LangGraph Backend: {langgraph_status}")

    if local_result and langgraph_result:
        local_score = (
            local_result.best_program.scores.get("main_score", "N/A")
            if local_result.best_program
            else "N/A"
        )
        langgraph_score = (
            langgraph_result.best_program.scores.get("main_score", "N/A")
            if langgraph_result.best_program
            else "N/A"
        )

        logger.info(f"LocalPython Best Score: {local_score}")
        logger.info(f"LangGraph Best Score: {langgraph_score}")

        if local_status == "COMPLETED" and langgraph_status == "COMPLETED":
            logger.info("🎉 Both backends working correctly!")

    logger.info("=" * 60)
    logger.info("✅ Backend Architecture Test Complete")


if __name__ == "__main__":
    main()
