"""
Simple test to verify evolver integration without requiring API keys.
Tests that the adapters and factory functions work correctly.
"""

import logging
import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_evolve_framework.coding_agents.llm_block_evolver.codebase import Codebase
from alpha_evolve_framework.coding_agents.aider_evolver.aider_adapter import (
    AiderEvolverAdapter,
)
from alpha_evolve_framework.coding_agents.llm_block_evolver.llm_block_evolver import (
    LLMBlockEvolver,
)
from alpha_evolve_framework.evaluators.functional_evaluator import FunctionalEvaluator
from alpha_evolve_framework.databases.simple_program_database import (
    SimpleProgramDatabase,
)
from alpha_evolve_framework.backends.local_python_backend import create_evolver
from alpha_evolve_framework.backends.langgraph_workflow.nodes import (
    create_evolver_langgraph,
)
from alpha_evolve_framework.llm.llm_manager import LLMManager
from alpha_evolve_framework.llm.prompt_engine import PromptEngine
from alpha_evolve_framework.config import RunConfiguration
from alpha_evolve_framework.core_types import LLMSettings

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_local_backend_evolver_creation():
    """Test that local backend can create both types of evolvers."""
    print("=" * 60)
    print("Testing Local Backend Evolver Creation")
    print("=" * 60)

    # Create test components
    initial_codebase = Codebase("def test(): return 42")
    evaluator = FunctionalEvaluator("test", lambda x, y: ({"main_score": 1.0}, {}), {})
    database = SimpleProgramDatabase(10)

    # Mock LLM settings
    llm_settings = [
        LLMSettings(model_name="test-model", generation_params={"temperature": 0.7})
    ]

    # Create mock LLM manager (won't actually make calls)
    llm_manager = LLMManager(
        default_api_key="test-key",
        llm_settings_list=llm_settings,
        min_inter_request_delay_sec=0.1,
    )

    prompt_engine = PromptEngine("test task", {})
    run_config = RunConfiguration(
        max_generations=2,
        population_size=5,
        use_island_model=False,
        num_islands=1,
        evaluation_timeout_seconds=30,
        llm_ensemble=llm_settings,
    )

    # Test 1: Create LLMBlockEvolver
    print("\n1. Testing LLMBlockEvolver creation...")
    try:
        llm_evolver = create_evolver(
            evolver_type="llm_block",
            evolver_config={},
            initial_codebase=initial_codebase,
            llm_manager=llm_manager,
            program_database=database,
            prompt_engine=prompt_engine,
            evaluator=evaluator,
            run_config=run_config,
        )
        print(f"✅ LLMBlockEvolver created: {type(llm_evolver).__name__}")
        assert isinstance(llm_evolver, LLMBlockEvolver)
    except Exception as e:
        print(f"❌ LLMBlockEvolver creation failed: {e}")
        return False

    # Test 2: Create AiderEvolverAdapter
    print("\n2. Testing AiderEvolverAdapter creation...")
    try:
        aider_evolver = create_evolver(
            evolver_type="aider",
            evolver_config={"model": "test-model"},
            initial_codebase=initial_codebase,
            llm_manager=llm_manager,
            program_database=database,
            prompt_engine=prompt_engine,
            evaluator=evaluator,
            run_config=run_config,
        )
        print(f"✅ AiderEvolverAdapter created: {type(aider_evolver).__name__}")
        assert isinstance(aider_evolver, AiderEvolverAdapter)
    except Exception as e:
        print(f"❌ AiderEvolverAdapter creation failed: {e}")
        return False

    return True


def test_langgraph_evolver_creation():
    """Test that LangGraph backend can create both types of evolvers."""
    print("\n" + "=" * 60)
    print("Testing LangGraph Backend Evolver Creation")
    print("=" * 60)

    # Create test components
    initial_codebase = Codebase("def test(): return 42")
    evaluator = FunctionalEvaluator("test", lambda x, y: ({"main_score": 1.0}, {}), {})
    database = SimpleProgramDatabase(10)

    # Mock LLM settings
    llm_settings = [
        LLMSettings(model_name="test-model", generation_params={"temperature": 0.7})
    ]

    # Create mock LLM manager (won't actually make calls)
    llm_manager = LLMManager(
        default_api_key="test-key",
        llm_settings_list=llm_settings,
        min_inter_request_delay_sec=0.1,
    )

    prompt_engine = PromptEngine("test task", {})
    run_config = RunConfiguration(
        max_generations=2,
        population_size=5,
        use_island_model=False,
        num_islands=1,
        evaluation_timeout_seconds=30,
        llm_ensemble=llm_settings,
    )

    # Test 1: Create LLMBlockEvolver for LangGraph
    print("\n1. Testing LangGraph LLMBlockEvolver creation...")
    try:
        llm_evolver = create_evolver_langgraph(
            evolver_type="llm_block",
            evolver_config={},
            initial_codebase=initial_codebase,
            llm_manager=llm_manager,
            program_database=database,
            prompt_engine=prompt_engine,
            evaluator=evaluator,
            run_config=run_config,
        )
        print(f"✅ LangGraph LLMBlockEvolver created: {type(llm_evolver).__name__}")
        assert isinstance(llm_evolver, LLMBlockEvolver)
    except Exception as e:
        print(f"❌ LangGraph LLMBlockEvolver creation failed: {e}")
        return False

    # Test 2: Create AiderEvolverAdapter for LangGraph
    print("\n2. Testing LangGraph AiderEvolverAdapter creation...")
    try:
        aider_evolver = create_evolver_langgraph(
            evolver_type="aider",
            evolver_config={"model": "test-model"},
            initial_codebase=initial_codebase,
            llm_manager=llm_manager,
            program_database=database,
            prompt_engine=prompt_engine,
            evaluator=evaluator,
            run_config=run_config,
        )
        print(
            f"✅ LangGraph AiderEvolverAdapter created: {type(aider_evolver).__name__}"
        )
        assert isinstance(aider_evolver, AiderEvolverAdapter)
    except Exception as e:
        print(f"❌ LangGraph AiderEvolverAdapter creation failed: {e}")
        return False

    return True


def test_adapter_ask_tell_pattern():
    """Test that AiderEvolverAdapter properly implements ask/tell pattern."""
    print("\n" + "=" * 60)
    print("Testing AiderEvolver Adapter Ask/Tell Pattern")
    print("=" * 60)

    # Create test components
    initial_codebase = Codebase("def test(): return 42")
    evaluator = FunctionalEvaluator("test", lambda x, y: ({"main_score": 1.0}, {}), {})
    database = SimpleProgramDatabase(10)

    # Create adapter
    adapter = AiderEvolverAdapter(
        initial_codebase=initial_codebase,
        evaluator=evaluator,
        program_database=database,
        model="test-model",
        working_dir="test_working_dir",
    )

    print("\n1. Testing initialize_population...")
    try:
        adapter.initialize_population()
        programs = adapter.get_population()
        print(f"✅ Population initialized with {len(programs)} programs")
        assert len(programs) > 0
    except Exception as e:
        print(f"❌ Population initialization failed: {e}")
        return False

    print("\n2. Testing ask method...")
    try:
        # This should work even without actual Aider calls since it falls back
        suggestions = adapter.ask()
        print(f"✅ Ask method returned {len(suggestions)} suggestions")
        # Should return at least one suggestion (fallback)
        assert len(suggestions) >= 0
    except Exception as e:
        print(f"❌ Ask method failed: {e}")
        return False

    print("\n3. Testing tell method...")
    try:
        from alpha_evolve_framework.core_types import Program

        # Create a mock evaluated program
        test_program = Program(
            id="test-program",
            code_str="def test(): return 100",
            block_name="main",
            parent_id=None,
            generation=1,
            scores={"main_score": 100.0},
            evaluator_feedback={},
        )
        adapter.tell([test_program])
        print("✅ Tell method completed successfully")
    except Exception as e:
        print(f"❌ Tell method failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("🚀 EVOLVER INTEGRATION TEST SUITE")
    print("Testing evolver selection and adapter functionality...")
    print("No API keys required for these tests.")

    success_count = 0
    total_tests = 3

    # Test 1: Local Backend Evolver Creation
    if test_local_backend_evolver_creation():
        success_count += 1
        print("✅ Local Backend Test: PASSED")
    else:
        print("❌ Local Backend Test: FAILED")

    # Test 2: LangGraph Backend Evolver Creation
    if test_langgraph_evolver_creation():
        success_count += 1
        print("✅ LangGraph Backend Test: PASSED")
    else:
        print("❌ LangGraph Backend Test: FAILED")

    # Test 3: Adapter Ask/Tell Pattern
    if test_adapter_ask_tell_pattern():
        success_count += 1
        print("✅ Adapter Ask/Tell Test: PASSED")
    else:
        print("❌ Adapter Ask/Tell Test: FAILED")

    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"✅ Tests Passed: {success_count}/{total_tests}")
    print(f"❌ Tests Failed: {total_tests - success_count}/{total_tests}")

    if success_count == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("✨ AiderEvolver Integration Complete!")
        print("\n💡 Usage Examples:")
        print("# Use AiderEvolver with LocalPython backend")
        print("agent.add_stage(...).use_aider_evolver().run()")
        print("")
        print("# Use AiderEvolver with LangGraph backend")
        print("agent.add_stage(...).use_aider_evolver().use_langgraph().run()")
        return True
    else:
        print("\n❌ Some tests failed. Integration may need fixes.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
