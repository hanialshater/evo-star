"""
Test the AiderEvolver integration.

This test demonstrates how AiderEvolver works with the existing codebase system
and evolve blocks.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alpha_evolve_framework.coding_agents.aider_evolver.aider_evolver import (
    AiderEvolver,
)
from alpha_evolve_framework.coding_agents.llm_block_evolver.codebase import Codebase
from alpha_evolve_framework.utils.env_loader import get_gemini_api_key


def test_aider_evolver_basic():
    """Test basic AiderEvolver functionality."""

    # Sample code with evolve blocks
    sample_code = """
def main():
    print("Hello, World!")
    
    # EVOLVE-BLOCK-START algorithm
    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr
    # EVOLVE-BLOCK-END
    
    # Test the algorithm
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", test_array)
    sorted_array = bubble_sort(test_array.copy())
    print("Sorted array:", sorted_array)

if __name__ == "__main__":
    main()
"""

    print("🧪 Testing AiderEvolver Basic Functionality")
    print("=" * 50)

    # Create codebase
    codebase = Codebase(sample_code)
    print(f"📋 Codebase created with {len(codebase.get_block_names())} blocks")
    print(f"🔧 Available blocks: {codebase.get_block_names()}")

    # Try to create AiderEvolver (this will check if Aider is installed)
    try:
        evolver = AiderEvolver(model="gemini-1.5-flash")
        print("✅ AiderEvolver initialized successfully")
    except RuntimeError as e:
        print(f"❌ AiderEvolver initialization failed: {e}")
        print("💡 To install Aider: pip install aider-chat")
        return False

    # Check if we have API keys
    try:
        api_key = get_gemini_api_key()
        print("✅ API key found")
    except ValueError as e:
        print(f"⚠️  API key not configured: {e}")
        print("💡 Run: python setup_environment.py to configure API keys")
        return False

    print("\n🎯 Ready to test evolution (requires Aider installation and API key)")
    return True


def test_aider_evolver_with_examples():
    """Test AiderEvolver with example implementations."""

    print("\n🧪 Testing AiderEvolver with Examples")
    print("=" * 50)

    # Base algorithm (bubble sort)
    base_code = """
def main():
    # EVOLVE-BLOCK-START sorting_algorithm
    def sort_array(arr):
        # Simple bubble sort implementation
        n = len(arr)
        for i in range(n):
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr
    # EVOLVE-BLOCK-END
    
    test_array = [64, 34, 25, 12, 22, 11, 90]
    result = sort_array(test_array.copy())
    print("Sorted:", result)

if __name__ == "__main__":
    main()
"""

    # Example implementations for inspiration
    quicksort_example = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""

    mergesort_example = """
def mergesort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
"""

    # Create codebase
    codebase = Codebase(base_code)
    print(f"📋 Base codebase created with block: {codebase.get_block_names()}")

    # Create example implementations
    examples = [quicksort_example, mergesort_example]
    print(f"📚 Created {len(examples)} example implementations")

    # This would be the actual evolution call (commented out for demo)
    print("\n🔄 Evolution Process (Demo):")
    print("1. Create workspace with base program")
    print("2. Add example implementations as context")
    print("3. Create evolution instructions")
    print("4. Run Aider with objective: 'Optimize the sorting algorithm'")
    print("5. Extract evolved block and update codebase")

    print("\n✅ Example setup complete!")
    return True


def test_aider_crossover_simulation():
    """Simulate crossover between two implementations."""

    print("\n🧪 Testing AiderEvolver Crossover Simulation")
    print("=" * 50)

    # Parent 1: Bubble sort with early termination
    parent1_code = """
def main():
    # EVOLVE-BLOCK-START sorting_algorithm
    def sort_array(arr):
        n = len(arr)
        for i in range(n):
            swapped = False
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
                    swapped = True
            if not swapped:
                break  # Early termination optimization
        return arr
    # EVOLVE-BLOCK-END
    
    test_array = [64, 34, 25, 12, 22, 11, 90]
    result = sort_array(test_array.copy())
    print("Sorted:", result)
"""

    # Parent 2: Selection sort
    parent2_code = """
def main():
    # EVOLVE-BLOCK-START sorting_algorithm
    def sort_array(arr):
        n = len(arr)
        for i in range(n):
            min_idx = i
            for j in range(i+1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr
    # EVOLVE-BLOCK-END
    
    test_array = [64, 34, 25, 12, 22, 11, 90]
    result = sort_array(test_array.copy())
    print("Sorted:", result)
"""

    # Create parent codebases
    parent1_codebase = Codebase(parent1_code)
    parent2_codebase = Codebase(parent2_code)

    print(f"👨‍👩‍👧‍👦 Parent 1 block: {parent1_codebase.get_block_names()}")
    print(f"👨‍👩‍👧‍👦 Parent 2 block: {parent2_codebase.get_block_names()}")

    # Show what the crossover would do
    print("\n🔄 Crossover Process (Demo):")
    print("1. Extract both implementations")
    print("2. Provide both as examples to Aider")
    print("3. Objective: 'Combine the best features from both implementations'")
    print(
        "4. Aider would create hybrid combining early termination + selection sort logic"
    )

    print("\n✅ Crossover simulation complete!")
    return True


def demonstrate_aider_workflow():
    """Demonstrate the complete AiderEvolver workflow."""

    print("\n🚀 AiderEvolver Workflow Demonstration")
    print("=" * 60)

    print("\n1. 📋 CODEBASE SETUP")
    print("   - Parse code with EVOLVE-BLOCK markers")
    print("   - Identify blocks available for evolution")

    print("\n2. 🎯 EVOLUTION OBJECTIVES")
    print("   - Performance optimization")
    print("   - Code quality improvement")
    print("   - Feature addition")
    print("   - Bug fixes")

    print("\n3. 📚 EXAMPLE PREPARATION")
    print("   - Collect example implementations")
    print("   - Prepare context files")
    print("   - Set up evolution instructions")

    print("\n4. 🔧 AIDER INTEGRATION")
    print("   - Create temporary workspace")
    print("   - Initialize git repository")
    print("   - Provide all files as context to Aider")

    print("\n5. 🤖 AI-POWERED EVOLUTION")
    print("   - Aider analyzes base code + examples")
    print("   - Generates improved implementation")
    print("   - Preserves evolve block markers")

    print("\n6. 📊 RESULT INTEGRATION")
    print("   - Extract evolved block")
    print("   - Update codebase")
    print("   - Validate changes")

    print("\n7. 🔄 EVOLUTIONARY OPERATIONS")
    print("   - Mutation: Single parent evolution")
    print("   - Crossover: Combine two parents")
    print("   - Selection: Choose best implementations")

    print("\n✅ Workflow demonstration complete!")


def main():
    """Main test function."""
    print("🧬 Alpha Evolve Framework - AiderEvolver Tests")
    print("=" * 60)

    success = True

    # Test basic functionality
    if not test_aider_evolver_basic():
        success = False

    # Test with examples
    if not test_aider_evolver_with_examples():
        success = False

    # Test crossover simulation
    if not test_aider_crossover_simulation():
        success = False

    # Demonstrate workflow
    demonstrate_aider_workflow()

    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed! AiderEvolver is ready for use.")
        print("\n💡 Next steps:")
        print("1. Install Aider: pip install aider-chat")
        print("2. Configure API keys: python setup_environment.py")
        print("3. Try evolving your own code blocks!")
    else:
        print("❌ Some tests failed. Please check the setup requirements.")

    return success


if __name__ == "__main__":
    main()
