"""
Comprehensive demo of AiderEvolver functionality.

This demo shows how to use AiderEvolver to evolve code blocks using Aider
with example implementations for inspiration.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha_evolve_framework.coding_agents.aider_evolver.aider_evolver import (
    AiderEvolver,
)
from alpha_evolve_framework.coding_agents.llm_block_evolver.codebase import Codebase
from alpha_evolve_framework.core_types import ProgramCandidate


def demo_basic_evolution():
    """Demo basic evolution functionality."""
    print("🧬 AiderEvolver Basic Evolution Demo")
    print("=" * 50)

    # Sample code with an evolve block
    initial_code = """
def main():
    print("Sorting Algorithm Demo")
    
    # EVOLVE-BLOCK-START sorting_algorithm
    def sort_numbers(numbers):
        # Simple bubble sort implementation
        n = len(numbers)
        for i in range(n):
            for j in range(0, n-i-1):
                if numbers[j] > numbers[j+1]:
                    numbers[j], numbers[j+1] = numbers[j+1], numbers[j]
        return numbers
    # EVOLVE-BLOCK-END
    
    # Test the sorting
    test_data = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original: {test_data}")
    sorted_data = sort_numbers(test_data.copy())
    print(f"Sorted: {sorted_data}")

if __name__ == "__main__":
    main()
"""

    print(f"📋 Initial Code:\n{initial_code}")

    # Create codebase
    codebase = Codebase(initial_code)
    print(f"\n🔧 Available blocks: {codebase.get_block_names()}")

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

    merge_sort_example = """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
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

    print("\n📚 Example Implementations:")
    print("1. QuickSort implementation")
    print("2. Merge Sort implementation")

    # This is where you would run the actual evolution
    print("\n🔄 Evolution Process:")
    print("1. Initialize AiderEvolver")
    print("2. Create workspace with base code and examples")
    print("3. Run Aider with objective: 'Optimize the sorting algorithm'")
    print("4. Extract evolved implementation")
    print("5. Update codebase with improved algorithm")

    print("\n✅ Demo setup complete!")
    print("💡 To run actual evolution, ensure:")
    print("   - Aider is installed: pip install aider-chat")
    print("   - API keys are configured in .env file")
    return codebase, [quicksort_example, merge_sort_example]


def demo_crossover_evolution():
    """Demo crossover between two implementations."""
    print("\n🧬 AiderEvolver Crossover Demo")
    print("=" * 50)

    # Parent 1: Bubble sort with early termination
    parent1_code = """
def main():
    # EVOLVE-BLOCK-START sorting_algorithm
    def sort_numbers(numbers):
        # Optimized bubble sort with early termination
        n = len(numbers)
        for i in range(n):
            swapped = False
            for j in range(0, n-i-1):
                if numbers[j] > numbers[j+1]:
                    numbers[j], numbers[j+1] = numbers[j+1], numbers[j]
                    swapped = True
            if not swapped:
                break  # Early termination if no swaps
        return numbers
    # EVOLVE-BLOCK-END
    
    test_data = [64, 34, 25, 12, 22, 11, 90]
    result = sort_numbers(test_data.copy())
    print(f"Sorted: {result}")
"""

    # Parent 2: Selection sort
    parent2_code = """
def main():
    # EVOLVE-BLOCK-START sorting_algorithm
    def sort_numbers(numbers):
        # Selection sort implementation
        n = len(numbers)
        for i in range(n):
            min_idx = i
            for j in range(i+1, n):
                if numbers[j] < numbers[min_idx]:
                    min_idx = j
            numbers[i], numbers[min_idx] = numbers[min_idx], numbers[i]
        return numbers
    # EVOLVE-BLOCK-END
    
    test_data = [64, 34, 25, 12, 22, 11, 90]
    result = sort_numbers(test_data.copy())
    print(f"Sorted: {result}")
"""

    print("👨‍👩‍👧‍👦 Parent 1: Bubble sort with early termination")
    print("👨‍👩‍👧‍👦 Parent 2: Selection sort")

    # Create parent codebases
    parent1_codebase = Codebase(parent1_code)
    parent2_codebase = Codebase(parent2_code)

    print(f"\n🔧 Parent 1 blocks: {parent1_codebase.get_block_names()}")
    print(f"🔧 Parent 2 blocks: {parent2_codebase.get_block_names()}")

    print("\n🔄 Crossover Process:")
    print("1. Extract both parent implementations")
    print("2. Provide both as examples to Aider")
    print("3. Objective: 'Combine the best features from both implementations'")
    print("4. Aider creates hybrid combining early termination + selection logic")
    print("5. Result: Optimized hybrid algorithm")

    print("\n✅ Crossover demo complete!")
    return parent1_codebase, parent2_codebase


async def demo_program_candidate_evolution():
    """Demo evolving a ProgramCandidate."""
    print("\n🧬 AiderEvolver Program Candidate Demo")
    print("=" * 50)

    # Create a program candidate
    candidate = ProgramCandidate(
        id="sort_v1",
        code_str="""
def sort_numbers(data):
    # Basic bubble sort
    for i in range(len(data)):
        for j in range(len(data) - 1):
            if data[j] > data[j+1]:
                data[j], data[j+1] = data[j+1], data[j]
    return data
""",
        block_name="sorting_algorithm",
    )

    print(f"📋 Initial candidate: {candidate.id}")
    print(f"🔧 Block name: {candidate.block_name}")
    print(f"📝 Code length: {len(candidate.code_str)} characters")

    # This would be the actual evolution call
    print("\n🔄 Evolution Process:")
    print("1. Create AiderEvolver instance")
    print("2. Wrap candidate code with evolve block markers")
    print("3. Add example implementations as context")
    print("4. Run Aider with task description")
    print("5. Extract evolved code and create new candidate")

    # Simulate evolution result
    evolved_candidate = ProgramCandidate(
        id=f"{candidate.id}_evolved",
        code_str="""
def sort_numbers(data):
    # Optimized bubble sort with early termination
    n = len(data)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            if data[j] > data[j+1]:
                data[j], data[j+1] = data[j+1], data[j]
                swapped = True
        if not swapped:
            break
    return data
""",
        block_name=candidate.block_name,
        parent_id=candidate.id,
        generation=candidate.generation + 1,
    )

    print(f"\n✅ Evolved candidate: {evolved_candidate.id}")
    print(f"📈 Generation: {evolved_candidate.generation}")
    print(f"👨‍👦 Parent: {evolved_candidate.parent_id}")

    return candidate, evolved_candidate


def demo_capabilities():
    """Demo AiderEvolver capabilities."""
    print("\n🧬 AiderEvolver Capabilities Demo")
    print("=" * 50)

    try:
        # This will fail without Aider installation, but that's expected
        evolver = AiderEvolver(model="gemini-1.5-flash")
        capabilities = evolver.get_capabilities()

        print("🎯 AiderEvolver Capabilities:")
        for i, capability in enumerate(capabilities, 1):
            print(f"   {i}. {capability}")

        print(f"\n✅ Total capabilities: {len(capabilities)}")

    except RuntimeError as e:
        print(f"⚠️  AiderEvolver initialization failed: {e}")
        print("📋 Expected capabilities:")
        capabilities = [
            "code_evolution",
            "block_evolution",
            "example_based_learning",
            "crossover",
            "mutation",
            "performance_optimization",
            "code_refactoring",
            "feature_addition",
            "aider_integration",
            "multi_language_support",
            "context_aware_evolution",
        ]

        for i, capability in enumerate(capabilities, 1):
            print(f"   {i}. {capability}")


def main():
    """Main demo function."""
    print("🚀 AiderEvolver Comprehensive Demo")
    print("=" * 60)

    # Demo 1: Basic evolution
    codebase, examples = demo_basic_evolution()

    # Demo 2: Crossover evolution
    parent1, parent2 = demo_crossover_evolution()

    # Demo 3: Program candidate evolution
    asyncio.run(demo_program_candidate_evolution())

    # Demo 4: Capabilities
    demo_capabilities()

    print("\n" + "=" * 60)
    print("✅ All demos completed successfully!")
    print("\n💡 Next Steps:")
    print("1. Install Aider: pip install aider-chat")
    print("2. Configure API keys in .env file")
    print("3. Run actual evolution experiments")
    print("4. Integrate with your evolutionary algorithms")

    print("\n🎯 Use Cases:")
    print("- Code optimization and refactoring")
    print("- Algorithm evolution and improvement")
    print("- Feature addition and enhancement")
    print("- Cross-language code translation")
    print("- Performance optimization")
    print("- Bug fixing and robustness improvement")


if __name__ == "__main__":
    main()
