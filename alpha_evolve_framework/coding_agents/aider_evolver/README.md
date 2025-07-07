# AiderEvolver - AI-Powered Code Evolution

AiderEvolver is an advanced code evolution agent that integrates [Aider](https://aider.chat) with the Alpha Evolve Framework to evolve code blocks using AI with example implementations for inspiration.

## Features

- 🧬 **AI-Powered Evolution**: Uses Aider's AI capabilities to evolve code blocks
- 📚 **Example-Based Learning**: Provides example implementations as inspiration
- 🔄 **Crossover Operations**: Combines features from multiple implementations
- 🎯 **Objective-Driven**: Evolves code with specific goals and constraints
- 🔧 **Block-Based**: Works with existing `EVOLVE-BLOCK-START/END` markers
- 🌍 **Multi-Language Support**: Supports any language that Aider supports
- 🚀 **Performance Optimization**: Optimizes code for specific performance targets
- 🔍 **Context-Aware**: Uses surrounding code context for better evolution

## Installation

1. Install Aider:
```bash
pip install aider-chat
```

2. Configure API keys in your `.env` file:
```bash
# For Gemini models
GEMINI_API_KEY=your_gemini_api_key

# For OpenAI models  
OPENAI_API_KEY=your_openai_api_key
```

## Quick Start

```python
from alpha_evolve_framework.coding_agents.aider_evolver import AiderEvolver
from alpha_evolve_framework.coding_agents.llm_block_evolver.codebase import Codebase

# Initialize AiderEvolver
evolver = AiderEvolver(model="gemini-1.5-flash")

# Create codebase with evolve blocks
code = """
def main():
    # EVOLVE-BLOCK-START sorting_algorithm
    def sort_numbers(data):
        # Simple bubble sort
        for i in range(len(data)):
            for j in range(len(data) - 1):
                if data[j] > data[j+1]:
                    data[j], data[j+1] = data[j+1], data[j]
        return data
    # EVOLVE-BLOCK-END
    
    test_data = [64, 34, 25, 12, 22, 11, 90]
    result = sort_numbers(test_data.copy())
    print(f"Sorted: {result}")
"""

codebase = Codebase(code)

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

# Evolve the code block
evolved_codebase = evolver.evolve_with_examples(
    codebase,
    block_name="sorting_algorithm",
    objective="Optimize the sorting algorithm for better performance",
    example_implementations=[quicksort_example]
)

print("Evolved code:")
print(evolved_codebase.reconstruct_full_code())
```

## Core Methods

### `evolve_with_examples()`

Evolve a code block using example implementations as inspiration.

```python
evolved_codebase = evolver.evolve_with_examples(
    base_codebase=codebase,
    block_name="my_block",
    objective="Improve performance and readability",
    example_implementations=["example1", "example2"],
    context_files=["context.md"],
    requirements="Must maintain backwards compatibility",
    constraints="Keep function signature unchanged"
)
```

### `crossover_with_aider()`

Combine features from two implementations using Aider.

```python
child_codebase = evolver.crossover_with_aider(
    parent1_codebase=parent1,
    parent2_codebase=parent2,
    block_name="algorithm",
    objective="Combine the best features from both implementations"
)
```

### `mutation_with_aider()`

Perform mutation on a code block.

```python
mutated_codebase = evolver.mutation_with_aider(
    codebase=codebase,
    block_name="algorithm",
    mutation_type="optimize",  # or "improve", "refactor", "robust", "creative"
    performance_target="10x faster execution"
)
```

### `evolve()` - BaseAgent Interface

Compatible with the BaseAgent interface for evolutionary frameworks.

```python
candidate = ProgramCandidate(
    id="sort_v1",
    code_str="def sort_numbers(data): ...",
    block_name="sorting_algorithm"
)

evolved_candidate = await evolver.evolve(
    candidate=candidate,
    task_description="Optimize for better performance",
    context={
        "examples": [quicksort_example],
        "performance_target": "O(n log n) complexity"
    }
)
```

## How It Works

1. **Workspace Creation**: Creates a temporary workspace with the base program and examples
2. **Git Repository**: Initializes a git repository for Aider to track changes
3. **Context Files**: Provides example implementations and instructions as context
4. **Aider Execution**: Runs Aider with the objective and context files
5. **Code Extraction**: Extracts the evolved code block from the result
6. **Codebase Update**: Updates the original codebase with the evolved block

## Evolution Process

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Base Code     │    │   Examples      │    │   Objective     │
│   with Blocks   │    │   & Context     │    │   & Goals       │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼───────────────┐
                    │     Aider Evolution        │
                    │   • Analyzes base code     │
                    │   • Studies examples       │
                    │   • Applies improvements   │
                    │   • Preserves markers      │
                    └─────────────┬───────────────┘
                                 │
                    ┌─────────────▼───────────────┐
                    │    Evolved Code Block      │
                    │   • Improved performance   │
                    │   • Better code quality    │
                    │   • Enhanced features      │
                    │   • Maintained compatibility│
                    └─────────────────────────────┘
```

## Capabilities

AiderEvolver supports the following capabilities:

- `code_evolution` - General code evolution and improvement
- `block_evolution` - Specific code block evolution
- `example_based_learning` - Learning from example implementations
- `crossover` - Combining features from multiple implementations
- `mutation` - Various types of code mutations
- `performance_optimization` - Performance-focused improvements
- `code_refactoring` - Code structure improvements
- `feature_addition` - Adding new features to existing code
- `aider_integration` - Native Aider tool integration
- `multi_language_support` - Support for multiple programming languages
- `context_aware_evolution` - Evolution based on surrounding code context

## Configuration

### Model Selection

```python
# Use Gemini models (default)
evolver = AiderEvolver(model="gemini-1.5-flash")

# Use OpenAI models
evolver = AiderEvolver(model="gpt-4")

# Use other supported models
evolver = AiderEvolver(model="claude-3-sonnet")
```

### Evolution Parameters

```python
evolver = AiderEvolver(
    model="gemini-1.5-flash",
    aider_executable="aider",  # Custom Aider path
    auto_commit=True,          # Auto-commit changes
    config={
        "max_iterations": 5,
        "timeout": 300
    }
)
```

## Use Cases

### 1. Algorithm Optimization

```python
# Optimize sorting algorithms
evolved_codebase = evolver.evolve_with_examples(
    codebase, "sorting_algorithm",
    objective="Optimize for O(n log n) complexity",
    example_implementations=[quicksort_example, mergesort_example]
)
```

### 2. Performance Improvement

```python
# Improve performance of existing code
evolved_codebase = evolver.mutation_with_aider(
    codebase, "data_processing",
    mutation_type="optimize",
    performance_target="50% faster execution"
)
```

### 3. Code Refactoring

```python
# Refactor for better maintainability
evolved_codebase = evolver.mutation_with_aider(
    codebase, "legacy_function",
    mutation_type="refactor",
    constraints="Maintain backwards compatibility"
)
```

### 4. Feature Addition

```python
# Add new features to existing code
evolved_codebase = evolver.evolve_with_examples(
    codebase, "api_handler",
    objective="Add error handling and retry logic",
    example_implementations=[robust_handler_example]
)
```

### 5. Cross-Language Translation

```python
# Convert code between languages
evolved_codebase = evolver.evolve_with_examples(
    codebase, "algorithm",
    objective="Convert Python implementation to Rust",
    example_implementations=[rust_examples]
)
```

## Integration with Evolutionary Algorithms

AiderEvolver can be integrated with various evolutionary algorithms:

```python
from alpha_evolve_framework.optimization import GeneticAlgorithm

# Create population of candidates
population = [
    ProgramCandidate(id=f"candidate_{i}", code_str=code, block_name="algorithm")
    for i, code in enumerate(initial_codes)
]

# Use AiderEvolver as the mutation operator
ga = GeneticAlgorithm(
    mutation_operator=evolver,
    crossover_operator=evolver,
    population_size=50,
    generations=100
)

# Run evolution
best_candidate = ga.evolve(population)
```

## Best Practices

1. **Clear Objectives**: Provide specific, measurable objectives for evolution
2. **Quality Examples**: Use high-quality example implementations as inspiration
3. **Proper Context**: Include relevant context files and documentation
4. **Incremental Evolution**: Make small, focused improvements rather than large changes
5. **Validation**: Always validate evolved code before deployment
6. **Version Control**: Keep track of evolution history and performance metrics

## Troubleshooting

### Common Issues

1. **Aider not found**: Install Aider with `pip install aider-chat`
2. **API key missing**: Configure API keys in `.env` file
3. **Model not supported**: Check Aider documentation for supported models
4. **Evolution timeout**: Increase timeout or simplify objectives
5. **Block not found**: Ensure block names match exactly

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

evolver = AiderEvolver(model="gemini-1.5-flash")
# This will show detailed logs of the evolution process
```

## Examples

See the `examples/` directory for comprehensive examples:

- `examples/aider_evolution_demo.py` - Complete demo of all features
- `tests/test_aider_evolver.py` - Unit tests and usage examples

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or contributions, please visit:
- GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- Documentation: [Read the docs](https://your-docs-site.com)
- Examples: [See examples](examples/)

---

*AiderEvolver - Evolving code with AI assistance, one block at a time.* 🧬✨
