# Evolutionary Framework Architecture Roadmap

## Current State Analysis

### Existing Architecture Strengths
- **Solid Foundation**: Well-structured components (LLMBlockEvolver, databases, evaluators)
- **Fluent API**: Clean, readable interface for configuring evolution experiments
- **Island Model**: Multi-population evolution with migration support
- **Staged Evolution**: Sequential phases with different LLM settings and strategies

### Current Limitations
- **No Sandboxing**: Code execution happens on host system (safety/isolation concerns)
- **Limited Scalability**: Single-machine execution, no cloud distribution
- **No Checkpointing**: Long runs are vulnerable to failures, no resume capability
- **Simple Orchestration**: Linear stage progression, limited advanced patterns

## Integration Plan

### Phase 1: SWE-ReX Integration
**Goal**: Safe, scalable code execution with sandboxing

#### Benefits
- **Sandboxed Execution**: Each evolved code candidate runs in isolated containers
- **Massively Parallel**: Island model can leverage parallel Docker containers/Modal sandboxes
- **Platform Flexibility**: Local development → Cloud production seamlessly
- **Interactive Sessions**: LLM agents can use interactive tools (ipython, debuggers, etc.)

#### Implementation
```python
# Current: Direct evaluation
result = evaluator.evaluate(code)

# With SWE-ReX: Sandboxed evaluation
class SWEReXEvaluator(FunctionalEvaluator):
    def __init__(self, deployment_type="docker"):
        self.deployment = self._create_deployment(deployment_type)
    
    async def evaluate(self, code):
        async with self.deployment.runtime as runtime:
            await runtime.create_session(CreateBashSessionRequest())
            result = await runtime.run_in_session(BashAction(command=f"python {code_file}"))
            return self._parse_result(result)
```

#### Technical Details
- Add `SWEReXExecutor` component for deployment management
- Modify all evaluators to support async sandboxed execution
- Support multiple deployment types: Local, Docker, Modal, AWS Fargate
- Implement parallel island execution across containers

### Phase 2: LangGraph Workflow Backend
**Goal**: Durable execution with checkpointing and advanced orchestration

#### Benefits
- **Durable Execution**: Built-in persistence and failure recovery
- **Checkpointing**: Resume interrupted runs from last checkpoint
- **State Management**: Better handling of complex state transitions
- **Human-in-the-loop**: Interactive evolution steering, manual candidate selection
- **Debugging**: LangSmith integration for evolution flow visualization

#### Implementation Strategy
- Keep fluent API as frontend interface
- Compile fluent API configurations to LangGraph workflow definitions
- Each stage becomes a LangGraph node with checkpointing
- Islands run as parallel branches in the graph

### Phase 3: Coding Agent Architecture
**Goal**: Replace simple LLMBlockEvolver with sophisticated coding agents

#### Current Limitations
- **LLMBlockEvolver**: Simple code block generation without context
- **No Tool Usage**: Cannot use debuggers, profilers, testing frameworks
- **Limited Refinement**: Single-shot generation without iterative improvement
- **No Codebase Understanding**: Works on isolated functions, not full projects

#### Coding Agent Benefits
- **Tool Integration**: Native support for python_repl, bash, editor, git, debuggers
- **Iterative Refinement**: Multi-step problem-solving with feedback loops
- **Codebase Awareness**: Can work with entire projects and dependencies
- **Interactive Development**: Can debug, test, and refine solutions systematically

#### Agent Integration Architecture
```python
# Current: Simple block evolution
class LLMBlockEvolver:
    def evolve(self, code):
        return self.llm.generate(f"Improve this code: {code}")

# New: Sophisticated agent evolution
class CodingAgentEvolver:
    def __init__(self, agent_type="swe-agent"):
        self.agent = self._create_agent(agent_type)
    
    async def evolve(self, code, context):
        task = f"Improve this code: {code}"
        result = await self.agent.solve(
            task=task,
            context=context,
            max_iterations=20,
            tools=["python_repl", "bash", "editor", "git"]
        )
        return result.solution
```

#### Two-Tier Evolution System
```python
# Population-level exploration (broad search)
islands = [
    Island(population_size=50, evolver=LLMBlockEvolver()),
    Island(population_size=50, evolver=LLMBlockEvolver()),
    Island(population_size=50, evolver=LLMBlockEvolver())
]

# Individual-level refinement (deep improvement)
coding_agents = [
    CodingAgentEvolver(agent_type="swe-agent"),
    CodingAgentEvolver(agent_type="aider"),
    CodingAgentEvolver(agent_type="custom")
]

# Combined workflow
for generation in range(max_generations):
    # Parallel island evolution
    for island in islands:
        island.evolve_parallel()
    
    # Select promising candidates
    promising = select_top_candidates(islands, top_k=10)
    
    # Deep refinement with agents
    refined = []
    for candidate in promising:
        agent = select_best_agent(candidate)
        improved = await agent.evolve(candidate, context)
        refined.append(improved)
    
    # Feed back to population
    for island in islands:
        island.inject_candidates(refined)
```

#### Agent-Specific Implementations
- **SWE-Agent**: Sandboxed environment, systematic problem-solving
- **Aider**: Git-based code improvement, commit history tracking
- **Custom Agents**: Domain-specific tools and workflows

### Phase 4: Enhanced Fluent API Abstractions
**Goal**: More expressive evolutionary algorithm patterns

## Enhanced Fluent API Design

### Current Stage-Based Model
```python
agent.add_stage("explore", max_generations=50, llm_settings=[...])
     .add_stage("optimize", max_generations=100, llm_settings=[...])
```

### Enhanced Abstractions

#### Resource-Aware Phases
```python
agent.define_problem(...)
     .explore_phase(
         budget=ResourceBudget(generations=50, time="30min", tokens=100000),
         strategy=ExplorationStrategy.DIVERSE_SAMPLING
     )
     .optimize_phase(
         budget=ResourceBudget(generations=100, time="1hr", tokens=200000),
         strategy=OptimizationStrategy.HYPERBAND,
         prune_threshold=0.8
     )
     .with_execution(SWEReXExecutor(deployment_type="modal"))
```

#### Advanced Search Patterns
```python
# Hyperband-style resource allocation
agent.hyperband_search(
    initial_budget=10,
    max_budget=1000,
    eta=3,  # pruning factor
    phases=["explore", "refine", "optimize"]
)

# Multi-objective optimization
agent.pareto_optimization(
    objectives=["performance", "code_quality", "readability"],
    budget_allocation="adaptive"
)

# Component co-evolution
agent.co_evolve(
    components=["code_generator", "test_generator"],
    interaction_frequency=10  # generations between interaction
)
```

#### Iterative Refinement Cycles
```python
agent.iterate_cycle(
    components=["evaluator", "feature_extractor"],
    solutions=["best_candidates"],
    max_cycles=5,
    convergence_criteria="diversity_threshold"
)
```

#### Exploration → Optimization Transitions
```python
agent.exploration_to_optimization(
    exploration_budget=0.3,  # 30% of total budget
    transition_criteria="performance_plateau",
    optimization_strategy="focused_search"
)
```

## Technical Architecture

### Component Integration
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Fluent API    │───▶│  LangGraph       │───▶│    SWE-ReX      │
│   (Frontend)    │    │  (Orchestration) │    │  (Execution)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
       │                        │                        │
       ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ • Resource      │    │ • State Mgmt     │    │ • Docker        │
│   Budgeting     │    │ • Checkpointing  │    │ • Modal         │
│ • Strategy      │    │ • Parallel Exec  │    │ • AWS Fargate   │
│   Selection     │    │ • Human-in-loop  │    │ • Local         │
│ • Phase Mgmt    │    │ • Recovery       │    │ • Daytona       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Enhanced Architecture with Parallel Evolution and Agent Integration
```
                       ┌─────────────────────────────────────────────────┐
                       │              Fluent API                         │
                       │ • Resource Budgeting  • Strategy Selection      │
                       │ • Phase Management    • Agent Configuration     │
                       └─────────────────────────┬───────────────────────┘
                                                 │
                                                 ▼
                       ┌─────────────────────────────────────────────────┐
                       │            LangGraph Compiler                   │
                       │ • Parallel Branch Generation                    │
                       │ • Agent Workflow Integration                    │
                       │ • Checkpoint Strategy Planning                  │
                       └─────────────────────────┬───────────────────────┘
                                                 │
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              LangGraph Workflow                                       │
│                                                                                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                   │
│  │    Island 1     │    │    Island 2     │    │    Island N     │                   │
│  │  Population     │    │  Population     │    │  Population     │                   │
│  │  Evolution      │    │  Evolution      │    │  Evolution      │                   │
│  │  (Parallel)     │    │  (Parallel)     │    │  (Parallel)     │                   │
│  └─────┬───────────┘    └─────┬───────────┘    └─────┬───────────┘                   │
│        │                      │                      │                               │
│        └──────────────────────┼──────────────────────┘                               │
│                               │                                                      │
│                               ▼                                                      │
│                    ┌─────────────────────┐                                          │
│                    │  Candidate Selection │                                          │
│                    │  (Top K Promising)   │                                          │
│                    └─────────┬───────────┘                                          │
│                              │                                                      │
│                              ▼                                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                   │
│  │   SWE-Agent     │    │     Aider       │    │  Custom Agent   │                   │
│  │  Deep Refine    │    │  Git-based      │    │  Domain-specific │                   │
│  │  (Parallel)     │    │  Improvement    │    │  Optimization   │                   │
│  └─────┬───────────┘    └─────┬───────────┘    └─────┬───────────┘                   │
│        │                      │                      │                               │
│        └──────────────────────┼──────────────────────┘                               │
│                               │                                                      │
│                               ▼                                                      │
│                    ┌─────────────────────┐                                          │
│                    │  Result Integration  │                                          │
│                    │  Back to Population  │                                          │
│                    └─────────────────────┘                                          │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                       ┌─────────────────────────────────────────────────┐
                       │                 SWE-ReX                         │
                       │ • Sandboxed Execution                           │
                       │ • Multi-deployment Support                      │
                       │ • Interactive Tool Access                       │
                       │ • Scalable Cloud Infrastructure                 │
                       └─────────────────────────────────────────────────┘
```

### Parallel Evolution Flow
```
Generation N:
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │   Island 1  │    │   Island 2  │    │   Island 3  │
  │   50 pop    │    │   50 pop    │    │   50 pop    │
  │   evolving  │    │   evolving  │    │   evolving  │
  │  (parallel) │    │  (parallel) │    │  (parallel) │
  └─────────────┘    └─────────────┘    └─────────────┘
          │                  │                  │
          └─────────┬────────┴────────┬─────────┘
                    │                 │
                    ▼                 ▼
              ┌─────────────────────────────┐
              │    Migration & Selection    │
              │    Top 10 candidates       │
              └─────────────────────────────┘
                            │
                            ▼
    ┌─────────────┐    ┌─────────

### Execution Flow
1. **Fluent API** defines high-level evolutionary strategy
2. **Compiler** translates to LangGraph workflow definition
3. **LangGraph** orchestrates execution with checkpointing
4. **SWE-ReX** provides sandboxed, parallel execution environment
5. **Results** flow back through the stack with state persistence

## Implementation Phases

### Phase 1: SWE-ReX Foundation (Weeks 1-2)
- [ ] Add SWE-ReX dependency to project
- [ ] Create `SWEReXExecutor` component
- [ ] Modify `FunctionalEvaluator` for async execution
- [ ] Add Docker deployment support
- [ ] Test parallel island execution

### Phase 2: LangGraph Parallel Evolution (Weeks 3-4)
- [ ] Design LangGraph workflow schema with parallel branches
- [ ] Implement parallel island execution using LangGraph
- [ ] Add migration synchronization points between islands
- [ ] Implement fluent API → LangGraph compiler
- [ ] Add checkpointing support for parallel workflows
- [ ] Test durable parallel execution scenarios

### Phase 3: Coding Agent Integration (Weeks 5-6)
- [ ] Create `CodingAgentEvolver` interface
- [ ] Integrate SWE-agent for individual candidate refinement
- [ ] Add Aider wrapper for git-based code improvements
- [ ] Implement two-tier evolution system (population + individual)
- [ ] Add intelligent candidate selection for deep refinement
- [ ] Create feedback loops from individual refinement to population

### Phase 4: Enhanced Abstractions (Weeks 7-8)
- [ ] Implement resource budgeting system
- [ ] Add Hyperband search pattern with parallel execution
- [ ] Create exploration/optimization phase transitions
- [ ] Implement component co-evolution
- [ ] Add multi-objective optimization support
- [ ] Integrate coding agents with advanced search patterns

### Phase 5: Multi-Block Evolution & Evolver Parity (Weeks 9-10)
- [ ] **Multi-Block Evolution Support**
  - [ ] Extend `Codebase` class to support multiple evolve blocks
  - [ ] Add block dependency tracking and execution order
  - [ ] Implement cross-block mutation and crossover operations
  - [ ] Support hierarchical code structure evolution
- [ ] **Aider Multi-Block Integration**
  - [ ] Extend `AiderEvolver` to work with multiple code blocks
  - [ ] Add support for file-level and function-level evolution
  - [ ] Implement Aider's git-based tracking for multi-block changes
  - [ ] Add context-aware evolution across related code blocks
- [ ] **Evolver Parity Analysis & Implementation**
  - [ ] Create comprehensive feature comparison matrix (Aider vs LLMBlock)
  - [ ] Implement missing features in both evolvers for feature parity
  - [ ] Add standardized evaluation metrics for both evolvers
  - [ ] Create benchmark suite to compare evolver performance
  - [ ] Implement cross-evolver result validation and comparison
- [ ] **Enhanced Multi-Block Capabilities**
  - [ ] Add inter-block communication patterns
  - [ ] Implement block-level fitness evaluation
  - [ ] Support conditional block evolution based on dependencies
  - [ ] Add visualization for multi-block evolution progress

### Phase 6: Production Features (Weeks 11-12)
- [ ] Modal/AWS deployment support for parallel execution
- [ ] Performance optimization for agent-based evolution
- [ ] Comprehensive testing of parallel + agent systems
- [ ] Documentation and examples for new capabilities
- [ ] Migration guide from current API

## Technical Considerations

### Key Questions to Resolve
1. **Execution Granularity**: SWE-ReX at individual candidate level or island level?
   - *Recommendation*: Island level for efficiency, with batch candidate evaluation
   
2. **State Persistence**: How to handle checkpointing with enhanced abstractions?
   - *Recommendation*: Each phase as checkpoint boundary, with incremental saves
   
3. **Resource Management**: How should fluent API express resource constraints?
   - *Recommendation*: `ResourceBudget` objects with time/compute/token limits
   
4. **Backward Compatibility**: Migration path for existing experiments?
   - *Recommendation*: Adapter layer to translate old API to new abstractions

### Design Principles
- **Expressiveness over Brevity**: API should clearly express intent
- **Composability**: Building blocks that combine naturally
- **Extensibility**: Easy to add new strategies and patterns
- **Safety**: Sandboxed execution by default
- **Scalability**: Cloud-native from the start

## Success Metrics
- **Safety**: 100% sandboxed execution, no host system access
- **Scalability**: Support 100+ parallel islands across cloud providers
- **Reliability**: Resume 95%+ of interrupted runs from checkpoints
- **Expressiveness**: Implement 5+ advanced evolutionary patterns
- **Performance**: <10% overhead compared to current direct execution

## Future Vision
The enhanced framework becomes a domain-specific language for evolutionary algorithms, where researchers can express complex search strategies in a few lines of readable code, while the system handles all the infrastructure complexity of distributed, durable, sandboxed execution.

```python
# Vision: Express complex evolutionary strategies simply
agent.define_problem(code_generation_task)
     .multi_objective_search(["performance", "readability", "maintainability"])
     .hyperband_phases(eta=3, max_budget=1000)
     .co_evolve(["generator", "critic", "tester"])
     .with_human_guidance(review_frequency=50)
     .scale_to_cloud(max_instances=100)
     .run_with_checkpoints()
