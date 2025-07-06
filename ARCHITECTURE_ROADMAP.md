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

### Phase 3: Enhanced Fluent API Abstractions
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

### Phase 2: LangGraph Integration (Weeks 3-4)
- [ ] Design LangGraph workflow schema
- [ ] Implement fluent API → LangGraph compiler
- [ ] Add checkpointing support
- [ ] Integrate LangSmith for debugging
- [ ] Test durable execution scenarios

### Phase 3: Enhanced Abstractions (Weeks 5-6)
- [ ] Implement resource budgeting system
- [ ] Add Hyperband search pattern
- [ ] Create exploration/optimization phase transitions
- [ ] Implement component co-evolution
- [ ] Add multi-objective optimization support

### Phase 4: Production Features (Weeks 7-8)
- [ ] Modal/AWS deployment support
- [ ] Performance optimization
- [ ] Comprehensive testing
- [ ] Documentation and examples
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
