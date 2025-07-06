"""
Simple POC demo showing LangGraph-style workflow concept.
This demonstrates the workflow pattern without requiring LangGraph.
"""

import sys
import os
import logging
from typing import Dict, Any, Tuple, List
import time
import uuid

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_evolve_framework.core_types import Program, LLMSettings, StageOutput
from alpha_evolve_framework.codebase import Codebase
from alpha_evolve_framework.llm_manager import LLMManager
from alpha_evolve_framework.prompt_engine import PromptEngine
from alpha_evolve_framework.functional_evaluator import FunctionalEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockLangGraphWorkflow:
    """Mock LangGraph workflow to demonstrate the concept."""
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.entry_point = None
        
    def add_node(self, name: str, func):
        self.nodes[name] = func
        logger.info(f"Added node: {name}")
        
    def add_edge(self, from_node: str, to_node: str):
        self.edges.append((from_node, to_node))
        logger.info(f"Added edge: {from_node} -> {to_node}")
        
    def set_entry_point(self, node_name: str):
        self.entry_point = node_name
        logger.info(f"Set entry point: {node_name}")
        
    def stream(self, initial_state: Dict[str, Any], config: Dict[str, Any]):
        """Simulate streaming execution of the workflow."""
        logger.info("Starting workflow execution")
        
        current_state = initial_state
        current_node = self.entry_point
        
        # Simple execution: follow the linear path for POC
        node_sequence = [
            "initialize_population",
            "generate_candidates", 
            "evaluate_candidates",
            "update_population",
            "check_termination"
        ]
        
        for node_name in node_sequence:
            if node_name in self.nodes:
                logger.info(f"Executing node: {node_name}")
                
                # Execute node function
                current_state = self.nodes[node_name](current_state)
                
                # Yield state update (simulating streaming)
                yield {node_name: current_state}
                
                # Check termination condition
                if current_state.get("should_terminate", False):
                    logger.info("Workflow terminated")
                    break
                    
                # Simple delay to simulate processing
                time.sleep(0.5)
        
        logger.info("Workflow execution completed")


def create_mock_evolution_workflow():
    """Create a mock workflow that demonstrates the LangGraph concept."""
    
    workflow = MockLangGraphWorkflow()
    
    # Add nodes (using the same node functions we created)
    workflow.add_node("initialize_population", initialize_population)
    workflow.add_node("generate_candidates", generate_candidates)
    workflow.add_node("evaluate_candidates", evaluate_candidates)
    workflow.add_node("update_population", update_population)
    workflow.add_node("check_termination", check_termination)
    
    # Set entry point
    workflow.set_entry_point("initialize_population")
    
    # Add edges
    workflow.add_edge("initialize_population", "generate_candidates")
    workflow.add_edge("generate_candidates", "evaluate_candidates")
    workflow.add_edge("evaluate_candidates", "update_population")
    workflow.add_edge("update_population", "check_termination")
    
    return workflow


# Simple node functions for the mock workflow
def initialize_population(state: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize the population for evolution."""
    logger.info(f"Initializing population for stage: {state.get('stage_name', 'unknown')}")
    
    if not state.get("current_population"):
        # Parse the initial codebase to extract the evolve block
        codebase = Codebase(state["initial_codebase_code"])
        block_names = codebase.get_block_names()
        
        if not block_names:
            logger.error("No evolve blocks found in initial codebase")
            state["error_message"] = "No evolve blocks found in initial codebase"
            state["should_terminate"] = True
            return state
        
        # Use the first block for the demo
        block_name = block_names[0]
        block = codebase.get_block(block_name)
        
        # Create initial program with the block code
        initial_program = Program(
            id=f"initial_{uuid.uuid4().hex[:8]}",
            code_str=block.current_code,
            block_name=block_name,
            parent_id=None,
            scores={},
            eval_details={},
            generation=0
        )
        state["current_population"] = [initial_program]
        logger.info(f"Created initial population with {len(state['current_population'])} program(s)")
        logger.info(f"Initial block '{block_name}' code: {block.current_code.strip()}")
    
    return state


def generate_candidates(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate candidate programs using LLM."""
    logger.info(f"Generating candidates for generation {state.get('generation', 0)}")
    
    try:
        # Create LLM manager
        llm_manager = LLMManager(
            default_api_key=state["api_key"],
            llm_settings_list=state["llm_settings"],
            min_inter_request_delay_sec=1.0
        )
        
        # Create prompt engine
        prompt_engine = PromptEngine(
            task_description=state["task_description"],
            problem_specific_instructions=state["stage_config"].get("problem_specific_instructions")
        )
        
        candidates = []
        current_population = state.get("current_population", [])
        
        if current_population:
            # Generate 2 candidates for POC
            for i in range(2):
                parent = current_population[i % len(current_population)]
                
                # Create a simple codebase for context
                codebase = Codebase(state["initial_codebase_code"])
                if parent.block_name:
                    codebase.update_block_code(parent.block_name, parent.code_str)
                
                # Generate improvement prompt  
                prompt = prompt_engine.build_evolution_prompt(
                    parent_program=parent,
                    target_block_name=parent.block_name,
                    codebase=codebase,
                    output_format="full_code"
                )
                
                # Get LLM response
                response = llm_manager.generate_code_modification(prompt)
                
                # Extract code from response
                candidate_code = extract_code_from_response(response, parent.code_str)
                
                # Create candidate program
                candidate = Program(
                    id=f"gen{state.get('generation', 0)}_cand{i}_{uuid.uuid4().hex[:8]}",
                    code_str=candidate_code,
                    block_name=parent.block_name or "main",
                    parent_id=parent.id,
                    scores={},
                    eval_details={},
                    generation=state.get("generation", 0)
                )
                
                candidates.append(candidate)
                logger.info(f"Generated candidate {candidate.id}")
        
        state["candidate_programs"] = candidates
        logger.info(f"Generated {len(candidates)} candidates")
        
    except Exception as e:
        logger.error(f"Error generating candidates: {e}")
        state["error_message"] = f"Failed to generate candidates: {e}"
        state["should_terminate"] = True
    
    return state


def evaluate_candidates(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate candidate programs."""
    candidates = state.get("candidate_programs", [])
    logger.info(f"Evaluating {len(candidates)} candidates")
    
    try:
        # Create evaluator
        evaluator_config = state["stage_config"].get("evaluator_config", {})
        evaluator = FunctionalEvaluator(
            problem_name=state["stage_name"],
            eval_fn=evaluator_config.get("evaluator_fn"),
            config=evaluator_config
        )
        
        evaluated_programs = []
        
        for candidate in candidates:
            try:
                # Create temporary codebase for evaluation
                temp_codebase = Codebase(state["initial_codebase_code"])
                if candidate.block_name:
                    temp_codebase.update_block_code(candidate.block_name, candidate.code_str)
                
                full_code = temp_codebase.reconstruct_full_code()
                
                # Evaluate the candidate
                scores, eval_details = evaluator.evaluate(
                    candidate.id,
                    full_code,
                    state.get("generation", 0),
                    timeout_seconds=30
                )
                
                # Update candidate with evaluation results
                candidate.scores = scores
                candidate.eval_details = eval_details
                
                evaluated_programs.append(candidate)
                logger.info(f"Evaluated {candidate.id}: score = {scores.get('main_score', 'N/A')}")
                
            except Exception as e:
                logger.warning(f"Failed to evaluate candidate {candidate.id}: {e}")
                candidate.scores = {"main_score": -float("inf"), "error": str(e)}
                candidate.eval_details = {"error": str(e)}
                evaluated_programs.append(candidate)
        
        state["evaluated_programs"] = evaluated_programs
        
        # Update best program
        all_programs = state.get("current_population", []) + evaluated_programs
        best_program = max(all_programs, key=lambda p: p.scores.get("main_score", -float("inf")))
        state["best_program"] = best_program
        
        logger.info(f"Evaluation complete. Best score: {best_program.scores.get('main_score', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Error evaluating candidates: {e}")
        state["error_message"] = f"Failed to evaluate candidates: {e}"
        state["should_terminate"] = True
    
    return state


def update_population(state: Dict[str, Any]) -> Dict[str, Any]:
    """Update population with new evaluated programs."""
    logger.info("Updating population")
    
    # Simple population update: keep best programs
    all_programs = state.get("current_population", []) + state.get("evaluated_programs", [])
    
    # Sort by score (descending)
    all_programs.sort(
        key=lambda p: p.scores.get("main_score", -float("inf")),
        reverse=True
    )
    
    # Keep top programs
    population_size = state.get("population_size", 5)
    state["current_population"] = all_programs[:population_size]
    
    # Clear temporary candidate storage
    state["candidate_programs"] = []
    state["evaluated_programs"] = []
    
    logger.info(f"Population updated. Size: {len(state['current_population'])}")
    
    return state


def check_termination(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check if evolution should terminate."""
    generation = state.get("generation", 0)
    max_generations = state.get("max_generations", 3)
    
    logger.info(f"Checking termination conditions (gen {generation}/{max_generations})")
    
    # Check generation limit
    if generation >= max_generations:
        logger.info("Reached maximum generations")
        state["should_terminate"] = True
        return state
    
    # Check for errors
    if state.get("error_message"):
        logger.warning(f"Terminating due to error: {state['error_message']}")
        state["should_terminate"] = True
        return state
    
    # Continue evolution
    state["generation"] = generation + 1
    logger.info(f"Continuing to generation {state['generation']}")
    
    return state


def extract_code_from_response(response: str, fallback_code: str) -> str:
    """Extract code from LLM response."""
    try:
        # Simple code extraction
        if "```python" in response:
            start = response.find("```python") + len("```python")
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        # Fallback: return the response as-is if it looks like code
        if any(keyword in response for keyword in ["def ", "class ", "import ", "return ", "if ", "for "]):
            return response.strip()
        
        return fallback_code
        
    except Exception:
        return fallback_code


# Demo functions
def get_initial_code() -> str:
    """Return initial code for optimization."""
    return '''
def target_function(x):
    """Function to optimize - we want to maximize this."""
    # EVOLVE-BLOCK-START main
    # Initial simple function
    return x * 2
    # EVOLVE-BLOCK-END

def main():
    result = target_function(5)
    return result
'''


def simple_evaluator(full_code: str, config: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Simple evaluator that runs the code and scores it."""
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
            {"result": result if 'result' in locals() else None, "execution": "success", "config": config}
        )
        
    except Exception as e:
        # Penalty for code that doesn't run
        return (
            {"main_score": -1000.0},
            {"error": str(e), "execution": "failed", "config": config}
        )


def run_mock_langgraph_demo(api_key: str) -> None:
    """Run the mock LangGraph demo."""
    logger.info("=== Starting Mock LangGraph POC Demo ===")
    
    try:
        # Create LLM settings
        llm_settings = [LLMSettings(
            model_name="gemini/gemini-1.5-flash",
            generation_params={
                "temperature": 0.7,
                "max_tokens": 1000
            }
        )]
        
        # Create mock workflow
        workflow = create_mock_evolution_workflow()
        
        # Create initial state
        initial_state = {
            "stage_name": "poc_stage",
            "max_generations": 2,
            "generation": 0,
            "llm_settings": llm_settings,
            "stage_config": {
                "evaluator_config": {
                    "evaluator_fn": simple_evaluator
                }
            },
            "api_key": api_key,
            "initial_codebase_code": get_initial_code(),
            "task_description": "Simple POC evolution",
            "current_population": [],
            "candidate_programs": [],
            "evaluated_programs": [],
            "best_program": None,
            "should_terminate": False,
            "error_message": None,
            "population_size": 3
        }
        
        # Execute workflow
        final_state = None
        for state_update in workflow.stream(initial_state, {}):
            final_state = list(state_update.values())[0]
            
            # Log progress
            node_name = list(state_update.keys())[0]
            gen = final_state.get('generation', 0)
            best_score = 'N/A'
            if final_state.get('best_program'):
                best_score = final_state['best_program'].scores.get('main_score', 'N/A')
            logger.info(f"  Node: {node_name} | Gen: {gen} | Best: {best_score}")
        
        # Create stage output
        status = "FAILED" if final_state.get("error_message") else "COMPLETED"
        message = final_state.get("error_message", "POC stage completed successfully")
        
        result = StageOutput(
            stage_name="poc_stage",
            status=status,
            message=message,
            best_program=final_state.get("best_program"),
            final_population=final_state.get("current_population", []),
            artifacts={}
        )
        
        # Display results
        logger.info("=== Mock LangGraph POC Results ===")
        logger.info(f"Status: {result.status}")
        logger.info(f"Message: {result.message}")
        
        if result.best_program:
            best = result.best_program
            logger.info(f"Best Program ID: {best.id}")
            logger.info(f"Best Score: {best.scores.get('main_score', 'N/A')}")
            logger.info(f"Generation: {best.generation}")
            logger.info(f"Best Code:\n{best.code_str}")
        else:
            logger.warning("No best program found!")
        
        logger.info(f"Final Population Size: {len(result.final_population)}")
        
        # Show population diversity
        if result.final_population:
            scores = [p.scores.get('main_score', -float('inf')) for p in result.final_population]
            logger.info(f"Population scores: {scores}")
        
        return result
        
    except Exception as e:
        logger.error(f"Mock LangGraph Demo failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # Your provided API key
    API_KEY = "AIzaSyBdiqeV2FSL63Y_rlazDyQEpORimWTt5-M"
    
    logger.info("Testing Mock LangGraph POC...")
    result = run_mock_langgraph_demo(API_KEY)
    
    if result:
        logger.info("Mock LangGraph POC Demo completed successfully!")
        logger.info("\n" + "="*50)
        logger.info("CONCEPT DEMONSTRATION:")
        logger.info("This shows how LangGraph would work with:")
        logger.info("- Node-based workflow execution")
        logger.info("- State management between nodes")
        logger.info("- Streaming execution with checkpoints")
        logger.info("- Integration with existing fluent API")
        logger.info("="*50)
    else:
        logger.error("Mock LangGraph POC Demo failed!")
