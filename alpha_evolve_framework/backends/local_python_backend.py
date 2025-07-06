"""
Local Python backend using MainLoopOrchestrator.
"""

import dataclasses
import logging
from typing import Dict, List, Optional, Any, Callable

from .base_backend import EvolutionBackend
from ..core_types import Program, StageOutput, LLMSettings
from ..config import RunConfiguration
from ..codebase import Codebase
from ..functional_evaluator import FunctionalEvaluator
from ..prompt_engine import PromptEngine
from ..llm_manager import LLMManager
from ..llm_judge import LLMJudge
from ..llm_block_evolver import LLMBlockEvolver
from .local_python_orchestrator import MainLoopOrchestrator
from ..map_elites_database import MAPElitesDatabase
from ..simple_program_database import SimpleProgramDatabase

logger = logging.getLogger(__name__)


class LocalPythonBackend(EvolutionBackend):
    """Backend using local Python execution with MainLoopOrchestrator."""

    def __init__(self, api_key: str, generation_logger: Optional[Callable] = None):
        """Initialize LocalPython backend."""
        super().__init__(api_key)
        self.generation_logger = generation_logger
        self.llm_judge_config = None

    def is_available(self) -> bool:
        """LocalPython backend is always available."""
        return True

    def get_backend_name(self) -> str:
        """Get backend name."""
        return "local_python"

    def set_llm_judge_config(self, llm_judge_config: Dict[str, Any]):
        """Set LLM judge configuration."""
        self.llm_judge_config = llm_judge_config

    def run_stage(
        self,
        stage_config: Dict[str, Any],
        initial_codebase_code: str,
        evaluator_fn,
        evaluator_config: Dict[str, Any],
        initial_population: Optional[List[Program]] = None,
    ) -> StageOutput:
        """Run a single evolution stage using MainLoopOrchestrator."""

        # Validate configuration
        self.validate_stage_config(stage_config)

        logger.info(
            f"Running stage '{stage_config['name']}' with {self.get_backend_name()} backend"
        )

        # Create RunConfiguration from stage config
        run_config_fields = {f.name for f in dataclasses.fields(RunConfiguration)}
        params_for_run_config = {
            key: stage_config[key] for key in stage_config if key in run_config_fields
        }
        run_config = RunConfiguration(**params_for_run_config)

        # Create components
        initial_codebase = Codebase(initial_codebase_code)
        evaluator = FunctionalEvaluator(
            stage_config["name"], evaluator_fn, evaluator_config
        )

        task_description = stage_config.get("task_description", "")
        prompt_engine = PromptEngine(
            task_description, stage_config.get("problem_specific_instructions")
        )

        llm_manager = LLMManager(
            self.api_key,
            run_config.llm_ensemble,
            min_inter_request_delay_sec=stage_config.get(
                "min_inter_request_delay_sec", 1.1
            ),
        )

        # Create LLM judge if configured
        judge = None
        if self.llm_judge_config:
            judge = LLMJudge(self.llm_judge_config["settings"], self.api_key)

        # Create optimizers
        optimizers = []
        num_islands = run_config.num_islands if run_config.use_island_model else 1

        for island_id in range(num_islands):
            # Database selection
            use_map_elites = stage_config.get("use_map_elites", False)
            if use_map_elites:
                # For MAP-Elites, we need feature definitions
                feature_definitions = stage_config.get("feature_definitions")
                if feature_definitions:
                    db = MAPElitesDatabase(feature_definitions)
                else:
                    logger.warning(
                        "MAP-Elites requested but no feature_definitions provided, using SimpleProgramDatabase"
                    )
                    db = SimpleProgramDatabase(run_config.population_size)
            else:
                db = SimpleProgramDatabase(run_config.population_size)

            # Feature extractor
            feature_extractor = None
            if "feature_extractor_fn" in stage_config:
                feature_extractor_fn = stage_config["feature_extractor_fn"]
                if feature_extractor_fn:
                    feature_extractor = (
                        lambda details, p_conf, f_defs: feature_extractor_fn(
                            details, evaluator_config
                        )
                    )

            # Create optimizer
            optimizer = LLMBlockEvolver(
                initial_codebase=initial_codebase,
                llm_manager=llm_manager,
                program_database=db,
                prompt_engine=prompt_engine,
                evaluator=evaluator,
                run_config=run_config,
                feature_definitions=stage_config.get("feature_definitions"),
                feature_extractor_fn=feature_extractor,
                problem_specific_feature_configs=evaluator_config,
                island_id=island_id,
            )
            optimizers.append(optimizer)

        # Create orchestrator
        orchestrator = MainLoopOrchestrator(
            optimizers=optimizers,
            evaluator=evaluator,
            run_config=run_config,
            initial_codebase=initial_codebase,
            generation_logger=self.generation_logger,
            llm_judge=judge,
            judge_visual_generator=(
                self.llm_judge_config.get("visual_generator_fn")
                if self.llm_judge_config
                else None
            ),
            judge_feedback_combiner=(
                self.llm_judge_config.get("feedback_combiner_fn")
                if self.llm_judge_config
                else None
            ),
            task_description=task_description,
            stage_name=stage_config["name"],
        )

        # Run evolution
        return orchestrator.evolve(initial_population=initial_population)
