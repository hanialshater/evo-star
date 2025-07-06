# -*- coding: utf-8 -*-
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple

from .backends import EvolutionBackend, LocalPythonBackend, LangGraphBackend
from .core_types import Program, LLMSettings, StageOutput
from .logging_utils import setup_logger


class EvoAgent:
    """A readable, fluent API for configuring and running staged evolutionary experiments."""

    def __init__(self, api_key: str, log_level=logging.INFO):
        self._api_key = api_key
        self._stages_config: List[Dict[str, Any]] = []
        self._initial_code_fn: Optional[Callable[[], str]] = None
        self._evaluator_fn: Optional[Callable] = None
        self._evaluator_config: Dict[str, Any] = {}
        self._feature_extractor_fn: Optional[Callable] = None
        self._feature_definitions_fn: Optional[Callable] = None
        self._generation_logger: Optional[Callable] = None
        self._llm_judge_config: Optional[Dict[str, Any]] = None
        self._use_langgraph: bool = False
        self.logger = setup_logger(level=log_level)

    def define_problem(
        self,
        initial_code_fn: Callable[[], str],
        evaluator_fn: Callable,
        evaluator_config: Optional[Dict[str, Any]] = None,
        feature_extractor_fn: Optional[Callable] = None,
        feature_definitions_fn: Optional[Callable] = None,
    ):
        self.logger.info(f"Problem defined with evaluator: '{evaluator_fn.__name__}'")
        self._initial_code_fn = initial_code_fn
        self._evaluator_fn = evaluator_fn
        self._evaluator_config = evaluator_config or {}
        self._feature_extractor_fn = feature_extractor_fn
        self._feature_definitions_fn = feature_definitions_fn
        return self

    def with_generation_logger(self, logger_fn: Callable):
        self.logger.info(f"Attached generation logger: '{logger_fn.__name__}'")
        self._generation_logger = logger_fn
        return self

    def with_llm_judge(
        self,
        judge_llm_settings: LLMSettings,
        visual_generator_fn: Optional[Callable[[Dict], Optional[Any]]] = None,
        feedback_combiner_fn: Optional[
            Callable[[Dict, Dict, Dict], Tuple[Dict, Dict]]
        ] = None,
    ):
        self.logger.info(
            f"LLM Judge configured with model: '{judge_llm_settings.model_name}'"
        )
        self._llm_judge_config = {
            "settings": judge_llm_settings,
            "visual_generator_fn": visual_generator_fn,
            "feedback_combiner_fn": feedback_combiner_fn,
        }
        return self

    def add_stage(
        self, name: str, max_generations: int, llm_settings: List[LLMSettings], **kwargs
    ):
        if not self._initial_code_fn:
            raise ValueError(
                "A problem must be defined via .define_problem() before adding a stage."
            )
        self.logger.info(f"Adding stage '{name}' for {max_generations} generations.")
        stage_config = {
            "name": name,
            "max_generations": max_generations,
            "llm_ensemble": llm_settings,
            **kwargs,
        }
        self._stages_config.append(stage_config)
        return self

    def use_langgraph(self):
        """Enable LangGraph backend for workflow orchestration instead of MainLoopOrchestrator."""
        self.logger.info("Enabling LangGraph backend for workflow orchestration")
        self._use_langgraph = True
        return self

    def run(self) -> Optional[StageOutput]:
        self.logger.info("--- Starting Evolution Pipeline ---")
        last_stage_output: Optional[StageOutput] = None

        for i, stage_config in enumerate(self._stages_config):
            self.logger.info(
                f"--- Pipeline Stage {i+1}/{len(self._stages_config)}: '{stage_config['name']}' ---"
            )

            # Create appropriate backend
            if self._use_langgraph:
                backend = LangGraphBackend(self._api_key)
            else:
                backend = LocalPythonBackend(self._api_key, self._generation_logger)
                if self._llm_judge_config:
                    backend.set_llm_judge_config(self._llm_judge_config)

            # Add feature extractor and definitions to stage config if present
            if self._feature_extractor_fn:
                stage_config["feature_extractor_fn"] = self._feature_extractor_fn
            if self._feature_definitions_fn:
                stage_config["feature_definitions"] = self._feature_definitions_fn()

            # Run the stage
            try:
                initial_population_for_this_stage = (
                    last_stage_output.final_population if last_stage_output else None
                )

                last_stage_output = backend.run_stage(
                    stage_config=stage_config,
                    initial_codebase_code=self._initial_code_fn(),
                    evaluator_fn=self._evaluator_fn,
                    evaluator_config=self._evaluator_config,
                    initial_population=initial_population_for_this_stage,
                )

                if last_stage_output.status != "COMPLETED":
                    self.logger.error(
                        f"Pipeline halting because stage '{last_stage_output.stage_name}' failed with message: {last_stage_output.message}"
                    )
                    break

            except Exception as e:
                self.logger.error(
                    f"Stage '{stage_config['name']}' failed with exception: {e}"
                )
                break

        self.logger.info("--- Evolution Pipeline Finished ---")
        if last_stage_output and last_stage_output.best_program:
            best_program_overall = last_stage_output.best_program
            self.logger.info(
                f"Final best program: {best_program_overall.id}, Score: {best_program_overall.scores.get('main_score', 'N/A')}"
            )
        else:
            self.logger.warning("Pipeline finished but no best program was found.")

        return last_stage_output
