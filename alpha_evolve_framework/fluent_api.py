# -*- coding: utf-8 -*-
import copy
import dataclasses
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple

from .llm_judge import LLMJudge
from .core_types import Program, LLMSettings, StageOutput
from .config import RunConfiguration
from .llm_manager import LLMManager
from .prompt_engine import PromptEngine
from .map_elites_database import MAPElitesDatabase
from .simple_program_database import SimpleProgramDatabase
from .llm_block_evolver import LLMBlockEvolver
from .orchestrator import MainLoopOrchestrator
from .codebase import Codebase
from .functional_evaluator import FunctionalEvaluator
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
        self.logger = setup_logger(level=log_level)

    def define_problem(self, initial_code_fn: Callable[[], str], evaluator_fn: Callable, evaluator_config: Optional[Dict[str, Any]] = None, feature_extractor_fn: Optional[Callable] = None, feature_definitions_fn: Optional[Callable] = None):
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

    def with_llm_judge(self,
                       judge_llm_settings: LLMSettings,
                       visual_generator_fn: Optional[Callable[[Dict], Optional[Any]]] = None,
                       feedback_combiner_fn: Optional[Callable[[Dict, Dict, Dict], Tuple[Dict, Dict]]] = None):
        self.logger.info(f"LLM Judge configured with model: '{judge_llm_settings.model_name}'")
        self._llm_judge_config = {
            "settings": judge_llm_settings,
            "visual_generator_fn": visual_generator_fn,
            "feedback_combiner_fn": feedback_combiner_fn
        }
        return self

    def add_stage(self, name: str, max_generations: int, llm_settings: List[LLMSettings], **kwargs):
        if not self._initial_code_fn:
            raise ValueError("A problem must be defined via .define_problem() before adding a stage.")
        self.logger.info(f"Adding stage '{name}' for {max_generations} generations.")
        stage_config = {"name": name, "max_generations": max_generations, "llm_ensemble": llm_settings, **kwargs}
        self._stages_config.append(stage_config)
        return self

    def run(self) -> Optional[StageOutput]:
        self.logger.info("--- Starting Evolution Pipeline ---")
        last_stage_output: Optional[StageOutput] = None
        run_config_fields = {f.name for f in dataclasses.fields(RunConfiguration)}

        for i, stage_config in enumerate(self._stages_config):
            self.logger.info(f"--- Pipeline Stage {i+1}/{len(self._stages_config)}: '{stage_config['name']}' ---")

            params_for_run_config = {key: stage_config[key] for key in stage_config if key in run_config_fields}
            run_config = RunConfiguration(**params_for_run_config)

            initial_codebase = Codebase(self._initial_code_fn())
            evaluator = FunctionalEvaluator(f"{stage_config['name']}", self._evaluator_fn, self._evaluator_config)

            task_desc = stage_config.get("task_description", "")
            prompt_engine = PromptEngine(task_desc, stage_config.get("problem_specific_instructions"))
            llm_manager = LLMManager(self._api_key, run_config.llm_ensemble, min_inter_request_delay_sec=stage_config.get("min_inter_request_delay_sec", 1.1))

            judge = None
            if self._llm_judge_config:
                judge = LLMJudge(self._llm_judge_config["settings"], self._api_key)

            optimizers = []
            for island_id in range(run_config.num_islands if run_config.use_island_model else 1):
                use_map_elites = stage_config.get("use_map_elites", False)
                feature_defs = self._feature_definitions_fn() if self._feature_definitions_fn else None
                db = MAPElitesDatabase(feature_defs) if use_map_elites and feature_defs else SimpleProgramDatabase(run_config.population_size)

                feature_extractor = lambda details, p_conf, f_defs: self._feature_extractor_fn(details, self._evaluator_config) if self._feature_extractor_fn else None

                optimizer = LLMBlockEvolver(
                    initial_codebase=initial_codebase,
                    llm_manager=llm_manager,
                    program_database=db,
                    prompt_engine=prompt_engine,
                    evaluator=evaluator,
                    run_config=run_config,
                    feature_definitions=feature_defs,
                    feature_extractor_fn=feature_extractor,
                    problem_specific_feature_configs=self._evaluator_config,
                    island_id=island_id
                )
                optimizers.append(optimizer)

            orchestrator = MainLoopOrchestrator(
                optimizers=optimizers,
                evaluator=evaluator,
                run_config=run_config,
                initial_codebase=initial_codebase,
                generation_logger=self._generation_logger,
                llm_judge=judge,
                judge_visual_generator=self._llm_judge_config.get("visual_generator_fn") if self._llm_judge_config else None,
                judge_feedback_combiner=self._llm_judge_config.get("feedback_combiner_fn") if self._llm_judge_config else None,
                task_description=task_desc,
                stage_name=stage_config['name']
            )

            initial_population_for_this_stage = last_stage_output.final_population if last_stage_output else None
            last_stage_output = orchestrator.evolve(initial_population=initial_population_for_this_stage)

            if last_stage_output.status != 'COMPLETED':
                self.logger.error(f"Pipeline halting because stage '{last_stage_output.stage_name}' failed with message: {last_stage_output.message}")
                break

        self.logger.info("--- Evolution Pipeline Finished ---")
        if last_stage_output and last_stage_output.best_program:
            best_program_overall = last_stage_output.best_program
            self.logger.info(f"Final best program: {best_program_overall.id}, Score: {best_program_overall.scores.get('main_score', 'N/A')}")
        else:
            self.logger.warning("Pipeline finished but no best program was found.")

        return last_stage_output
