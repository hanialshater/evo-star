# -*- coding: utf-8 -*-
from typing import Tuple, Dict, Any, Callable
from .evaluator_abc import BaseEvaluator
from ..utils.logging_utils import logger
import multiprocessing as mp


def _execute_in_process(
    eval_fn: Callable,
    program_id: str,
    code: str,
    generation: int,
    config: Dict[str, Any],
    result_queue: mp.Queue,
):
    """
    A target function to be run in a separate process.
    Executes the user's evaluation function and puts the result in a queue.
    This isolates the main process from errors or hangs in the evaluation.
    """
    try:
        # This is where the potentially unsafe/long-running code is executed
        scores, details = eval_fn(program_id, code, generation, **config)
        result_queue.put((scores, details))
    except Exception as e:
        # If any exception occurs in the user's code, capture it
        logger.error(f"Exception in evaluation subprocess: {e}", exc_info=True)
        error_scores = {"main_score": -1000.0}
        error_details = {"error_message": f"Evaluation subprocess failed: {e}"}
        result_queue.put((error_scores, error_details))


class FunctionalEvaluator(BaseEvaluator):
    """An adapter to use a simple function as an evaluator within the framework."""

    def __init__(self, problem_name: str, eval_fn: Callable, config: Dict[str, Any]):
        super().__init__(problem_name)
        self._eval_fn = eval_fn
        self._config = config

    def evaluate(
        self,
        program_id: str,
        full_code_to_evaluate: str,
        program_generation: int,
        timeout_seconds: int,  # <-- Receive timeout from orchestrator
        stage: int = 0,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Calls the wrapped evaluation function in a separate process with a timeout.
        """
        runtime_config = self._config.copy()
        runtime_config.update(
            {
                "program_generation": program_generation,
                "evaluation_stage": stage,
            }
        )

        result_queue = mp.Queue()
        process = mp.Process(
            target=_execute_in_process,
            args=(
                self._eval_fn,
                program_id,
                full_code_to_evaluate,
                program_generation,
                runtime_config,
                result_queue,
            ),
        )

        logger.info(
            f"Starting evaluation for {program_id} in a subprocess with a {timeout_seconds}s timeout."
        )
        process.start()
        process.join(timeout=timeout_seconds)

        if process.is_alive():
            # Process is still running, meaning it timed out.
            logger.warning(
                f"Evaluation for {program_id} timed out after {timeout_seconds} seconds. Terminating process."
            )
            process.terminate()  # Send SIGTERM
            process.join()  # Wait for termination to complete

            scores = {"main_score": -2000.0}  # Special score for timeouts
            details = {
                "error_message": f"Evaluation timed out after {timeout_seconds} seconds."
            }
            return scores, details

        # Process finished in time, get the result.
        logger.info(f"Evaluation for {program_id} completed within the time limit.")
        scores, details = result_queue.get()
        return scores, details
