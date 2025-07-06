# -*- coding: utf-8 -*-
import re
import numpy as np
from typing import Optional, Dict, Any, List
import litellm
import time
from ..core_types import LLMSettings
from ..utils.logging_utils import logger


class LLMManager:
    def __init__(
        self,
        default_api_key: str,
        llm_settings_list: Optional[List[LLMSettings]] = None,
        selection_strategy: str = "weighted_random",
        **kwargs,
    ):
        self.default_api_key = default_api_key
        self.llm_instances: List[Dict[str, Any]] = []
        self.selection_strategy = selection_strategy
        self.current_llm_index = 0

        self.min_inter_request_delay_sec: float = float(
            kwargs.get("min_inter_request_delay_sec", 1.1)
        )
        self.last_api_call_timestamp: float = 0.0
        logger.info(
            f"LLMManager: Minimum inter-request delay set to {self.min_inter_request_delay_sec:.2f}s."
        )

        if not self.default_api_key:
            raise ValueError("default_api_key must be provided.")

        # Set up litellm with the default API key
        litellm.set_verbose = False  # Set to True for debugging

        if not llm_settings_list:
            logger.info(
                "LLMManager: No LLM ensemble, using default 'gemini-1.5-flash-latest'."
            )
            llm_settings_list = [LLMSettings(model_name="gemini-1.5-flash-latest")]

        for settings in llm_settings_list:
            try:
                # For litellm, we don't need to create model objects, just store the settings
                self.llm_instances.append({"settings": settings})
                logger.info(f"Initialized model '{settings.model_name}'.")
            except Exception as e:
                logger.error(f"Error initializing model '{settings.model_name}': {e}")

        if not self.llm_instances:
            raise RuntimeError("LLMManager: No LLMs initialized.")
        logger.info(
            f"LLMManager ready with {len(self.llm_instances)} model(s). Strategy: '{self.selection_strategy}'."
        )

    def _select_llm(self) -> Dict[str, Any]:
        if not self.llm_instances:
            raise RuntimeError("No LLMs for selection.")
        if len(self.llm_instances) == 1:
            return self.llm_instances[0]

        models, weights = [inst for inst in self.llm_instances], [
            inst["settings"].selection_weight for inst in self.llm_instances
        ]

        # Normalize weights if they don't sum to 1
        total_weight = sum(weights)
        if not np.isclose(total_weight, 1.0):
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:  # If all weights are 0, use uniform
                weights = [1.0 / len(models)] * len(models)

        return np.random.choice(models, p=weights)

    def generate_code_modification(
        self, prompt: str, attempt_info: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_timestamp
        if time_since_last_call < self.min_inter_request_delay_sec:
            sleep_duration = self.min_inter_request_delay_sec - time_since_last_call
            if sleep_duration > 0:
                logger.info(
                    f"Delaying for {sleep_duration:.2f}s to respect inter-request delay."
                )
                time.sleep(sleep_duration)

        selected_llm = self._select_llm()
        settings = selected_llm["settings"]

        logger.info(
            f"Sending prompt to Worker LLM ({settings.model_name}). Full prompt in DEBUG mode."
        )
        logger.debug(
            f"--- Full Prompt for {settings.model_name} ---\n{prompt}\n--- End of Prompt ---"
        )

        try:
            # Prepare the API key for this model
            api_key = settings.api_key or self.default_api_key

            # Prepare generation parameters
            completion_kwargs = {
                "model": settings.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "api_key": api_key,
            }

            # Add generation parameters if provided
            if settings.generation_params:
                # Map common Google AI parameters to litellm parameters
                gen_params = settings.generation_params.copy()
                if "temperature" in gen_params:
                    completion_kwargs["temperature"] = gen_params["temperature"]
                if "max_output_tokens" in gen_params:
                    completion_kwargs["max_tokens"] = gen_params["max_output_tokens"]
                if "top_p" in gen_params:
                    completion_kwargs["top_p"] = gen_params["top_p"]
                if "top_k" in gen_params:
                    # litellm doesn't support top_k for all models, so we'll skip it
                    pass

            api_response = litellm.completion(**completion_kwargs)
            self.last_api_call_timestamp = time.time()

            response_text = api_response.choices[0].message.content

            if not response_text or not response_text.strip():
                logger.warning(
                    f"LLM ({settings.model_name}) returned an empty response."
                )
                return None

            if "<<<<<<<< SEARCH" in response_text:
                logger.info("Detected diff format in response.")
                return response_text.strip()

            patterns = [r"```python\n(.*?)\n```", r"```(?:[a-zA-Z]*\n)?(.*?)\n```"]
            for i, p in enumerate(patterns):
                match = re.search(p, response_text, re.DOTALL)
                if match and match.group(1).strip():
                    logger.info(f"Extracted code using regex pattern {i+1}.")
                    return match.group(1).strip()

            logger.warning(
                f"Could not extract structured code/diff from LLM response. Returning None."
            )
            logger.debug(f"--- Failed Raw Response ---\n{response_text}\n---")
            return None

        except Exception as e:
            self.last_api_call_timestamp = time.time()
            logger.error(
                f"LLM call failed for model {settings.model_name}: {e}", exc_info=True
            )
            return None
