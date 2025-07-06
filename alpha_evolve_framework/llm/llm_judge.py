# -*- coding: utf-8 -*-
import json
import re
import numpy as np
from typing import Dict, Any, Optional, List
from PIL import Image
import litellm
from ..core_types import Program, LLMSettings
from ..utils.logging_utils import logger


def numpy_converter(obj):
    """A custom converter for JSON to handle NumPy data types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class LLMJudge:
    def __init__(self, judge_llm_settings: LLMSettings, default_api_key: str):
        self.settings = judge_llm_settings
        self.api_key = judge_llm_settings.api_key or default_api_key
        if not self.api_key:
            raise ValueError("LLM Judge requires a valid API key.")
        logger.info(f"LLM Judge initialized with model: {self.settings.model_name}")

    def _build_judge_prompt(
        self,
        program: Program,
        task_description: str,
        image: Optional[Image.Image] = None,
    ) -> List[Any]:
        eval_details_json = json.dumps(
            program.eval_details, indent=2, default=numpy_converter
        )
        scores_json = json.dumps(program.scores, indent=2, default=numpy_converter)

        prompt_parts = [
            "You are an expert software engineer and algorithm designer, acting as a strict judge.",
            "Your task is to evaluate a proposed solution for the following problem:",
            f"\n--- TASK DESCRIPTION ---\n{task_description}\n",
            "\n--- PROPOSED SOLUTION ---\n",
            f"Program ID: {program.id}",
            f"Parent ID: {program.parent_id}",
            f"\n**Quantitative Scores (from script):**\n```json\n{scores_json}\n```",
            f"\n**Evaluation Details (from script):**\n```json\n{eval_details_json}\n```",
            f"\n**Generated Code Block ('{program.block_name}'):**\n```python\n{program.code_str}\n```",
        ]

        if image:
            prompt_parts.append("\n**Visual Output:**\n")
            prompt_parts.append(image)

        prompt_parts.extend(
            [
                "\n--- YOUR JUDGEMENT ---\n",
                "Based on all the information above, provide a critical analysis.",
                "1.  **Critique:** What are the strengths and weaknesses of this approach? Is it elegant, efficient, or a dead end?",
                "2.  **Suggestions:** Provide concrete, actionable suggestions for the next evolution.",
                "3.  **Score:** Assign a score from 0.0 (terrible) to 1.0 (perfect).",
                "\nFormat your response as a single, valid JSON object with the keys: 'critique', 'suggestions', and 'judge_score'. Do not include any other text or markdown formatting.",
            ]
        )

        return prompt_parts

    def judge_program(
        self,
        program: Program,
        task_description: str,
        image: Optional[Image.Image] = None,
    ) -> Optional[Dict[str, Any]]:
        prompt_parts = self._build_judge_prompt(program, task_description, image)

        # Convert prompt parts to text (skip images for now as they need special handling)
        text_parts = []
        for part in prompt_parts:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, Image.Image):
                # For now, we'll skip images as they require special handling in litellm
                text_parts.append("[Image would be displayed here]")

        prompt_text = "\n".join(text_parts)

        logger.info(
            f"Judging Program {program.id} with model {self.settings.model_name}..."
        )
        logger.debug(
            f"--- Full Judge Prompt for {program.id} ---\n{prompt_text}\n--- End of Judge Prompt ---"
        )

        try:
            # Prepare completion parameters
            completion_kwargs = {
                "model": self.settings.model_name,
                "messages": [{"role": "user", "content": prompt_text}],
                "api_key": self.api_key,
            }

            # Add generation parameters if provided
            if self.settings.generation_params:
                gen_params = self.settings.generation_params.copy()
                if "temperature" in gen_params:
                    completion_kwargs["temperature"] = gen_params["temperature"]
                if "max_output_tokens" in gen_params:
                    completion_kwargs["max_tokens"] = gen_params["max_output_tokens"]
                if "top_p" in gen_params:
                    completion_kwargs["top_p"] = gen_params["top_p"]

            response = litellm.completion(**completion_kwargs)
            raw_text = response.choices[0].message.content.strip()

            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if not json_match:
                logger.error(
                    f"Judge Error: No JSON object found in response for {program.id}."
                )
                return None

            feedback = json.loads(json_match.group(0))

            if (
                "critique" not in feedback
                or "suggestions" not in feedback
                or "judge_score" not in feedback
            ):
                logger.error(
                    f"Judge Error: JSON response for {program.id} is missing required keys."
                )
                return None

            logger.info(
                f"Judge for {program.id} Complete. Assigned Score: {feedback.get('judge_score')}"
            )
            return feedback

        except Exception as e:
            logger.error(
                f"LLM Judge API call or parsing failed for program {program.id}: {e}",
                exc_info=True,
            )
            return None
