import re
import numpy as np
from typing import Optional, Dict, Any, List
import google.generativeai as genai
from .core_types import LLMSettings # Relative import

class LLMManager:
    def __init__(self, default_api_key: str, llm_settings_list: Optional[List[LLMSettings]] = None, selection_strategy: str = "weighted_random"):
        self.default_api_key = default_api_key
        self.llm_instances: List[Dict[str, Any]] = []
        self.selection_strategy = selection_strategy
        self.current_llm_index = 0
        if not self.default_api_key: raise ValueError("default_api_key must be provided.")
        try:
            print(f"LLMManager: Configuring genai globally with provided default_api_key.")
            genai.configure(api_key=self.default_api_key)
        except Exception as e: print(f"LLMManager: Warning - Error during genai.configure: {e}.")
        if not llm_settings_list:
            print("LLMManager: No LLM ensemble, using default 'gemini-1.5-flash-latest'.")
            llm_settings_list = [LLMSettings(model_name="gemini-1.5-flash-latest")]
        for settings in llm_settings_list:
            try:
                model_obj = genai.GenerativeModel(settings.model_name)
                self.llm_instances.append({'model_obj': model_obj, 'settings': settings})
                print(f"LLMManager: Initialized model '{settings.model_name}'.")
            except Exception as e: print(f"LLMManager: Error initializing model '{settings.model_name}': {e}")
        if not self.llm_instances: raise RuntimeError("LLMManager: No LLMs initialized.")
        if self.selection_strategy == "weighted_random" and self.llm_instances:
            total_weight = sum(inst['settings'].selection_weight for inst in self.llm_instances)
            if total_weight <= 0:
                print("LLMManager Warning: Weights sum to <=0, using uniform.")
                if self.llm_instances:
                    uniform_w = 1.0 / len(self.llm_instances)
                    for inst in self.llm_instances: inst['settings'].selection_weight = uniform_w
            elif not np.isclose(total_weight, 1.0):
                print(f"LLMManager: Normalizing weights (sum was {total_weight}).")
                for inst in self.llm_instances: inst['settings'].selection_weight /= total_weight
        print(f"LLMManager ready with {len(self.llm_instances)} model(s). Strategy: '{self.selection_strategy}'.")

    def _select_llm(self) -> Dict[str, Any]:
        if not self.llm_instances: raise RuntimeError("No LLMs for selection.")
        if len(self.llm_instances) == 1: return self.llm_instances[0]
        if self.selection_strategy == "weighted_random":
            models, weights = self.llm_instances, [inst['settings'].selection_weight for inst in self.llm_instances]
            if not np.isclose(sum(weights), 1.0): # Re-normalize if needed (defensive)
                weights = [w/sum(weights) if sum(weights) > 0 else 1.0/len(models) for w in weights]
            return models[np.random.choice(len(models), p=weights)]
        elif self.selection_strategy == "sequential_loop":
            inst = self.llm_instances[self.current_llm_index % len(self.llm_instances)]
            self.current_llm_index += 1; return inst
        else: return self.llm_instances[0]

    def generate_code_modification(self, prompt: str, attempt_info: Optional[Dict[str, Any]] = None) -> Optional[str]:
        selected_llm = self._select_llm()
        model_obj, settings = selected_llm['model_obj'], selected_llm['settings']
        print(f"\n--- Sending to LLM ({settings.model_name}, Strat: {self.selection_strategy}) ---")
        # print prompt snippet
        print(f"{prompt[:300]}...\n...\n...{prompt[-300:]}" if len(prompt) > 600 else prompt)
        print("--- End of Prompt ---")
        try:
            gen_cfg = None
            if settings.generation_params and isinstance(settings.generation_params, dict):
                try: gen_cfg = genai.types.GenerationConfig(**settings.generation_params)
                except Exception as e: print(f"LLMManager Warn: Bad gen_params {settings.generation_params}: {e}")
            response = model_obj.generate_content(prompt, generation_config=gen_cfg)
            if not response.candidates: print(f"LLM Warn ({settings.model_name}): No candidates."); return None
            full_text = "".join([part.text for part in response.parts if hasattr(part, 'text') and part.text])
            if not full_text: print(f"LLM Warn ({settings.model_name}): No text in response parts."); return None
            print(f"\n--- LLM Raw Response ({settings.model_name}, first 300): ---\n{full_text[:300]}...\n---")
            # Code extraction logic...
            patterns = [r"```python\n(.*?)\n```", r"```(?:[a-zA-Z]*\n)?(.*?)\n```", r"```python\n(.*)", r"```(?:[a-zA-Z]*\n)?(.*)"]
            for i, p in enumerate(patterns):
                match = re.search(p, full_text, re.DOTALL)
                if match: print(f"Extracted code using pattern {i+1}."); return match.group(1).strip()
            if "<<<<<<<< SEARCH" in full_text and "========" in full_text and ">>>>>>>>> REPLACE" in full_text:
                print("LLMManager Warn: Using full response as diff (no markdown)."); return full_text.strip()
            print(f"LLMManager Warn ({settings.model_name}): Could not extract code/diff."); return None
        except Exception as e: print(f"LLMManager Error ({settings.model_name}): {e}"); return None

print("alpha_evolve_framework/llm_manager.py written")
