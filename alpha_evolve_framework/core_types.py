import dataclasses
import time
import re
from typing import Dict, Any, Optional, List, Tuple

@dataclasses.dataclass
class Program:
    id: str
    code_str: str # For LLMBlockEvolver, this is the block's code. For other optimizers, could be different.
    block_name: Optional[str] = None # Relevant for block-based evolution
    scores: Dict[str, float] = dataclasses.field(default_factory=dict)
    generation: int = 0
    parent_id: Optional[str] = None
    timestamp: float = dataclasses.field(default_factory=time.time)
    eval_details: Dict[str, Any] = dataclasses.field(default_factory=dict)
    features: Optional[Tuple[float, ...]] = None # For MAP-Elites
    judge_feedback: Optional[Dict[str, Any]] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not self.scores:
            self.scores['main_score'] = -float('inf')

    def __repr__(self):
        feature_str = f", features={self.features}" if self.features is not None else ""
        judge_score_str = f", judge_score={self.scores.get('judge_score', 'N/A')}" if 'judge_score' in self.scores else ""
        return (f"Program(id='{self.id}', block='{self.block_name}', "
                f"score={self.scores.get('main_score', -float('inf')):.4f}{judge_score_str}, gen={self.generation}{feature_str})")

@dataclasses.dataclass
class EvolveBlock:
    name: str
    initial_code: str
    current_code: str

    def __post_init__(self):
        if self.current_code is None:
            self.current_code = self.initial_code

    def update_code(self, new_code: str):
        self.current_code = new_code

    def apply_structured_diff(self, diff_str: str) -> bool:
        print(f"\nAttempting to apply diff to block '{self.name}':")
        print(f"--- Received Diff String (stripped, first 500 chars): ---\n{diff_str.strip()[:500]}...\n---")
        normalized_diff_str = diff_str.replace('\r\n', '\n').strip()
        search_marker_str = "<<<<<<<< SEARCH"
        separator_marker_str = "========"
        replace_marker_str = ">>>>>>>>> REPLACE"
        search_start_idx = normalized_diff_str.find(search_marker_str)
        if search_start_idx == -1:
            print(f"Diff Parse Error: '{search_marker_str}' not found.")
            return False
        search_content_start_idx = search_start_idx + len(search_marker_str)
        separator_marker_pos = normalized_diff_str.find(separator_marker_str, search_content_start_idx)
        if separator_marker_pos == -1:
            print(f"Diff Parse Error: '{separator_marker_str}' not found after '{search_marker_str}'.")
            return False
        search_code = normalized_diff_str[search_content_start_idx:separator_marker_pos].strip()
        replace_content_start_idx = separator_marker_pos + len(separator_marker_str)
        replace_marker_actual_pos = normalized_diff_str.rfind(replace_marker_str, replace_content_start_idx)
        if replace_marker_actual_pos == -1 or replace_marker_actual_pos < replace_content_start_idx:
            print(f"Diff Parse Error: '{replace_marker_str}' not found after '{separator_marker_str}'.")
            print(f"--- Section where REPLACE marker was expected: ---\n{normalized_diff_str[replace_content_start_idx:].strip()[:300]}...\n---")
            return False
        replace_code = normalized_diff_str[replace_content_start_idx:replace_marker_actual_pos].strip()
        if replace_code.startswith("="):
            print("Diff Info: replace_code started with '=', attempting to strip it.")
            replace_code = replace_code.lstrip("= \t\n\r")
        normalized_current_code = self.current_code.replace('\r\n', '\n')
        if not search_code:
            print("Diff Error: Extracted SEARCH code is empty.")
            return False
        if search_code not in normalized_current_code:
            print(f"Diff Apply Error: SEARCH code not found in current code of block '{self.name}'.")
            return False
        new_block_code_normalized = normalized_current_code.replace(search_code, replace_code, 1)
        if new_block_code_normalized == normalized_current_code:
            print(f"Diff Warning: No change after applying diff to block '{self.name}'.")
        self.current_code = new_block_code_normalized
        print(f"Diff applied successfully to block '{self.name}'. New code len: {len(self.current_code)}")
        return True

    def __repr__(self):
        return (f"EvolveBlock(name='{self.name}', current_code_len={len(self.current_code)})")

@dataclasses.dataclass
class LLMSettings:
    model_name: str
    api_key: Optional[str] = None
    selection_weight: float = 1.0
    generation_params: Optional[Dict[str, Any]] = None
    def __post_init__(self):
        if self.selection_weight < 0:
            raise ValueError("LLMSettings.selection_weight must be non-negative.")

# --- NEWLY ADDED DATACLASS ---
@dataclasses.dataclass
class StageOutput:
    """A standardized data structure to hold the results of a single pipeline stage."""
    stage_name: str
    status: str  # e.g., 'COMPLETED', 'FAILED', 'TERMINATED_EARLY'
    message: str # A human-readable summary of the stage outcome.
    best_program: Optional[Program]
    final_population: List[Program] = dataclasses.field(default_factory=list)
    artifacts: Dict[str, Any] = dataclasses.field(default_factory=dict) # For file paths, etc.
