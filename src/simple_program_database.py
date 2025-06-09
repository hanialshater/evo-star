import pickle
from typing import List, Optional, Dict, Type # Make sure Type is imported
from .core_types import Program # Relative import
from .database_abc import BaseProgramDatabase, DB # Make sure DB is imported

class SimpleProgramDatabase(BaseProgramDatabase):
    def __init__(self, population_size: int = 10):
        self.programs: Dict[str, Program] = {}
        self.population_size = population_size
        print(f"SimpleProgramDatabase initialized with population_size: {population_size}")

    def add_program(self, program: Program):
        if program.id in self.programs: print(f"Warning: Program '{program.id}' exists. Overwriting.")
        self.programs[program.id] = program
        print(f"Added {program.id} (Score: {program.scores.get('main_score',-float('inf')):.4f}). Total: {len(self.programs)}")
        self._prune()

    def get_program(self, program_id: str) -> Optional[Program]: return self.programs.get(program_id)
    def get_all_programs(self) -> List[Program]: return list(self.programs.values())

    def _prune(self):
        if len(self.programs) > self.population_size:
            sorted_progs = sorted(list(self.programs.values()), key=lambda p: p.scores.get('main_score',-float('inf')), reverse=True)
            self.programs = {p.id: p for p in sorted_progs[:self.population_size]}
            print(f"Pruned SimpleProgramDatabase to {len(self.programs)} programs.")

    def select_parent_program(self, strategy: str = "best") -> Optional[Program]:
        if not self.programs: return None
        progs = list(self.programs.values())
        if strategy == "best":
            selected = max(progs, key=lambda p: p.scores.get('main_score', -float('inf')), default=None)
        # Add other strategies like "roulette_wheel", "tournament" later
        else: selected = max(progs, key=lambda p: p.scores.get('main_score', -float('inf')), default=None) # Default to best
        if selected: print(f"Selected parent ({strategy}): {selected.id}")
        return selected

    def get_best_program(self) -> Optional[Program]:
        if not self.programs: return None
        return max(list(self.programs.values()), key=lambda p: p.scores.get('main_score',-float('inf')), default=None)

    def get_context_programs(self, num_programs: int, exclude_id: Optional[str] = None, strategy: str = "best_alternate") -> List[Program]:
        if num_programs <= 0: return []
        candidates = [p for pid, p in self.programs.items() if pid != exclude_id]
        if not candidates: return []
        # For simple DB, "best_alternate" is the main sensible strategy
        candidates.sort(key=lambda p: p.scores.get('main_score',-float('inf')), reverse=True)
        return candidates[:num_programs]

    def __len__(self): return len(self.programs)

    def save_checkpoint(self, filepath: str) -> None:
        data_to_save = {
            'programs': self.programs,
            'population_size': self.population_size,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"SimpleProgramDatabase checkpoint saved to {filepath}")

    @classmethod
    def load_checkpoint(cls: Type[DB], filepath: str) -> DB:
        with open(filepath, 'rb') as f:
            loaded_data = pickle.load(f)

        instance = cls(population_size=loaded_data['population_size'])

        instance.programs = loaded_data['programs']

        print(f"SimpleProgramDatabase checkpoint loaded from {filepath}")
        return instance

print("alpha_evolve_framework/simple_program_database.py written")
