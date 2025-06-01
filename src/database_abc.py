import abc
from typing import List, Optional
from .core_types import Program # Relative import

class BaseProgramDatabase(abc.ABC):
    @abc.abstractmethod
    def add_program(self, program: Program): pass

    @abc.abstractmethod
    def get_program(self, program_id: str) -> Optional[Program]: pass

    @abc.abstractmethod
    def get_all_programs(self) -> List[Program]: pass # Might return all elites, or all in population

    @abc.abstractmethod
    def select_parent_program(self, strategy: str = "best") -> Optional[Program]: pass

    @abc.abstractmethod
    def get_context_programs(self, num_programs: int, exclude_id: Optional[str] = None, strategy: str = "best_alternate") -> List[Program]: pass

    @abc.abstractmethod
    def get_best_program(self) -> Optional[Program]: pass # Overall best based on main_score

    @abc.abstractmethod
    def __len__(self) -> int: pass # Number of elites, or population size

print("alpha_evolve_framework/database_abc.py written")
