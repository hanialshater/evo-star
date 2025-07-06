import numpy as np  # Ensure numpy is imported
from typing import List, Optional, Dict, Tuple, Any
from ..core_types import Program  # Relative import
from .database_abc import BaseProgramDatabase  # Relative import


class MAPElitesDatabase(BaseProgramDatabase):
    def __init__(
        self,
        feature_definitions: List[Dict[str, Any]],
        context_program_capacity: int = 10,
    ):
        self.feature_definitions = feature_definitions
        self.num_dimensions = len(feature_definitions)
        self.context_program_capacity = context_program_capacity
        self.map_elites: Dict[Tuple[int, ...], Program] = {}
        self.recent_good_programs: List[Program] = []  # Fallback & context source
        if not self.feature_definitions:
            raise ValueError("MAPElitesDatabase requires feature_definitions.")
        self.bin_edges = [
            np.linspace(fd["min_val"], fd["max_val"], fd["bins"] + 1)
            for fd in self.feature_definitions
        ]
        self.num_bins_per_dim = [fd["bins"] for fd in self.feature_definitions]
        print(
            f"MAPElitesDatabase: {self.num_dimensions} feature dimensions, map cells: {np.prod(self.num_bins_per_dim)}"
        )

    def _get_bin_indices(
        self, features: Tuple[float, ...]
    ) -> Optional[Tuple[int, ...]]:
        if len(features) != self.num_dimensions:
            return None
        indices = [
            int(
                np.clip(
                    np.digitize([ft_val], self.bin_edges[i])[0] - 1,
                    0,
                    self.num_bins_per_dim[i] - 1,
                )
            )
            for i, ft_val in enumerate(features)
        ]
        return tuple(indices)

    def add_program(self, program: Program):
        if program.features is None:
            if program.scores.get("main_score", -float("inf")) > -50:
                self._add_to_recent_good_programs(program)
            return
        bin_idx = self._get_bin_indices(program.features)
        if bin_idx is None:
            return
        current_elite = self.map_elites.get(bin_idx)
        prog_score = program.scores.get("main_score", -float("inf"))
        if current_elite is None or prog_score > current_elite.scores.get(
            "main_score", -float("inf")
        ):
            self.map_elites[bin_idx] = program
            print(
                f"Program {program.id} (Score: {prog_score:.4f}, Feats: {program.features}, Bin: {bin_idx}) new elite."
            )
            self._add_to_recent_good_programs(program)

    def _add_to_recent_good_programs(self, program: Program):
        self.recent_good_programs.append(program)
        self.recent_good_programs.sort(
            key=lambda p: p.scores.get("main_score", -float("inf")), reverse=True
        )
        self.recent_good_programs = self.recent_good_programs[
            : self.context_program_capacity * 2
        ]  # Keep a bit more

    def get_program(self, program_id: str) -> Optional[Program]:
        for elite in self.map_elites.values():
            if elite.id == program_id:
                return elite
        for prog in self.recent_good_programs:
            if prog.id == program_id:
                return prog
        return None

    def get_all_programs(self) -> List[Program]:
        return list(self.map_elites.values())

    def select_parent_program(
        self, strategy: str = "random_elite"
    ) -> Optional[Program]:
        if not self.map_elites:
            if self.recent_good_programs:
                return np.random.choice(self.recent_good_programs)
            return None
        elites = list(self.map_elites.values())
        if strategy == "random_elite":
            selected = np.random.choice(elites) if elites else None
        elif strategy == "highest_score_elite":
            selected = max(
                elites,
                key=lambda p: p.scores.get("main_score", -float("inf")),
                default=None,
            )
        else:
            selected = np.random.choice(elites) if elites else None  # Default to random
        if selected:
            print(
                f"Selected parent ({strategy}): {selected.id} from bin {self._get_bin_indices(selected.features) if selected.features else 'N/A'}."
            )
        return selected

    def get_context_programs(
        self,
        num_programs: int,
        exclude_id: Optional[str] = None,
        strategy: str = "diverse_elites_from_map",
    ) -> List[Program]:
        if num_programs <= 0:
            return []
        candidates = []
        elites = [p for p in self.map_elites.values() if p.id != exclude_id]
        if strategy == "diverse_elites_from_map":
            candidates = (
                list(
                    np.random.choice(
                        elites, size=min(num_programs, len(elites)), replace=False
                    )
                )
                if elites
                else []
            )
        elif strategy == "best_elites_from_map":
            elites.sort(
                key=lambda p: p.scores.get("main_score", -float("inf")), reverse=True
            )
            candidates = elites[:num_programs]
        else:  # Fallback to recent good programs
            fallbacks = [p for p in self.recent_good_programs if p.id != exclude_id]
            fallbacks.sort(
                key=lambda p: p.scores.get("main_score", -float("inf")), reverse=True
            )
            candidates = fallbacks[:num_programs]
        if (
            len(candidates) < num_programs
        ):  # Supplement if not enough from primary strategy
            needed = num_programs - len(candidates)
            others = [
                p
                for p in self.recent_good_programs
                if p.id != exclude_id and p not in candidates
            ]
            others.sort(
                key=lambda p: p.scores.get("main_score", -float("inf")), reverse=True
            )
            candidates.extend(others[:needed])
        print(
            f"Selected {len(candidates)} context programs (strategy: {strategy}) excluding {exclude_id}."
        )
        return candidates

    def get_best_program(self) -> Optional[Program]:  # Best overall from map + recent
        all_contenders = list(self.map_elites.values()) + self.recent_good_programs
        if not all_contenders:
            return None
        return max(
            all_contenders,
            key=lambda p: p.scores.get("main_score", -float("inf")),
            default=None,
        )

    def __len__(self):
        return len(self.map_elites)
