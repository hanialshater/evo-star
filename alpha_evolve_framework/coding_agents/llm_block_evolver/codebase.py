import re
from typing import Dict, List, Optional
from ...core_types import EvolveBlock  # Relative import


class Codebase:
    # Updated patterns to support both Python and HTML evolve block markers
    EVOLVE_BLOCK_START_PATTERN_LINE = r"^\s*(?:#|\<\!\-\-)\s*EVOLVE-BLOCK-START\s*(?P<name>[a-zA-Z0-9_.-]+)\s*(?:-->)?\s*$"
    EVOLVE_BLOCK_END_PATTERN_LINE = (
        r"^\s*(?:#|\<\!\-\-)\s*EVOLVE-BLOCK-END\s*(?:-->)?\s*$"
    )

    def __init__(self, initial_full_code: str):
        self.original_full_code: str = initial_full_code
        self.evolve_blocks: Dict[str, EvolveBlock] = {}
        self.code_template_parts: List[str] = []
        self._parse_initial_code(initial_full_code)

    def _parse_initial_code(self, full_code: str):
        lines = full_code.splitlines(keepends=True)
        current_fixed_lines = []
        current_evolve_block_name = None
        current_evolve_block_lines = []
        for line_idx, line_content in enumerate(lines):
            start_match = re.match(self.EVOLVE_BLOCK_START_PATTERN_LINE, line_content)
            end_match = re.match(self.EVOLVE_BLOCK_END_PATTERN_LINE, line_content)
            if start_match:
                if current_evolve_block_name is not None:
                    raise ValueError(f"L{line_idx+1}: Nested START")
                if current_fixed_lines:
                    self.code_template_parts.append("".join(current_fixed_lines))
                    current_fixed_lines = []
                current_evolve_block_name = start_match.group("name")
                if current_evolve_block_name in self.evolve_blocks:
                    raise ValueError(
                        f"L{line_idx+1}: Duplicate name: {current_evolve_block_name}"
                    )
                self.code_template_parts.append(line_content)
                current_evolve_block_lines = []
            elif end_match:
                if current_evolve_block_name is None:
                    raise ValueError(f"L{line_idx+1}: END without START")
                block_code_content = "".join(current_evolve_block_lines)
                self.evolve_blocks[current_evolve_block_name] = EvolveBlock(
                    current_evolve_block_name, block_code_content, block_code_content
                )
                self.code_template_parts.append(current_evolve_block_name)
                self.code_template_parts.append(line_content)
                current_evolve_block_name = None
                current_evolve_block_lines = []
            elif current_evolve_block_name is not None:
                current_evolve_block_lines.append(line_content)
            else:
                current_fixed_lines.append(line_content)
        if current_evolve_block_name is not None:
            raise ValueError(f"Unterminated block: {current_evolve_block_name}")
        if current_fixed_lines:
            self.code_template_parts.append("".join(current_fixed_lines))
        if not full_code and not self.code_template_parts:
            pass
        elif (
            full_code
            and not self.evolve_blocks
            and len(self.code_template_parts) == 1
            and self.code_template_parts[0] == full_code
        ):
            pass
        elif full_code and not self.evolve_blocks and not self.code_template_parts:
            self.code_template_parts.append(full_code)

    def get_block(self, name: str) -> Optional[EvolveBlock]:
        return self.evolve_blocks.get(name)

    def get_block_names(self) -> List[str]:
        return list(self.evolve_blocks.keys())

    def update_block_code(self, name: str, new_code: str):
        if name not in self.evolve_blocks:
            raise ValueError(f"Block '{name}' not found.")
        self.evolve_blocks[name].update_code(new_code)

    def reconstruct_full_code(self) -> str:
        return "".join(
            [
                (
                    self.evolve_blocks[part].current_code
                    if part in self.evolve_blocks
                    else part
                )
                for part in self.code_template_parts
            ]
        )

    def __repr__(self):
        return f"Codebase(blocks={list(self.evolve_blocks.keys())}, template_parts_count={len(self.code_template_parts)})"
