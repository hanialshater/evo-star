"""
Base class for all coding agents in the evolutionary programming framework.

This module defines the abstract base class that all coding agents must inherit from.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from ..core_types import ProgramCandidate


class BaseAgent(ABC):
    """
    Abstract base class for all coding agents.

    This class defines the interface that all coding agents must implement
    for evolutionary programming tasks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.

        Args:
            config: Configuration dictionary for the agent
        """
        self.config = config or {}
        self.name = self.__class__.__name__

    @abstractmethod
    async def evolve(
        self,
        candidate: ProgramCandidate,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProgramCandidate:
        """
        Evolve a program candidate to improve it.

        Args:
            candidate: The candidate program to evolve
            task_description: Description of the task/improvement needed
            context: Additional context for the evolution process

        Returns:
            Evolved program candidate
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get the capabilities of this agent.

        Returns:
            List of capabilities this agent supports
        """
        pass

    def cleanup(self):
        """
        Clean up any resources used by the agent.

        This method should be called when the agent is no longer needed.
        """
        pass

    def __str__(self) -> str:
        return f"{self.name}(config={self.config})"

    def __repr__(self) -> str:
        return self.__str__()
