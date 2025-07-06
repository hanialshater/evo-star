"""
Evolution backends for different execution strategies.
"""

from .base_backend import EvolutionBackend
from .local_python_backend import LocalPythonBackend

# LangGraph backend - optional import
try:
    from .langgraph_backend import LangGraphBackend
except ImportError:
    # LangGraph not available - create a placeholder
    class LangGraphBackend:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LangGraph backend not available. Install with: pip install langgraph"
            )


__all__ = [
    "EvolutionBackend",
    "LocalPythonBackend",
    "LangGraphBackend",
]
