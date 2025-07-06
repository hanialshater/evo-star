"""
Environment variable loader utility.
Loads environment variables from .env file and provides fallback values.
"""

import os
from typing import Optional


def load_env_file(env_file: str = ".env") -> None:
    """
    Load environment variables from a .env file.

    Args:
        env_file: Path to the .env file
    """
    if not os.path.exists(env_file):
        return

    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                os.environ[key] = value


def get_api_key(key_name: str, fallback: Optional[str] = None) -> str:
    """
    Get an API key from environment variables.

    Args:
        key_name: Name of the environment variable
        fallback: Fallback value if environment variable is not set

    Returns:
        API key value

    Raises:
        ValueError: If no API key is found and no fallback is provided
    """
    # Try to load from .env file first
    load_env_file()

    api_key = os.getenv(key_name, fallback)

    if not api_key or api_key == "your-api-key-here" or api_key.startswith("your-"):
        if fallback and fallback != "your-api-key-here":
            return fallback
        raise ValueError(
            f"API key '{key_name}' not found. "
            f"Please set it in your .env file or as an environment variable."
        )

    return api_key


def get_gemini_api_key(fallback: Optional[str] = None) -> str:
    """Get Gemini API key from environment."""
    return get_api_key("GEMINI_API_KEY", fallback)


def get_openai_api_key(fallback: Optional[str] = None) -> str:
    """Get OpenAI API key from environment."""
    return get_api_key("OPENAI_API_KEY", fallback)
