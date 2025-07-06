#!/usr/bin/env python3
"""
Setup script for the Alpha Evolve Framework.
This script helps users set up their environment and API keys.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False

    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install the framework dependencies."""
    print("📦 Installing dependencies...")

    try:
        # Install in development mode
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            check=True,
            capture_output=True,
        )
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def setup_env_file():
    """Setup .env file with API keys."""
    env_file = Path(".env")

    if env_file.exists():
        print("📄 .env file already exists")
        response = input("Do you want to update it? (y/N): ").strip().lower()
        if response != "y":
            return True

    print("\n🔑 Setting up API keys...")
    print("The framework supports multiple LLM providers:")
    print("- Google Gemini (recommended)")
    print("- OpenAI GPT models")
    print("- Others can be added later")

    gemini_key = input("\nEnter your Gemini API key (or press Enter to skip): ").strip()
    openai_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()

    env_content = """# Environment variables for Alpha Evolve Framework
# Get your API keys from:
# - Gemini: https://ai.google.dev/
# - OpenAI: https://platform.openai.com/api-keys

"""

    if gemini_key:
        env_content += f"GEMINI_API_KEY={gemini_key}\n"
    else:
        env_content += "GEMINI_API_KEY=your-gemini-api-key-here\n"

    if openai_key:
        env_content += f"OPENAI_API_KEY={openai_key}\n"
    else:
        env_content += "OPENAI_API_KEY=your-openai-api-key-here\n"

    env_content += """
# Other API keys can be added here
# Example:
# ANTHROPIC_API_KEY=your-anthropic-key-here
# COHERE_API_KEY=your-cohere-key-here
"""

    with open(env_file, "w") as f:
        f.write(env_content)

    print(f"✅ .env file created at {env_file.absolute()}")

    if not gemini_key and not openai_key:
        print("\n⚠️  No API keys provided. Please edit .env file to add your keys.")
        print("   You can get API keys from:")
        print("   - Gemini: https://ai.google.dev/")
        print("   - OpenAI: https://platform.openai.com/api-keys")

    return True


def run_basic_test():
    """Run a basic test to verify the installation."""
    print("\n🧪 Running basic tests...")

    try:
        # Import the framework
        from alpha_evolve_framework import EvoAgent

        print("✅ Framework import successful")

        # Test environment loading
        from alpha_evolve_framework.utils import get_gemini_api_key

        try:
            get_gemini_api_key()
            print("✅ API key configuration working")
        except ValueError:
            print("⚠️  API key not configured (this is expected if you haven't set it)")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main setup process."""
    print("🚀 Alpha Evolve Framework Setup")
    print("=" * 40)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        sys.exit(1)

    # Setup environment file
    if not setup_env_file():
        sys.exit(1)

    # Run basic test
    if not run_basic_test():
        print("\n❌ Setup completed with issues")
        sys.exit(1)

    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file to add your API keys if not done already")
    print("2. Run test scripts to verify everything works:")
    print("   - python tests/test_new_backend_architecture.py")
    print("   - python tests/test_fluent_langgraph_integration.py")
    print("3. Check out the examples/ directory for more complex examples")
    print("4. Read the documentation for advanced usage")
    print("\nHappy evolving! 🧬")


if __name__ == "__main__":
    main()
