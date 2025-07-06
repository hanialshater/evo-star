# API Keys Setup Guide

This guide explains how to set up API keys for the Alpha Evolve Framework.

## Quick Setup

1. **Run the setup script:**
   ```bash
   python setup_environment.py
   ```

2. **Or manually create a `.env` file:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Supported LLM Providers

### Google Gemini (Recommended)
- Get your API key from: https://ai.google.dev/
- Set in `.env`: `GEMINI_API_KEY=your-key-here`

### OpenAI
- Get your API key from: https://platform.openai.com/api-keys
- Set in `.env`: `OPENAI_API_KEY=your-key-here`

### Other Providers
The framework can be extended to support other providers:
- Anthropic Claude
- Cohere
- Local models (Ollama, LM Studio)

## Environment Variables

The framework uses the following environment variables:

```bash
# Required for most examples
GEMINI_API_KEY=your-gemini-key-here

# Optional - for OpenAI models
OPENAI_API_KEY=your-openai-key-here

# Add other providers as needed
ANTHROPIC_API_KEY=your-anthropic-key-here
COHERE_API_KEY=your-cohere-key-here
```

## Security Notes

- ✅ **Never commit API keys to version control**
- ✅ **The `.env` file is already in `.gitignore`**
- ✅ **Use environment variables in production**
- ✅ **Rotate keys regularly**

## Usage in Code

The framework automatically loads API keys from the `.env` file:

```python
from alpha_evolve_framework import EvoAgent
from alpha_evolve_framework.utils import get_gemini_api_key

# Automatic API key loading
agent = EvoAgent()  # Uses GEMINI_API_KEY from .env

# Or explicit key loading
api_key = get_gemini_api_key()
agent = EvoAgent(api_key)
```

## Troubleshooting

### "API key not found" Error
1. Check that `.env` file exists
2. Verify the API key is correctly set
3. Ensure no extra spaces or quotes
4. Try running `python setup_environment.py` again

### Import Errors
1. Install the framework: `pip install -e .`
2. Check Python version (3.8+ required)
3. Verify virtual environment is activated

## Examples

All test scripts now use environment variables:
- `python tests/test_new_backend_architecture.py`
- `python tests/test_fluent_langgraph_integration.py`
- `python tests/simple_poc_demo.py`
- `python tests/test_langgraph_poc.py`

## Production Deployment

For production environments:
1. Set environment variables directly (don't use `.env` file)
2. Use secure key management systems
3. Implement key rotation
4. Monitor usage and costs

```bash
# Production environment variables
export GEMINI_API_KEY="your-production-key"
export OPENAI_API_KEY="your-production-key"
