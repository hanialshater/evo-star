# Environment Setup

## API Key Configuration

For security reasons, all API keys must be provided via environment variables. **Never commit API keys to the repository.**

### Setting up your Gemini API Key

1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

2. Set the environment variable:

**On macOS/Linux:**
```bash
export GEMINI_API_KEY="your_actual_api_key_here"
```

**On Windows:**
```cmd
set GEMINI_API_KEY=your_actual_api_key_here
```

**For permanent setup, add to your shell profile:**
```bash
# Add to ~/.bashrc, ~/.zshrc, or ~/.profile
echo 'export GEMINI_API_KEY="your_actual_api_key_here"' >> ~/.zshrc
source ~/.zshrc
```

### Running Examples

Once your environment variable is set, you can run any of the examples:

```bash
# Test API integration
python test_litellm_integration.py

# Run circle packing demo
python examples/run_circle_packing_demo.py

# Run circle packing with fluent API
python examples/circle_packing/run_new_fluent_api.py

# Run city page evolution
python examples/city_page/run_city_evolution.py
```

### Jupyter Notebooks

For Jupyter notebooks, make sure the environment variable is available in your notebook environment:

```python
import os
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError('Please set the GEMINI_API_KEY environment variable')
```

## Security Best Practices

- ✅ Use environment variables for API keys
- ✅ Add sensitive files to .gitignore
- ❌ Never commit API keys in code
- ❌ Never commit .env files with real secrets
- ❌ Never share API keys in chat or documentation

## Troubleshooting

**Error: "Please set the GEMINI_API_KEY environment variable"**
- Make sure you've set the environment variable
- Restart your terminal/IDE after setting it
- Check the variable is set: `echo $GEMINI_API_KEY` (macOS/Linux) or `echo %GEMINI_API_KEY%` (Windows)

**Error: "API key not found" or authentication errors**
- Verify your API key is correct
- Check that your Google AI Studio account is properly set up
- Ensure the API key has the necessary permissions
