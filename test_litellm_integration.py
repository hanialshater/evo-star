#!/usr/bin/env python3
"""
Simple test to verify the litellm integration works correctly.
"""
import os
import sys
from alpha_evolve_framework.core_types import LLMSettings
from alpha_evolve_framework.llm_manager import LLMManager
from alpha_evolve_framework.llm_judge import LLMJudge
from alpha_evolve_framework.logging_utils import setup_logger

def test_google_ai_studio():
    """Test Google AI Studio API with different model name formats."""
    import litellm
    
    # Set up logger
    logger = setup_logger()
    logger.info("Testing Google AI Studio integration...")
    
    api_key = "***REMOVED***"
    
    # Test different model name formats for Google AI Studio
    model_formats = [
        "gemini-1.5-flash",
        "gemini/gemini-1.5-flash", 
        "google/gemini-1.5-flash",
        "gemini-1.5-flash-latest"
    ]
    
    working_format = None
    for model_name in model_formats:
        logger.info(f"Testing model: {model_name}")
        try:
            # Set environment variable for Google AI Studio
            os.environ["GOOGLE_API_KEY"] = api_key
            
            response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": "Say hello"}],
                api_key=api_key,
                max_tokens=10
            )
            logger.info(f"✅ SUCCESS with {model_name}: {response.choices[0].message.content}")
            working_format = model_name
            break  # Found working format
        except Exception as e:
            logger.info(f"❌ FAILED with {model_name}: {e}")
    
    if working_format:
        logger.info(f"Working Google AI Studio model format: {working_format}")
        return True, working_format
    else:
        logger.error("No working Google AI Studio model format found")
        return False, None

def test_basic_functionality():
    """Test basic functionality without making API calls."""
    
    # Set up logger
    logger = setup_logger()
    logger.info("Testing litellm integration...")
    
    # Test LLM Settings
    settings = LLMSettings(
        model_name="gemini-1.5-flash-latest",
        generation_params={"temperature": 0.7, "max_output_tokens": 1000}
    )
    logger.info(f"LLMSettings created: {settings}")
    
    # Test LLM Manager (without API key - just construction)
    try:
        llm_manager = LLMManager(
            default_api_key="dummy-key-for-testing",
            llm_settings_list=[settings]
        )
        logger.info("LLMManager created successfully")
    except Exception as e:
        logger.error(f"Error creating LLMManager: {e}")
        return False
    
    # Test LLM Judge (without API key - just construction)
    try:
        judge = LLMJudge(settings, "dummy-key-for-testing")
        logger.info("LLMJudge created successfully")
    except Exception as e:
        logger.error(f"Error creating LLMJudge: {e}")
        return False
    
    logger.info("All basic tests passed! ✅")
    return True

if __name__ == "__main__":
    # Test Google AI Studio integration first
    success, working_format = test_google_ai_studio()
    if success:
        print(f"\n🎉 Found working Google AI Studio format: {working_format}")
    
    # Then test basic functionality
    basic_success = test_basic_functionality()
    
    sys.exit(0 if success and basic_success else 1)
