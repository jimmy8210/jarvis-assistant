"""
Jarvis Assistant - Configuration Template
=========================================

INSTRUCTIONS FOR USER:
----------------------
1. Get a Gemini API Key:
   - Go to Google AI Studio: https://aistudio.google.com/
   - Sign in with your Google Account.
   - Click on "Create API Key" (or "Get API Key").
   - Copy your generated API key.

2. Create your local config file:
   - Copy this file (`config_example.py`) and rename the copy to `config.py` in the project root directory.
   - Command line example:
       copy config_example.py config.py    (Windows)
       cp config_example.py config.py      (Linux/macOS)

3. Add your API Key:
   - Open `config.py` in your code editor.
   - Replace "YOUR_GEMINI_API_KEY_HERE" with your actual Gemini API key string:
       GEMINI_API_KEY = "AIzaSy..."

SECURITY NOTICE:
----------------
`config.py` is listed in `.gitignore` so your actual API key will NEVER be pushed to GitHub.
Do NOT commit `config.py` to Git repository.
"""

import os

# ==============================================================================
# GEMINI API CONFIGURATION
# ==============================================================================
# Paste your Google Gemini API key below:
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

# Default Gemini model to use (e.g. gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash)
GEMINI_MODEL_NAME = "gemini-3.6-flash"


def get_api_key() -> str:
    """
    Retrieves the Gemini API key from `GEMINI_API_KEY` variable or environment.
    
    :return: Gemini API key string.
    :raises ValueError: If the API key is not configured or left as placeholder.
    """
    api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        raise ValueError(
            "Gemini API key is missing or invalid!\n"
            "Please open `config.py` and set `GEMINI_API_KEY = 'your_api_key_here'`.\n"
            "You can get a free API key at https://aistudio.google.com/"
        )
    return api_key
