"""
Gemini LLM Handler for Jarvis Assistant
======================================

Handles communication with Google's Gemini LLM using the official `google-genai` SDK.
Provides intent parsing for voice commands, structured response generation,
and automatic fallback models when free-tier rate limits (429) or invalid model names (404) are encountered.
"""

import os
import sys
import json
from google import genai
from google.genai import types
import config

# Force UTF-8 encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Recommended model cascade (fallback order if a model hits rate limits or is invalid)
FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.1-pro-preview",
    "gemini-pro-latest",
    "gemma-4-31b-it",
]

INTENT_PARSER_SYSTEM_PROMPT = """You are the intelligence core of Jarvis AI assistant.
Your job is to analyze the user's spoken voice command and convert it into a structured JSON action.

Available actions:
1. "open_app": The user wants to open or launch an application installed on their computer (e.g., "open Notion", "launch calculator", "start Antigravity", "open Obsidian", "open YouTube app", "start notepad", "open brave browser").
   - Set "target" to the clean name of the application (e.g. "Notion", "Calculator", "Antigravity", "Obsidian", "YouTube", "Notepad", "Brave").
2. "web_search": The user explicitly wants to perform a web search or navigate to a web URL (e.g., "search Google for quantum computing", "open youtube.com").
   - Set "target" to the target URL or web search URL (e.g. "https://www.google.com/search?q=quantum+computing").
3. "stop": The user wants to exit, stop, or turn off Jarvis (e.g., "stop", "exit", "quit", "goodbye", "turn off").
   - Set "target" to "exit".
4. "general_response": The user is asking a general question, seeking information, or chatting with Jarvis (e.g., "what is the date?", "who created you?", "how are you doing?").
   - Set "target" to a helpful, concise AI voice assistant response.

You MUST respond strictly with valid JSON conforming to this schema:
{
  "action": "open_app" | "web_search" | "stop" | "general_response",
  "target": "target app name OR search URL OR exit OR conversational response",
  "explanation": "brief reasoning"
}"""


class GeminiLLM:
    """Wrapper class for interacting with Google Gemini models."""

    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initializes the Gemini client.
        
        :param api_key: Optional API key. If not provided, loaded from config.py
        :param model_name: Optional model name. Default: config.GEMINI_MODEL_NAME
        """
        self.api_key = api_key or config.get_api_key()
        self.primary_model = model_name or getattr(config, "GEMINI_MODEL_NAME", "gemini-3.6-flash")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = self.primary_model

    def _get_model_cascade(self) -> list[str]:
        cascade = [self.primary_model]
        for m in FALLBACK_MODELS:
            if m not in cascade:
                cascade.append(m)
        return cascade

    def generate_response(self, prompt: str) -> str:
        """
        Sends a text prompt to Gemini and returns the response with auto-fallback.
        """
        for model in self._get_model_cascade():
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                self.model_name = model
                return response.text
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "404" in err_msg:
                    print(f"[Gemini LLM]: Model '{model}' unavailable. Trying fallback model...")
                    continue
                return f"Error communicating with Gemini API ({model}): {e}"
        
        return "Error: All Gemini models rate-limited or unavailable. Please try again shortly."

    def parse_command_intent(self, command_text: str) -> dict:
        """
        Parses a transcribed user command into a structured intent dictionary.
        Automatically cascades to fallback models if free-tier quota limits or model errors are hit.
        
        :param command_text: Transcribed voice command text.
        :return: Dict containing 'action', 'target', and 'explanation'.
        """
        prompt = f"User voice command: \"{command_text}\""

        for model in self._get_model_cascade():
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=INTENT_PARSER_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        temperature=0.1,
                    )
                )
                self.model_name = model
                parsed_json = json.loads(response.text)
                return parsed_json
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "404" in err_msg or "NOT_FOUND" in err_msg:
                    print(f"[Gemini LLM]: Model '{model}' unavailable/quota limit hit. Cascading to fallback model...")
                    continue
                else:
                    print(f"[Gemini LLM]: Error with model '{model}': {e}")
                    continue

        # Heuristic fallback if all model calls fail
        print("[Gemini LLM]: Falling back to local heuristic intent parser.")
        cmd_lower = command_text.lower().strip()
        if any(w in cmd_lower for w in ["stop", "exit", "quit", "goodbye"]):
            return {"action": "stop", "target": "exit", "explanation": "Fallback keyword stop"}
        elif "open" in cmd_lower or "launch" in cmd_lower or "start" in cmd_lower:
            words = cmd_lower.replace("open", "").replace("launch", "").replace("start", "").strip()
            return {"action": "open_app", "target": words, "explanation": "Fallback keyword open"}
        else:
            return {"action": "general_response", "target": "I'm having trouble connecting right now.", "explanation": "Fallback"}


if __name__ == "__main__":
    print("Testing Gemini Intent Parser with fallback support...")
    bot = GeminiLLM()
    res = bot.parse_command_intent("open notion")
    print("Result:", res)
