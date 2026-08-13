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
import datetime
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


def get_intent_parser_system_prompt() -> str:
    """
    Generates system prompt dynamically injected with current local time and date context.
    Enforces ordered JSON lists for multi-step / compound voice commands.
    """
    now = datetime.datetime.now()
    time_str = now.strftime("%I:%M %p")
    date_str = now.strftime("%A, %B %d, %Y")

    return f"""You are the intelligence core of Jarvis AI assistant.
Current System Context:
- Current Local Time: {time_str}
- Current Local Date: {date_str}

Your job is to analyze the user's spoken voice command and convert it into structured JSON action(s).

CRITICAL RULE FOR MULTI-STEP / COMPOUND COMMANDS:
If the user's voice command contains MULTIPLE actions or requests (for example: "tell me the time and open notion", "what is today's date and open calculator", "open brave and search for news"), you MUST return a JSON LIST containing EACH action object in the EXACT order requested by the user. Do NOT prioritize one action over another or skip any request!

Available actions:
1. "open_app": The user wants to open or launch an application installed on their computer (e.g., "open Notion", "launch calculator", "start Antigravity", "open Obsidian", "open YouTube app", "start notepad", "open brave browser").
   - Set "target" to the clean name of the application (e.g. "Notion", "Calculator", "Antigravity", "Obsidian", "YouTube", "Notepad", "Brave").
2. "web_search": The user explicitly wants to perform a web search or navigate to a web URL (e.g., "search Google for quantum computing", "open youtube.com").
   - Set "target" to the target URL or web search URL (e.g. "https://www.google.com/search?q=quantum+computing").
3. "stop": The user wants to exit, stop, or turn off Jarvis (e.g., "stop", "exit", "quit", "goodbye", "turn off").
   - Set "target" to "exit".
4. "general_response": The user is asking a general question, asking for current time/date, or chatting with Jarvis (e.g., "what time is it?", "what is today's date?", "who created you?", "how are you doing?").
   - Set "target" to a helpful, concise AI voice assistant response. (Use Current System Context to accurately answer time and date queries).

Response Format:
- Always respond with a valid JSON array/list of action objects:
  [
    {{"action": "general_response", "target": "The current time is {time_str}.", "explanation": "Provide time"}},
    {{"action": "open_app", "target": "Notion", "explanation": "Open Notion app"}}
  ]"""


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
        now = datetime.datetime.now()
        context = f"[Current Date/Time: {now.strftime('%A, %B %d, %Y %I:%M %p')}]\n"
        full_prompt = context + prompt

        for model in self._get_model_cascade():
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=full_prompt,
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

    def parse_command_intent(self, command_text: str) -> dict | list:
        """
        Parses a transcribed user command into a structured intent dictionary or list of intents.
        Automatically cascades to fallback models if free-tier quota limits or model errors are hit.
        
        :param command_text: Transcribed voice command text.
        :return: Dict or List of Dicts containing 'action', 'target', and 'explanation'.
        """
        prompt = f"User voice command: \"{command_text}\""
        system_instruction = get_intent_parser_system_prompt()

        for model in self._get_model_cascade():
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
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
                    continue
                else:
                    continue

        # Heuristic fallback if all model calls fail
        cmd_lower = command_text.lower().strip()
        now = datetime.datetime.now()

        # Split potential compound commands on ' and '
        sub_commands = [c.strip() for c in cmd_lower.split(" and ") if c.strip()]
        intents = []

        for sub_cmd in sub_commands:
            if any(w in sub_cmd for w in ["stop", "exit", "quit", "goodbye"]):
                intents.append({"action": "stop", "target": "exit", "explanation": "Fallback keyword stop"})
            elif "time" in sub_cmd or "clock" in sub_cmd:
                t_str = now.strftime("%I:%M %p")
                intents.append({"action": "general_response", "target": f"The current time is {t_str}.", "explanation": "Fallback time query"})
            elif "date" in sub_cmd or "today" in sub_cmd or "day" in sub_cmd:
                d_str = now.strftime("%A, %B %d, %Y")
                intents.append({"action": "general_response", "target": f"Today is {d_str}.", "explanation": "Fallback date query"})
            elif "open" in sub_cmd or "launch" in sub_cmd or "start" in sub_cmd:
                words = sub_cmd.replace("open", "").replace("launch", "").replace("start", "").strip()
                intents.append({"action": "open_app", "target": words, "explanation": "Fallback keyword open"})
            else:
                intents.append({"action": "general_response", "target": "Command processed.", "explanation": "Fallback"})

        return intents if len(intents) > 1 else (intents[0] if intents else {"action": "general_response", "target": "I couldn't understand that command."})


if __name__ == "__main__":
    print("Testing Gemini Intent Parser with fallback support...")
    bot = GeminiLLM()
    res = bot.parse_command_intent("open notion")
    print("Result:", res)
