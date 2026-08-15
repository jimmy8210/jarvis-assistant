"""
Jarvis Assistant - Intelligence & Gemini Dispatch Core
=====================================================

Handles communication with Google's Gemini LLM using the official `google-genai` SDK.
Receives text input, sends it to Gemini with FastMCP tools registered, dispatches any
returned tool calls to `tools.py`, and returns final text responses.

Includes multi-model fallback cascade for 429 rate limits and 404 model errors.
Contains ZERO voice/audio code — pure text in, text out.
"""

import os
import sys
import datetime
import logging
from google import genai
from google.genai import types

import config
from tools import TOOLS, TOOL_MAP

# Force UTF-8 encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Configure file logging for debug/warning logs (keeps console output clean)
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wednesday.log")
logger = logging.getLogger("WednesdayBrain")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


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


def get_system_instruction() -> str:
    """
    Generates system prompt dynamically injected with current local time and date context.
    """
    now = datetime.datetime.now()
    time_str = now.strftime("%I:%M %p")
    date_str = now.strftime("%A, %B %d, %Y")

    return f"""You are the intelligence core of Wednesday AI assistant.
Current System Context:
- Current Local Time: {time_str}
- Current Local Date: {date_str}

Analyze the user's spoken voice command and decide whether to call available tool(s) or respond directly with concise text.

Instructions:
1. Use `find_and_open_app` when the user asks to open or launch an application installed on their computer (e.g., Notion, Calculator, Antigravity, Obsidian, Notepad, Brave, Spotify).
2. Use `close_app` when the user asks to close, quit, terminate, or shut down a running application on their computer (e.g., Notion, Calculator, Notepad, Obsidian).
3. Use `web_search` when the user explicitly asks to perform a web search or navigate to a URL.
4. Use `get_current_time` or `get_current_date` when asked about time or date.
5. Use `stop_assistant` when the user asks to stop, exit, quit, or turn off Wednesday.
6. If the user prompt contains multiple requests, invoke all corresponding tools in order.
7. Keep direct conversational responses clear, friendly, and concise for spoken delivery.

"""


class GeminiBrain:
    """Intelligence Core managing Gemini model selection, tool dispatching, and auto-fallback."""

    def __init__(self, api_key: str = None, primary_model: str = None):
        self.api_key = api_key or config.get_api_key()
        self.primary_model = primary_model or getattr(config, "GEMINI_MODEL_NAME", "gemini-3.6-flash")
        self.client = genai.Client(api_key=self.api_key)
        self.active_model = self.primary_model

    def _get_model_cascade(self) -> list[str]:
        cascade = [self.primary_model]
        for m in FALLBACK_MODELS:
            if m not in cascade:
                cascade.append(m)
        return cascade

    def process_command(self, command_text: str) -> str:
        """
        Sends text prompt + TOOLS to Gemini. Checks if Gemini returned tool call(s) or text,
        dispatches matching tool calls to tools.py, and returns final text response.
        
        :param command_text: Text prompt string.
        :return: Final response string.
        """
        if not command_text or not command_text.strip():
            return "No command provided."

        system_instruction = get_system_instruction()
        config_obj = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=TOOLS,
            temperature=0.1,
        )

        for model in self._get_model_cascade():
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=command_text,
                    config=config_obj,
                )
                self.active_model = model

                # Check if Gemini returned tool function calls
                if hasattr(response, "function_calls") and response.function_calls:
                    results = []
                    for fc in response.function_calls:
                        func_name = fc.name
                        args = fc.args or {}
                        if func_name in TOOL_MAP:
                            tool_res = TOOL_MAP[func_name](**args)
                            results.append(str(tool_res))
                        else:
                            results.append(f"Tool '{func_name}' not recognized.")
                    return "\n".join(results)

                # Return text response if available
                if response.text:
                    return response.text.strip()

                return "Command executed."

            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "404" in err_msg or "NOT_FOUND" in err_msg:
                    logger.warning(f"Model '{model}' unavailable ({e}). Fallback to next model...")
                    continue
                else:
                    logger.warning(f"Error with model '{model}': {e}. Fallback to next model...")
                    continue


        # Heuristic fallback if all API calls fail
        return self._heuristic_fallback(command_text)

    def _heuristic_fallback(self, command_text: str) -> str:
        cmd_lower = command_text.lower().strip()
        sub_commands = [c.strip() for c in cmd_lower.split(" and ") if c.strip()]
        results = []

        for sub in sub_commands:
            if any(w in sub for w in ["stop", "exit", "quit", "goodbye"]):
                results.append(TOOL_MAP["stop_assistant"]())
            elif "time" in sub or "clock" in sub:
                results.append(TOOL_MAP["get_current_time"]())
            elif "date" in sub or "today" in sub:
                results.append(TOOL_MAP["get_current_date"]())
            elif "open" in sub or "launch" in sub or "start" in sub:
                words = sub.replace("open", "").replace("launch", "").replace("start", "").strip()
                results.append(TOOL_MAP["find_and_open_app"](words))
            elif "close" in sub or "kill" in sub or "terminate" in sub:
                words = sub.replace("close", "").replace("kill", "").replace("terminate", "").strip()
                results.append(TOOL_MAP["close_app"](words))
            elif "search" in sub:

                words = sub.replace("search", "").replace("google", "").replace("for", "").strip()
                results.append(TOOL_MAP["web_search"](words))
            else:
                results.append("I couldn't process that command offline.")

        return "\n".join(results)


# Singleton module function instance
_brain_instance = None

def process_command(command_text: str) -> str:
    """Module-level helper to process command text using GeminiBrain."""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = GeminiBrain()
    return _brain_instance.process_command(command_text)


if __name__ == "__main__":
    print("Testing Brain Module with tools dispatch...")
    print("Test 1 (Time):", process_command("what time is it?"))
    print("Test 2 (Open app):", process_command("open notepad"))
