import os
import sys
import time
import datetime
import logging
from google import genai
from google.genai import types

import config
from tools import TOOLS, TOOL_MAP

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wednesday.log")
logger = logging.getLogger("WednesdayBrain")
logger.setLevel(logging.ERROR)
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Optimized model cascade starting with ultra-fast gemini-flash-lite-latest
FALLBACK_MODELS = [
    "models/gemini-flash-lite-latest",
    "models/gemini-flash-latest",
    "models/gemini-pro-latest",
    "models/gemini-3-flash-preview",
    "models/gemini-3.1-pro-preview",
]

def get_system_instruction() -> str:
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
    def __init__(self, api_key: str = None, primary_model: str = None):
        self.api_key = api_key or config.get_api_key()
        self.primary_model = primary_model or getattr(config, "GEMINI_MODEL_NAME", "models/gemini-flash-lite-latest")
        self.client = genai.Client(api_key=self.api_key)
        self.active_model = self.primary_model

    def _get_model_cascade(self) -> list[str]:
        cascade = [self.primary_model]
        for m in FALLBACK_MODELS:
            if m not in cascade:
                cascade.append(m)
        return cascade

    def process_command_with_timing(self, command_text: str) -> tuple[str, float, float, list[dict]]:
        """
        Sends text prompt to Gemini with isolated API call timing and fallback tracking.
        
        :return: Tuple of (response_str, total_api_time, tool_exec_time, attempt_logs)
        """
        if not command_text or not command_text.strip():
            return "No command provided.", 0.0, 0.0, []

        system_instruction = get_system_instruction()
        config_obj = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=TOOLS,
            temperature=0.1,
        )

        cascade = self._get_model_cascade()
        attempt_logs = []
        total_api_start = time.perf_counter()

        for attempt_idx, model_name in enumerate(cascade, 1):
            t0_single = time.perf_counter()
            print(f" [Gemini API Attempt {attempt_idx}/{len(cascade)}]: Target Model = '{model_name}'...")
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=command_text,
                    config=config_obj,
                )
                single_api_time = time.perf_counter() - t0_single
                self.active_model = model_name

                print(f"  └─ SUCCESS in {single_api_time:.3f}s (Model: '{model_name}')")
                attempt_logs.append({
                    "attempt": attempt_idx,
                    "model": model_name,
                    "status": "SUCCESS",
                    "isolated_time": single_api_time,
                })

                total_api_time = time.perf_counter() - total_api_start

                # Check tool calls
                if hasattr(response, "function_calls") and response.function_calls:
                    results = []
                    t0_tool = time.perf_counter()
                    for fc in response.function_calls:
                        func_name = fc.name
                        args = fc.args or {}
                        if func_name in TOOL_MAP:
                            tool_res = TOOL_MAP[func_name](**args)
                            results.append(str(tool_res))
                        else:
                            results.append(f"Tool '{func_name}' not recognized.")
                    tool_time = time.perf_counter() - t0_tool
                    return "\n".join(results), total_api_time, tool_time, attempt_logs

                if response.text:
                    return response.text.strip(), total_api_time, 0.0, attempt_logs

                return "Command executed.", total_api_time, 0.0, attempt_logs

            except Exception as e:
                single_api_time = time.perf_counter() - t0_single
                err_first_line = str(e).split("\n")[0][:80]
                print(f"  └─ FALLBACK TRIGGERED! Failed in {single_api_time:.3f}s: {err_first_line}")
                
                attempt_logs.append({
                    "attempt": attempt_idx,
                    "model": model_name,
                    "status": "FAILED",
                    "isolated_time": single_api_time,
                    "error": err_first_line
                })
                logger.warning(f"Attempt {attempt_idx} ({model_name}) failed in {single_api_time:.3f}s: {e}")
                continue

        total_api_time = time.perf_counter() - total_api_start
        t0_fb = time.perf_counter()
        res = self._heuristic_fallback(command_text)
        tool_time = time.perf_counter() - t0_fb
        return res, total_api_time, tool_time, attempt_logs

    def process_command(self, command_text: str) -> str:
        res, _, _, _ = self.process_command_with_timing(command_text)
        return res

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

_brain_instance = None

def get_brain() -> GeminiBrain:
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = GeminiBrain()
    return _brain_instance

def process_command(command_text: str) -> str:
    return get_brain().process_command(command_text)

def process_command_with_timing(command_text: str) -> tuple[str, float, float, list[dict]]:
    return get_brain().process_command_with_timing(command_text)
