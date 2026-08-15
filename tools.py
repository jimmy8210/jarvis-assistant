"""
Jarvis Assistant - Tools & MCP Server Module
============================================

Contains all executable actions (opening apps, time/date queries, web search, stop)
registered as MCP tools using FastMCP, as well as the application cache system.
Also exports a TOOLS registry compatible with Gemini LLM function calling.
"""

import os
import sys
import json
import shutil
import difflib
import datetime
import subprocess
import webbrowser
import logging
import psutil
from fastmcp import FastMCP


# Force UTF-8 encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Configure file logging for debug/warning logs (keeps console output clean)
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wednesday.log")
logger = logging.getLogger("WednesdayTools")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Initialize FastMCP Server
mcp = FastMCP("WednesdayTools")


CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_cache.json")
SIMILARITY_THRESHOLD = 0.75

# Common spoken names mapped to executable names or system aliases
COMMON_ALIASES = {
    "calculator": "calc",
    "calc": "calc",
    "notepad": "notepad",
    "editor": "notepad",
    "text editor": "notepad",
    "cmd": "cmd",
    "terminal": "wt",
    "command prompt": "cmd",
    "paint": "mspaint",
    "explorer": "explorer",
    "file manager": "explorer",
    "files": "explorer",
    "my computer": "explorer",
}

# Windows UWP app protocol URIs
PROTOCOL_APPS = {
    "whatsapp": "whatsapp:",
    "spotify": "spotify:",
}


def calculate_similarity(name1: str, name2: str) -> float:
    """
    Computes fuzzy matching similarity ratio using difflib.SequenceMatcher.
    Returns a score between 0.0 and 1.0.
    """
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    if n1 == n2:
        return 1.0
    return difflib.SequenceMatcher(None, n1, n2).ratio()


def load_cache() -> dict:
    """
    Reads app_cache.json and returns its contents as a Python dictionary.
    Self-heals if missing or corrupt.
    """
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except Exception as e:
        logger.warning(f"[App Cache]: Warning loading cache: {e}")
        return {}


def save_cache(cache_dict: dict) -> None:
    """
    Writes the given dictionary back to app_cache.json.
    """
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_dict, f, indent=4)
    except Exception as e:
        logger.error(f"[App Cache]: Error saving cache: {e}")


def _execute_detached(target_path_or_uri: str) -> bool:
    """
    Executes a shortcut, binary, or URI fully detached from the Python console.
    Redirects stdout and stderr to DEVNULL to prevent launched apps from dumping debug logs.
    """
    try:
        cmd = f'start "" "{target_path_or_uri}"'
        subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
        return True
    except Exception as e:
        logger.error(f"[App Launcher]: Error executing '{target_path_or_uri}': {e}")
        return False



@mcp.tool()
def find_and_open_app(app_name: str) -> str:
    """
    Dynamically searches for and launches a Windows application by name.
    Uses app_cache.json for fast opening, fuzzy searching for shortcuts/executables,
    and self-heals invalid cache paths.

    :param app_name: Name of the application to open (e.g. 'Notion', 'Calculator', 'Notepad', 'Obsidian').
    :return: A status string indicating if the app was launched or not found.
    """
    if not app_name or not app_name.strip():
        return "No application name specified."

    app_clean = app_name.lower().strip()
    cache = load_cache()

    # 1. Direct or fuzzy match in app_cache.json
    matched_cache_key = None
    if app_clean in cache:
        matched_cache_key = app_clean
    else:
        # Fuzzy match against existing cached application names
        best_cache_sim = 0.0
        for cached_key in cache.keys():
            sim = calculate_similarity(app_clean, cached_key)
            if sim >= SIMILARITY_THRESHOLD and sim > best_cache_sim:
                best_cache_sim = sim
                matched_cache_key = cached_key

    if matched_cache_key:
        cached_path = cache[matched_cache_key]
        is_protocol = cached_path.endswith(":") or cached_path in PROTOCOL_APPS.values()
        if not is_protocol and not os.path.exists(cached_path):
            del cache[matched_cache_key]
            save_cache(cache)
        else:
            success = _execute_detached(cached_path)
            if success:
                return f"Successfully opened {matched_cache_key}."
            else:
                del cache[matched_cache_key]
                save_cache(cache)

    # 2. Check aliases and fuzzy match against alias keys
    target_name = COMMON_ALIASES.get(app_clean, app_clean)
    if target_name == app_clean:
        best_alias_sim = 0.0
        for alias_key, alias_val in COMMON_ALIASES.items():
            sim = calculate_similarity(app_clean, alias_key)
            if sim >= SIMILARITY_THRESHOLD and sim > best_alias_sim:
                best_alias_sim = sim
                target_name = alias_val

    found_path = None

    which_path = shutil.which(target_name) or shutil.which(f"{target_name}.exe")
    if which_path:
        found_path = which_path
    elif target_name in PROTOCOL_APPS:
        found_path = PROTOCOL_APPS[target_name]
    else:
        search_directories = [
            os.path.expandvars(r'%APPDATA%\Microsoft\Windows\Start Menu\Programs'),
            os.path.expandvars(r'%PROGRAMDATA%\Microsoft\Windows\Start Menu\Programs'),
            os.path.expandvars(r'%LOCALAPPDATA%\Programs'),
            r'C:\Program Files',
            r'C:\Program Files (x86)',
            r'C:\Windows\System32',
        ]

        candidates = []
        for sdir in search_directories:
            if not os.path.exists(sdir):
                continue
            for root, _, files in os.walk(sdir):
                for file_name in files:
                    f_lower = file_name.lower()
                    if any(kw in f_lower for kw in ['uninstall', 'unins', 'update', 'setup', 'helper']):
                        continue
                    if f_lower.endswith(('.lnk', '.exe')):
                        basename = os.path.splitext(f_lower)[0]
                        sim_target = calculate_similarity(target_name, basename)
                        sim_clean = calculate_similarity(app_clean, basename)
                        similarity = max(sim_target, sim_clean)

                        if similarity >= SIMILARITY_THRESHOLD:
                            full_path = os.path.join(root, file_name)
                            score = similarity * 100
                            if f_lower.endswith('.lnk'):
                                score += 20
                            if 'start menu' in full_path.lower():
                                score += 10
                            candidates.append((score, similarity, full_path))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            found_path = candidates[0][2]


    if not found_path:
        return f"Could not find an application matching '{app_name}' on this device."

    # Cache and launch
    cache[app_clean] = found_path
    save_cache(cache)
    success = _execute_detached(found_path)
    if success:
        return f"Successfully opened {app_name}."
    return f"Failed to launch application at path: {found_path}"


@mcp.tool()
def get_current_time() -> str:
    """
    Returns the current local system time.

    :return: Formatted current time string (e.g. '11:37 AM').
    """
    now = datetime.datetime.now()
    return f"The current time is {now.strftime('%I:%M %p')}."


@mcp.tool()
def get_current_date() -> str:
    """
    Returns the current local system date.

    :return: Formatted current date string (e.g. 'Thursday, August 13, 2026').
    """
    now = datetime.datetime.now()
    return f"Today is {now.strftime('%A, %B %d, %Y')}."


@mcp.tool()
def web_search(query_or_url: str) -> str:
    """
    Performs a web search using Google or opens a specific web URL in the default browser.

    :param query_or_url: Search query or web URL to open (e.g. 'quantum computing' or 'https://github.com').
    :return: Status string confirming web search launch.
    """
    if not query_or_url or not query_or_url.strip():
        return "No search query or URL specified."

    target = query_or_url.strip()
    if target.startswith("http://") or target.startswith("https://"):
        url = target
    else:
        url = f"https://www.google.com/search?q={target}"

    webbrowser.open(url)
    return f"Opened web search for '{target}' in default browser."


@mcp.tool()
def stop_assistant() -> str:
    """
    Stops and shuts down the Jarvis voice assistant loop.

    :return: Exit message string.
    """
    return "Stopping Wednesday voice loop. Goodbye!"


@mcp.tool()
def close_app(app_name: str) -> str:
    """
    Closes a running application by name. Searches active system processes using fuzzy matching.
    Attempts a clean close (terminate), and if it does not exit within 2 seconds, force closes (kill).

    :param app_name: Spoken or written name of the application to close (e.g. 'Notepad', 'Calculator', 'Notion', 'Obsidian', 'Chrome').
    :return: A status message indicating success or if the application was not found/running.
    """
    if not app_name or not app_name.strip():
        return "No application name specified to close."

    app_clean = app_name.lower().strip()

    # Determine potential target executable names (aliases, cache paths, or original name)
    target_names = {app_clean}
    if app_clean in COMMON_ALIASES:
        target_names.add(COMMON_ALIASES[app_clean].lower())
    for alias_k, alias_v in COMMON_ALIASES.items():
        if calculate_similarity(app_clean, alias_k) >= SIMILARITY_THRESHOLD:
            target_names.add(alias_v.lower())

    cache = load_cache()
    if app_clean in cache:
        cached_path = cache[app_clean]
        exec_name = os.path.basename(cached_path).lower()
        base_name = os.path.splitext(exec_name)[0]
        target_names.add(exec_name)
        target_names.add(base_name)

    # Find matching running processes
    matching_procs = []
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            pname = proc.info['name']
            if not pname:
                continue
            pname_clean = pname.lower()
            pname_base = os.path.splitext(pname_clean)[0]

            # Direct match against target names or fuzzy similarity match
            is_match = False
            for t in target_names:
                if (
                    pname_clean == t
                    or pname_base == t
                    or calculate_similarity(t, pname_clean) >= SIMILARITY_THRESHOLD
                    or calculate_similarity(t, pname_base) >= SIMILARITY_THRESHOLD
                ):
                    is_match = True
                    break

            if not is_match and calculate_similarity(app_clean, pname_base) >= SIMILARITY_THRESHOLD:
                is_match = True

            if is_match:
                matching_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    if not matching_procs:
        return f"Application '{app_name}' is not currently running."

    # Attempt clean termination first
    for proc in matching_procs:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Wait for processes to exit (timeout=2 seconds)
    gone, alive = psutil.wait_procs(matching_procs, timeout=2.0)

    # Force kill any stubborn processes remaining
    for proc in alive:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return f"Successfully closed {app_name}."


# Export registry of tool functions for Gemini function calling
TOOLS = [
    find_and_open_app,
    close_app,
    get_current_time,
    get_current_date,
    web_search,
    stop_assistant,
]

# Map of tool function names to callables for easy dispatching
TOOL_MAP = {func.__name__: func for func in TOOLS}



if __name__ == "__main__":
    print("Starting Wednesday FastMCP Tools Server...")
    mcp.run()
