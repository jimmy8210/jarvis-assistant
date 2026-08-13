"""
Dynamic Windows Application Launcher for Jarvis Assistant
==========================================================

Locates and launches applications on Windows dynamically without hardcoded file paths.
Searches System PATH, Start Menu shortcuts, AppData, Program Files, and Windows Protocol URIs.
Spawns processes fully detached from terminal output to prevent Electron/App debug logging.
"""

import os
import sys
import json
import shutil
import difflib
import subprocess

# Force UTF-8 encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
    Computes fuzzy matching similarity ratio between two strings using difflib.SequenceMatcher.
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
    If the file doesn't exist yet or is empty, returns an empty dictionary instead of erroring.
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
        print(f"[App Launcher]: Warning loading cache: {e}")
        return {}


def save_cache(cache_dict: dict) -> None:
    """
    Writes the given dictionary back to app_cache.json, overwriting the file.
    """
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_dict, f, indent=4)
    except Exception as e:
        print(f"[App Launcher]: Error saving cache: {e}")


def _execute_detached(target_path_or_uri: str) -> bool:
    """
    Executes a shortcut, binary, or URI fully detached from the Python console.
    Redirects stdout and stderr to DEVNULL to prevent launched apps (like Notion,
    Discord, Chrome) from dumping debug logs into the terminal window.
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
        print(f"[App Launcher]: Error executing '{target_path_or_uri}': {e}")
        return False


def launch_application(app_name: str) -> bool:
    """
    Dynamically searches for and launches a Windows application by name.
    First checks app_cache.json; if cached path exists, opens with _execute_detached().
    Otherwise performs fuzzy search (threshold >= 0.75), caches the path, and launches the app.
    
    :param app_name: Name of the application to open (e.g. 'Notion', 'Calculator', 'Hermes').
    :return: True if an application launch was initiated, False if not found.
    """
    if not app_name or not app_name.strip():
        print("[App Launcher]: No application name specified.")
        return False

    app_clean = app_name.lower().strip()
    cache = load_cache()

    # 1. Check if requested app name exists in cache
    if app_clean in cache:
        cached_path = cache[app_clean]
        # Verify path existence if it's a file path
        is_protocol = cached_path.endswith(":") or cached_path in PROTOCOL_APPS.values()
        if not is_protocol and not os.path.exists(cached_path):
            del cache[app_clean]
            save_cache(cache)
        else:
            success = _execute_detached(cached_path)
            if success:
                return True
            else:
                del cache[app_clean]
                save_cache(cache)

    # 2. If app isn't in cache (or cached path failed), run search as before
    target_name = COMMON_ALIASES.get(app_clean, app_clean)
    found_path = None

    # Check System PATH (for standard executables like calc, notepad, cmd)
    which_path = shutil.which(target_name) or shutil.which(f"{target_name}.exe")
    if which_path:
        found_path = which_path

    # Check Windows Protocol URIs (e.g. whatsapp:, spotify:)
    elif target_name in PROTOCOL_APPS:
        found_path = PROTOCOL_APPS[target_name]

    # Recursive Search in Start Menu and Program Directories
    else:
        search_directories = [
            os.path.expandvars(r'%APPDATA%\Microsoft\Windows\Start Menu\Programs'),
            os.path.expandvars(r'%PROGRAMDATA%\Microsoft\Windows\Start Menu\Programs'),
            os.path.expandvars(r'%LOCALAPPDATA%\Programs'),
            r'C:\Program Files',
            r'C:\Program Files (x86)',
        ]

        candidates = []
        for sdir in search_directories:
            if not os.path.exists(sdir):
                continue
            for root, _, files in os.walk(sdir):
                for file_name in files:
                    f_lower = file_name.lower()
                    # Skip installer / uninstaller files
                    if any(kw in f_lower for kw in ['uninstall', 'unins', 'update', 'setup', 'helper']):
                        continue

                    if f_lower.endswith(('.lnk', '.exe')):
                        basename = os.path.splitext(f_lower)[0]
                        
                        # Fuzzy match calculation using difflib
                        sim_target = calculate_similarity(target_name, basename)
                        sim_clean = calculate_similarity(app_clean, basename)
                        similarity = max(sim_target, sim_clean)

                        # Only accept candidates meeting the similarity threshold
                        if similarity >= SIMILARITY_THRESHOLD:
                            full_path = os.path.join(root, file_name)
                            score = similarity * 100
                            if f_lower.endswith('.lnk'):
                                score += 20
                            if 'start menu' in full_path.lower():
                                score += 10
                            candidates.append((score, similarity, full_path))

        if candidates:
            # Sort candidates by highest score
            candidates.sort(key=lambda x: x[0], reverse=True)
            found_path = candidates[0][2]

    # If no candidate scored above the similarity threshold
    if not found_path:
        return False

    # 3. Add to cache dictionary and call save_cache() BEFORE opening
    cache[app_clean] = found_path
    save_cache(cache)

    # 4. Open application fully detached to prevent Electron/app logs from polluting standard output
    return _execute_detached(found_path)


if __name__ == "__main__":
    # Self-test launcher
    print("Testing dynamic app launcher with caching...")
    # Test loading cache
    c = load_cache()
    print(f"Current Cache: {c}")

