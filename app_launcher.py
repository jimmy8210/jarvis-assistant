"""
Dynamic Windows Application Launcher for Jarvis Assistant
==========================================================

Locates and launches applications on Windows dynamically without hardcoded file paths.
Searches System PATH, Start Menu shortcuts, AppData, Program Files, and Windows Protocol URIs.
Spawns processes fully detached from terminal output to prevent Electron/App debug logging.
"""

import os
import sys
import shutil
import subprocess

# Force UTF-8 encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
    
    :param app_name: Name of the application to open (e.g. 'Notion', 'Calculator', 'Obsidian', 'Antigravity').
    :return: True if an application launch was initiated, False if not found.
    """
    if not app_name or not app_name.strip():
        print("[App Launcher]: No application name specified.")
        return False

    app_clean = app_name.lower().strip()
    target_name = COMMON_ALIASES.get(app_clean, app_clean)
    print(f"[App Launcher]: Searching for application '{app_name}'...")

    # 1. Check System PATH (for standard executables like calc, notepad, cmd)
    which_path = shutil.which(target_name) or shutil.which(f"{target_name}.exe")
    if which_path:
        print(f"[App Launcher]: Found in System PATH -> {which_path}")
        return _execute_detached(which_path)

    # 2. Check Windows Protocol URIs (e.g. whatsapp:, spotify:)
    if target_name in PROTOCOL_APPS:
        proto = PROTOCOL_APPS[target_name]
        print(f"[App Launcher]: Launching via Windows Protocol -> {proto}")
        return _execute_detached(proto)

    # 3. Recursive Search in Start Menu and Program Directories
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
                    # Check substring or exact match
                    if target_name == basename or target_name in basename or basename in target_name:
                        full_path = os.path.join(root, file_name)
                        score = 100 if f_lower.endswith('.lnk') else 50
                        if 'start menu' in full_path.lower():
                            score += 50
                        if target_name == basename:
                            score += 30
                        candidates.append((score, full_path))

    if candidates:
        # Sort candidates by highest score
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_match = candidates[0][1]
        print(f"[App Launcher]: Found application shortcut -> {best_match}")
        return _execute_detached(best_match)

    # 4. Fallback to Windows Shell 'start' command
    print(f"[App Launcher]: Attempting Windows shell start fallback for '{target_name}'...")
    return _execute_detached(target_name)


if __name__ == "__main__":
    # Self-test launcher
    print("Testing dynamic app launcher...")
    test_apps = ["Notion", "Calculator", "Antigravity", "Notepad"]
    for test_app in test_apps:
        launch_application(test_app)
