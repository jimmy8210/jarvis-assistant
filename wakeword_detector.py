"""
Jarvis Assistant - Voice Control System
=======================================

Wake word detection via openWakeWord + Vosk speech-to-text.
LLM-driven command interpretation via Gemini API (gemini-3.6-flash).
Dynamic Windows application launcher with zero hardcoded file paths.
"""

import os
import sys
import json
import time
import argparse
import webbrowser
import numpy as np
import pyaudio
from openwakeword.model import Model
from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel

from llm_handler import GeminiLLM
from app_launcher import launch_application

# Force UTF-8 encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Suppress verbose Vosk C++ logs
SetLogLevel(-1)


def process_command(command_text: str, llm: GeminiLLM) -> bool:
    """
    Processes transcribed voice commands using Gemini LLM intent parsing
    and dynamic application launching.
    
    :param command_text: Transcribed user command from Vosk STT.
    :param llm: Initialized GeminiLLM instance.
    :return: True if a stop/exit action was requested, False otherwise.
    """
    print(f"\n[Gemini LLM]: Interpreting command: \"{command_text}\"...")
    
    intent = llm.parse_command_intent(command_text)
    action = intent.get("action", "general_response")
    target = intent.get("target", "")
    explanation = intent.get("explanation", "")

    print(f"[Intent Parsed]: Action = '{action}' | Target = '{target}'")
    if explanation:
        print(f"[Reasoning]: {explanation}")

    # 1. Action: Open / Launch Application
    if action == "open_app":
        success = launch_application(target)
        if not success:
            print(f"[Jarvis]: Could not locate application '{target}' on this device.")
        return False

    # 2. Action: Web Search / Open URL
    elif action == "web_search" or action == "open_url":
        url = target if target.startswith("http") else f"https://www.google.com/search?q={target}"
        print(f"[Jarvis Action]: Opening Web Browser -> {url}")
        webbrowser.open(url)
        return False

    # 3. Action: Stop / Shutdown Jarvis
    elif action == "stop":
        print("[Jarvis Action]: Stop request received! Shutting down Jarvis voice loop...")
        return True

    # 4. Action: General Conversational Response
    elif action == "general_response":
        print(f"[Jarvis AI]: {target}")
        return False

    # Fallback default
    else:
        print(f"[Jarvis AI]: {target if target else 'Command processed.'}")
        return False


def listen_and_transcribe(
    wakeword_models: list[str] = None,
    threshold: float = 0.5,
    listen_duration: float = 4.0,
    vosk_model_path: str = "model",
    sample_rate: int = 16000,
    chunk_size: int = 1280,
):
    """
    Continuous voice loop:
    1. Listens for wake word ('Hey Jarvis') using openWakeWord.
    2. Upon detection, records speech for `listen_duration` seconds.
    3. Transcribes recorded speech using Vosk STT.
    4. Passes transcription to Gemini LLM for intent interpretation.
    5. Dynamically launches requested application or handles request.
    """
    if wakeword_models is None:
        wakeword_models = ["hey_jarvis"]

    if not os.path.exists(vosk_model_path):
        raise FileNotFoundError(
            f"Vosk model path '{vosk_model_path}' not found! Please check the folder."
        )

    print("Initializing Gemini LLM engine...")
    llm = GeminiLLM()

    print(f"Loading openWakeWord model(s): {wakeword_models}...")
    oww_model = Model(wakeword_models=wakeword_models, inference_framework="onnx")

    print(f"Loading Vosk STT model from '{vosk_model_path}'...")
    vosk_model = VoskModel(vosk_model_path)

    # PyAudio setup
    audio = pyaudio.PyAudio()
    mic_stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )

    clean_model_names = [
        os.path.basename(m).replace(".onnx", "").replace(".tflite", "")
        for m in wakeword_models
    ]

    print("\n" + "=" * 60)
    print(f" Jarvis Voice Control System Active")
    print(f" Wake word(s): {', '.join(clean_model_names)}")
    print(f" Detection threshold: {threshold}")
    print(f" Listening window: {listen_duration} seconds after trigger")
    print(f" Intelligence Core: Gemini LLM ({llm.model_name})")
    print(f" Application Launcher: Dynamic Windows Search (Zero Hardcoding)")
    print(" Press Ctrl+C to exit.")
    print("=" * 60 + "\n")

    try:
        while True:
            # Read 16-bit PCM audio chunk from microphone
            raw_data = mic_stream.read(chunk_size, exception_on_overflow=False)
            audio_frame = np.frombuffer(raw_data, dtype=np.int16)

            # Feed audio frame to openWakeWord model
            predictions = oww_model.predict(audio_frame)

            # Check if any wake word score crosses threshold
            triggered_word = None
            for model_name, score in predictions.items():
                if score >= threshold:
                    clean_name = os.path.basename(model_name).replace(".onnx", "").replace(".tflite", "")
                    triggered_word = clean_name
                    print(f"\n[Wake Word]: Detected [{clean_name}] (Score: {score:.3f})")
                    break

            if triggered_word:
                print(f"[Listening]: Listening for your command ({listen_duration}s)... Speak now!")

                # Initialize fresh Vosk recognizer for command transcription
                recognizer = KaldiRecognizer(vosk_model, sample_rate)

                start_time = time.time()
                while time.time() - start_time < listen_duration:
                    cmd_data = mic_stream.read(chunk_size, exception_on_overflow=False)
                    recognizer.AcceptWaveform(cmd_data)

                # Get final transcription result from Vosk
                final_json = json.loads(recognizer.FinalResult())
                command_text = final_json.get("text", "").strip()

                if command_text:
                    print(f"[Vosk STT]: User said: \"{command_text}\"")
                    # Process command using Gemini LLM and dynamic app launcher
                    should_exit = process_command(command_text, llm)
                    if should_exit:
                        break
                else:
                    print("[Vosk STT]: No speech detected or text recognized.")

                print("\nResuming wake word listening...")
                # Reset openWakeWord model predictions buffer after command window
                oww_model.reset()

    except KeyboardInterrupt:
        print("\nStopping Jarvis voice loop...")
    except Exception as err:
        print(f"\n[Voice Loop Notice]: {err}")
    finally:
        try:
            if mic_stream and mic_stream.is_active():
                mic_stream.stop_stream()
        except Exception:
            pass
        try:
            if mic_stream:
                mic_stream.close()
        except Exception:
            pass
        try:
            audio.terminate()
        except Exception:
            pass
        print("Microphone stream closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jarvis LLM-Driven Voice Control System")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["hey_jarvis"],
        help="Wake word models to use (e.g. hey_jarvis alexa)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold score (0.0 to 1.0, default: 0.5)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Command recording duration in seconds after wake word (default: 4.0)",
    )
    parser.add_argument(
        "--vosk-model",
        type=str,
        default="model",
        help="Path to Vosk STT model directory (default: 'model')",
    )

    args = parser.parse_args()
    listen_and_transcribe(
        wakeword_models=args.models,
        threshold=args.threshold,
        listen_duration=args.duration,
        vosk_model_path=args.vosk_model,
    )
