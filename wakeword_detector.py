import os
import json
import time
import argparse
import webbrowser
import numpy as np
import pyaudio
from openwakeword.model import Model
from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel

# Suppress verbose Vosk C++ logs
SetLogLevel(-1)


def process_command(command_text: str) -> bool:
    """
    Processes recognized text using 'contains keyword' matching logic.
    
    :param command_text: The transcribed string from Vosk STT.
    :return: True if a stop/exit keyword was detected, False otherwise.
    """
    cmd_lower = command_text.lower().strip()

    # Contains-keyword matching for stop/exit commands
    stop_keywords = ["stop", "exit", "quit", "goodbye", "bye"]
    if any(keyword in cmd_lower for keyword in stop_keywords):
        print("🛑 Stop keyword detected! Shutting down Jarvis voice loop...")
        return True

    # Contains-keyword matching for actions
    if "time" in cmd_lower:
        current_time_str = time.strftime("%I:%M %p")
        print(f"⏰ [Jarvis Action]: The current time is {current_time_str}")
    elif "date" in cmd_lower or "today" in cmd_lower:
        current_date_str = time.strftime("%A, %B %d, %Y")
        print(f"📅 [Jarvis Action]: Today's date is {current_date_str}")
    elif "name" in cmd_lower:
        print("🤖 [Jarvis Action]: My name is Jarvis, your AI assistant.")
    elif "editor" in cmd_lower:
        print("📝 [Jarvis Action]: Opening Notepad / Editor...")
        os.system("start notepad.exe")
    elif any(kw in cmd_lower for kw in ["browser", "chrome", "edge", "web", "google", "internet"]):
        print("🌐 [Jarvis Action]: Opening default web browser...")
        webbrowser.open("https://www.google.com")
    elif "open" in cmd_lower:
        # Extract target app or word after 'open' using contains logic
        target = cmd_lower.split("open", 1)[1].strip()
        print(f"🚀 [Jarvis Action]: Opening '{target if target else 'requested item'}'...")
    elif "how are you" in cmd_lower:
        print("😊 [Jarvis Action]: I am running smoothly and ready for your command!")
    else:
        print(f"💡 [Jarvis Action]: Command received containing: \"{command_text}\"")

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
    1. Listens for wake word using openWakeWord.
    2. Upon detection, records user speech for `listen_duration` seconds.
    3. Transcribes the recorded speech using Vosk STT.
    4. Evaluates commands using 'contains keyword' matching.
    """
    if wakeword_models is None:
        wakeword_models = ["hey_jarvis"]

    if not os.path.exists(vosk_model_path):
        raise FileNotFoundError(
            f"Vosk model path '{vosk_model_path}' not found! Please check the folder."
        )

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

    print("\n" + "=" * 55)
    print(f" 🤖 Jarvis Voice System Active")
    print(f" Wake word(s): {', '.join(clean_model_names)}")
    print(f" Detection threshold: {threshold}")
    print(f" Listening window: {listen_duration} seconds after trigger")
    print(f" Command Matching: 'contains keyword' logic")
    print(" Press Ctrl+C to exit.")
    print("=" * 55 + "\n")

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
                    print(f"\n✨ Wake word detected! [{clean_name}] (Score: {score:.3f})")
                    break

            if triggered_word:
                print(f"🎙️ Listening for your command ({listen_duration}s)... Speak now!")

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
                    print(f"🗣️ Jarvis heard: \"{command_text}\"")
                    # Process command using 'contains keyword' matching logic
                    should_exit = process_command(command_text)
                    if should_exit:
                        break
                else:
                    print("⚠️ No speech detected or text recognized.")

                print("\nResuming wake word listening...")
                # Reset openWakeWord model predictions buffer after command window
                oww_model.reset()

    except KeyboardInterrupt:
        print("\nStopping Jarvis voice loop...")
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        audio.terminate()
        print("Microphone stream closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jarvis Wake Word & Vosk STT Voice Loop")
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
