"""
Jarvis Assistant - Main Voice Loop Module
=========================================

Handles the voice pipeline exclusively: wake word detection (openWakeWord),
dynamic voice activity detection (webrtcvad), and Vosk speech-to-text transcription.
Delegates command interpretation and tool execution entirely to brain.py.
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import pyaudio
import webrtcvad
from openwakeword.model import Model
from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel

import brain

# Force UTF-8 encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Configure file logging for debug/warning logs (keeps console output clean)
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis.log")
logger = logging.getLogger("JarvisMain")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Suppress verbose Vosk C++ logs
SetLogLevel(-1)



def listen_and_transcribe(
    wakeword_models: list[str] = None,
    threshold: float = 0.5,
    vosk_model_path: str = "model",
    sample_rate: int = 16000,
    chunk_size: int = 1280,
    silence_duration: float = 1.2,
    max_recording_time: float = 15.0,
    initial_speech_timeout: float = 4.0,
    vad_mode: int = 2,
):
    """
    Continuous voice loop with Dynamic VAD Listening:
    1. Listens for wake word ('Hey Jarvis') using openWakeWord.
    2. Upon detection, records 30ms audio chunks dynamically using webrtcvad.
    3. Continues recording while speech is active; stops upon continuous silence.
    4. Transcribes recorded speech using Vosk STT.
    5. Hands transcribed text to brain.process_command() and outputs response.
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

    # Initialize WebRTC VAD (mode 2: balanced aggressiveness)
    vad = webrtcvad.Vad(vad_mode)
    vad_chunk_samples = 480  # 30ms at 16000Hz = 480 samples
    vad_chunk_bytes = vad_chunk_samples * 2  # 16-bit mono = 960 bytes

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
    print(f" Dynamic Listening: Active ({silence_duration}s silence threshold)")
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
                    print(f"\n[Wake Word]: Detected [{clean_name}]")
                    break

            if triggered_word:
                print(f"[Listening]: Speak your command now...")

                # Initialize fresh Vosk recognizer for command transcription
                recognizer = KaldiRecognizer(vosk_model, sample_rate)

                speech_started = False
                silent_chunks_count = 0
                max_silent_chunks = int(silence_duration / 0.03)  # 30ms chunks count

                start_time = time.time()
                while True:
                    now = time.time()
                    elapsed = now - start_time

                    # Safety net 1: Max recording duration reached
                    if elapsed >= max_recording_time:
                        break

                    # Safety net 2: Initial speech timeout if user hasn't spoken
                    if not speech_started and elapsed >= initial_speech_timeout:
                        print(f"[Listening]: No speech detected.")
                        break

                    # Read 30ms audio chunk for VAD and Vosk
                    cmd_data = mic_stream.read(vad_chunk_samples, exception_on_overflow=False)
                    if len(cmd_data) != vad_chunk_bytes:
                        continue

                    # Feed 30ms audio chunk into Vosk STT
                    recognizer.AcceptWaveform(cmd_data)

                    # Check voice activity using webrtcvad
                    try:
                        is_speech = vad.is_speech(cmd_data, sample_rate)
                    except Exception:
                        is_speech = False

                    if is_speech:
                        if not speech_started:
                            speech_started = True
                        silent_chunks_count = 0
                    else:
                        if speech_started:
                            silent_chunks_count += 1
                            if silent_chunks_count >= max_silent_chunks:
                                break

                # Get final transcription result from Vosk
                final_json = json.loads(recognizer.FinalResult())
                transcribed_text = final_json.get("text", "").strip()

                if transcribed_text:
                    print(f"[User]: \"{transcribed_text}\"")
                    # Single call to brain.process_command
                    response = brain.process_command(transcribed_text)
                    print(f"[Jarvis]: {response}\n")

                    if "Stopping Jarvis voice loop" in response or "Goodbye!" in response:
                        break
                else:
                    print("[Jarvis]: No speech recognized.\n")

                # Reset openWakeWord model predictions buffer after command window
                oww_model.reset()

    except KeyboardInterrupt:
        print("\nStopping Jarvis voice loop...")
    except Exception as err:
        logger.error(f"[Voice Loop Notice]: {err}")

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
        "--silence-duration",
        type=float,
        default=1.2,
        help="Silence duration in seconds to stop recording (default: 1.2)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=15.0,
        help="Maximum safety net recording duration in seconds (default: 15.0)",
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
        silence_duration=args.silence_duration,
        max_recording_time=args.max_duration,
        vosk_model_path=args.vosk_model,
    )
