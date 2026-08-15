import argparse
import json
import logging
import os
import time
import numpy as np
import pyaudio
import webrtcvad

import brain
import tts_handler
from openwakeword.model import Model
from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel

LOG_FILE = "wednesday.log"
logger = logging.getLogger("WednesdayMain")
logger.setLevel(logging.ERROR)
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

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
    if wakeword_models is None:
        wakeword_models = ["wednesday.onnx"]

    if not os.path.exists(vosk_model_path):
        raise FileNotFoundError(
            f"Vosk model path '{vosk_model_path}' not found! Please check the folder."
        )

    print(f"Loading openWakeWord model(s): {wakeword_models}...")
    oww_model = Model(wakeword_models=wakeword_models, inference_framework="onnx")

    print(f"Loading Vosk STT model from '{vosk_model_path}'...")
    vosk_model = VoskModel(vosk_model_path)

    print("[Wednesday TTS Engine Active (Ultra-Fast Neural Voice: en-US-AvaNeural)]")
    tts_handler.get_speaker()

    vad = webrtcvad.Vad(vad_mode)
    vad_chunk_samples = 480
    vad_chunk_bytes = vad_chunk_samples * 2

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
    print(" Wednesday Voice Control System Active")
    print(f" Wake word(s): {', '.join(clean_model_names)}")
    print(f" Detection threshold: {threshold}")
    print(f" Dynamic Listening: Active ({silence_duration}s silence threshold)")
    print(" Press Ctrl+C to exit.")
    print("=" * 60 + "\n")

    try:
        while True:
            raw_data = mic_stream.read(chunk_size, exception_on_overflow=False)
            audio_frame = np.frombuffer(raw_data, dtype=np.int16)

            predictions = oww_model.predict(audio_frame)

            triggered_word = None
            for model_name, score in predictions.items():
                if score >= threshold:
                    clean_name = os.path.basename(model_name).replace(".onnx", "").replace(".tflite", "")
                    triggered_word = clean_name
                    print(f"\n[Wake Word]: Detected [{clean_name}]")
                    break

            if triggered_word:
                t_pipeline_start = time.perf_counter()
                print("[Listening]: Speak your command now...")

                recognizer = KaldiRecognizer(vosk_model, sample_rate)

                speech_started = False
                silent_chunks_count = 0
                max_silent_chunks = int(silence_duration / 0.03)

                t_rec_start = time.perf_counter()
                start_time = time.time()
                while True:
                    now = time.time()
                    elapsed = now - start_time

                    if elapsed >= max_recording_time:
                        break

                    if not speech_started and elapsed >= initial_speech_timeout:
                        print("[Listening]: No speech detected.")
                        break

                    cmd_data = mic_stream.read(vad_chunk_samples, exception_on_overflow=False)
                    if len(cmd_data) != vad_chunk_bytes:
                        continue

                    recognizer.AcceptWaveform(cmd_data)

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

                t_rec_duration = time.perf_counter() - t_rec_start

                t_stt_start = time.perf_counter()
                final_json = json.loads(recognizer.FinalResult())
                transcribed_text = final_json.get("text", "").strip()
                t_stt_duration = time.perf_counter() - t_stt_start

                if transcribed_text:
                    print(f"[User]: \"{transcribed_text}\"")
                    
                    response, llm_total_time, tool_time, attempt_logs = brain.process_command_with_timing(transcribed_text)
                    print(f"[Wednesday]: {response}\n")
                    
                    tts_gen_time, tts_play_time = tts_handler.speak(response)

                    t_total = time.perf_counter() - t_pipeline_start
                    perceived_start = t_stt_duration + llm_total_time + tool_time + tts_gen_time

                    print("=" * 68)
                    print(" PIPELINE TIMING TELEMETRY REPORT")
                    print("=" * 68)
                    print(f" 1. Recording / VAD Listening      : {t_rec_duration:6.3f} s")
                    print(f" 2. Vosk STT Transcription          : {t_stt_duration:6.3f} s")
                    print(f" 3. Gemini LLM API Total Time       : {llm_total_time:6.3f} s  (Attempts: {len(attempt_logs)})")
                    for log_entry in attempt_logs:
                        att_num = log_entry['attempt']
                        m_name = log_entry['model']
                        status = log_entry['status']
                        iso_t = log_entry['isolated_time']
                        print(f"     └─ [Attempt {att_num}]: {m_name:32s} -> {status} in {iso_t:.3f} s")
                    print(f" 4. Tool Execution (if any)         : {tool_time:6.3f} s")
                    print(f" 5. TTS Audio Generation            : {tts_gen_time:6.3f} s")
                    print(f" 6. Audio Playback Duration         : {tts_play_time:6.3f} s")
                    print("-" * 68)
                    print(f" TOTAL PIPELINE TIME                : {t_total:6.3f} s")
                    print(f" TIME TO FIRST AUDIO SOUND (Delay)  : {perceived_start:6.3f} s")
                    print("=" * 68 + "\n")

                    if "Stopping Wednesday voice loop" in response or "Goodbye!" in response:
                        break
                else:
                    print("[Wednesday]: No speech recognized.\n")

                oww_model.reset()

    except KeyboardInterrupt:
        print("\nStopping Wednesday voice loop...")
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
    parser = argparse.ArgumentParser(description="Wednesday Assistant LLM-Driven Voice Control System")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["wednesday.onnx"],
        help="Wake word models to use (e.g. wednesday.onnx)",
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
