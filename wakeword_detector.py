import time
import argparse
import numpy as np
import pyaudio
from openwakeword.model import Model


def listen_for_wakeword(
    model_names: list[str] = None,
    threshold: float = 0.5,
    cooldown_seconds: float = 2.0,
    chunk_size: int = 1280,
    sample_rate: int = 16000,
):
    """
    Listens to the microphone stream continuously and feeds audio chunks to openWakeWord.
    
    :param model_names: List of wake word model names to load (default: ['hey_jarvis']).
    :param threshold: Prediction score threshold between 0.0 and 1.0 to trigger detection.
    :param cooldown_seconds: Seconds to pause detection after a trigger to prevent repeated prints.
    :param chunk_size: Number of audio samples per frame (1280 = 80ms at 16kHz).
    :param sample_rate: Audio sampling rate in Hz (16000Hz required by openWakeWord).
    """
    if model_names is None:
        model_names = ["hey_jarvis"]

    print(f"Loading openWakeWord model(s): {model_names}...")
    oww_model = Model(wakeword_models=model_names, inference_framework="onnx")

    # PyAudio setup
    audio = pyaudio.PyAudio()
    mic_stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )

    print("\n" + "=" * 50)
    print(f" Listening for wake word(s): {', '.join(model_names)}")
    print(f" Detection threshold: {threshold}")
    print(" Press Ctrl+C to stop listening.")
    print("=" * 50 + "\n")

    last_detection_time = 0.0

    try:
        while True:
            # Read 16-bit PCM audio chunk from microphone
            raw_data = mic_stream.read(chunk_size, exception_on_overflow=False)
            
            # Convert raw bytes to 16-bit signed integer numpy array
            audio_frame = np.frombuffer(raw_data, dtype=np.int16)

            # Feed audio frame to openWakeWord model
            predictions = oww_model.predict(audio_frame)

            current_time = time.time()

            # Check prediction scores against threshold
            for model_name, score in predictions.items():
                if score >= threshold:
                    if current_time - last_detection_time >= cooldown_seconds:
                        print(f"✨ Wake word detected! [{model_name}] (Score: {score:.3f})")
                        last_detection_time = current_time

    except KeyboardInterrupt:
        print("\nStopping wake word listener...")
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        audio.terminate()
        print("Microphone stream closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="openWakeWord Voice Detection Loop")
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
        "--cooldown",
        type=float,
        default=2.0,
        help="Cooldown period in seconds between detections (default: 2.0)",
    )

    args = parser.parse_args()
    listen_for_wakeword(
        model_names=args.models,
        threshold=args.threshold,
        cooldown_seconds=args.cooldown,
    )
