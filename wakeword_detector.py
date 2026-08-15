"""
Wednesday Assistant - Legacy Entry Point Wrapper
=================================================

Redirects execution to main.py voice control pipeline.
"""

import sys
import main

if __name__ == "__main__":
    parser = main.argparse.ArgumentParser(description="Jarvis LLM-Driven Voice Control System")
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
    main.listen_and_transcribe(
        wakeword_models=args.models,
        threshold=args.threshold,
        silence_duration=args.silence_duration,
        max_recording_time=args.max_duration,
        vosk_model_path=args.vosk_model,
    )
