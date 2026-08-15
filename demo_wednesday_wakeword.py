import openwakeword
import numpy as np

# Load your custom 'Wednesday' wake word model
model_path = "wednesday.onnx"
oww_model = openwakeword.Model(
    wakeword_models=[model_path],
    inference_framework="onnx"
)

print("OpenWakeWord loaded successfully!")
print("Active models:", list(oww_model.models.keys()))

# Example: Feed 1280 samples (80ms at 16kHz mono audio) to get predictions
dummy_audio = np.random.randint(-1000, 1000, 1280).astype(np.int16)
prediction = oww_model.predict(dummy_audio)
print("Sample prediction scores:", prediction)
