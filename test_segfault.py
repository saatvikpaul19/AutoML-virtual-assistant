import os
import sys

# Try to replicate the crash
print("--- Diagnostic Script: Testing for Segfaults (Python 3.11 / myenv) ---")

try:
    print("1. Testing numpy...")
    import numpy as np
    a = np.random.rand(100, 100)
    b = np.dot(a, a)
    print(f"   Numpy OK (version: {np.__version__})")
except Exception as e:
    print(f"   Numpy FAILED: {e}")

try:
    print("2. Testing torch...")
    import torch
    x = torch.rand(5, 3)
    print(f"   Torch OK (version: {torch.__version__}, Device: {torch.device('cpu')})")
    print(f"   MPS (Metal) available: {torch.backends.mps.is_available()}")
except Exception as e:
    print(f"   Torch FAILED: {e}")

try:
    print("3. Testing whisper...")
    import whisper
    # Just load the model, don't necessarily run it yet
    model = whisper.load_model("tiny")
    print("   Whisper load OK")
except Exception as e:
    print(f"   Whisper FAILED: {e}")

try:
    print("4. Testing onnxruntime (Used by OpenWakeWord)...")
    import onnxruntime
    print(f"   Onnxruntime OK (version: {onnxruntime.get_version_string()})")
except Exception as e:
    print(f"   Onnxruntime FAILED: {e}")

try:
    print("5. Testing sounddevice (audio drivers)...")
    import sounddevice as sd
    devices = sd.query_devices()
    print(f"   Sounddevice OK. Found {len(devices)} devices.")
except Exception as e:
    print(f"   Sounddevice FAILED: {e}")

print("--- End of basic diagnostics ---")
