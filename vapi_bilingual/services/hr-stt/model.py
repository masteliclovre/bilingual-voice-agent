import os
from faster_whisper import WhisperModel


device = os.getenv("DEVICE", "cuda")
compute_type = os.getenv("COMPUTE_TYPE", "float16")
model_path = os.getenv("MODEL_PATH", "medium")


model = WhisperModel(model_path, device=device, compute_type=compute_type)