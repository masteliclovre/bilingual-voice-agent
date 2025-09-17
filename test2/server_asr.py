# server_asr.py
import os, io, wave
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from scipy.signal import resample

TARGET_SR = 16000

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "GoranS/whisper-large-v3-turbo-hr-parla-ctranslate2")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")        # use the GPU
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16")   # float16 on GPU
CPU_THREADS = int(os.getenv("CPU_THREADS", "8"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))

app = FastAPI()
model = None

def _resample_to_16k(audio_np: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return audio_np
    target_len = int(len(audio_np) * TARGET_SR / src_sr)
    if target_len <= 0:
        return np.zeros(1, dtype=np.float32)
    return resample(audio_np, target_len).astype(np.float32)

@app.on_event("startup")
def _load_model():
    global model
    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
        cpu_threads=CPU_THREADS,
        num_workers=NUM_WORKERS,
    )

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/transcribe")
async def transcribe(file: UploadFile | None = File(None), request: Request | None = None):
    # Accept either multipart file=… or raw body
    if file is not None:
        data = await file.read()
    else:
        data = await request.body()

    if not data:
        return JSONResponse({"error": "no audio"}, status_code=400)

    # Expect 16 kHz mono WAV (int16), but resample if needed.
    try:
        with wave.open(io.BytesIO(data), 'rb') as wf:
            sr = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if sr != TARGET_SR:
            audio = _resample_to_16k(audio, sr)
    except Exception:
        return JSONResponse({"error": "invalid wav"}, status_code=400)

    segments, info = model.transcribe(
        audio=audio,
        beam_size=1,                  # greedy for speed
        vad_filter=True,
        temperature=0.0,
        language=None,                # auto
        condition_on_previous_text=False,
        word_timestamps=False,
        without_timestamps=True,
    )
    text = "".join(s.text for s in segments).strip()
    lang = getattr(info, "language", None) or ""
    return {"text": text, "lang": lang}
