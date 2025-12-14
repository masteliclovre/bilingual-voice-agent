import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import websockets
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# --------------------------------------------------
# ENV CONFIG (tune these)
# --------------------------------------------------

PORT = int(os.getenv("PORT", "8765"))
MODEL_ID = os.getenv("MODEL_ID", "GoranS/whisper-large-v3-turbo-hr-parla")

# Vapi default
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 2

# Language / task (helps a lot)
LANGUAGE = os.getenv("LANGUAGE", "hr")          # "hr" or "en" etc.
TASK = os.getenv("TASK", "transcribe")          # "transcribe" or "translate"

# Chunking
MIN_CHUNK_SECONDS = float(os.getenv("MIN_CHUNK_SECONDS", "1.0"))   # don't transcribe under this (unless forced)
MAX_CHUNK_SECONDS = float(os.getenv("MAX_CHUNK_SECONDS", "4.0"))   # force transcribe if buffer grows this big
SILENCE_FLUSH_MS = int(os.getenv("SILENCE_FLUSH_MS", "450"))       # flush after this much silence

# Simple energy VAD
VAD_RMS_THRESHOLD = float(os.getenv("VAD_RMS_THRESHOLD", "0.008")) # raise to be stricter, lower to be more sensitive

# Whisper generate params (quality vs latency)
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "1"))       # 1 fastest; 2-4 can improve quality
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

# Dedupe / spam control
MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", "2"))
DEDUP_WINDOW = int(os.getenv("DEDUP_WINDOW", "3"))                 # don't send same text if it appeared in last N sends
DEDUP_MIN_INTERVAL_MS = int(os.getenv("DEDUP_MIN_INTERVAL_MS", "600"))

# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# DEVICE + MODEL (LOAD ONCE)
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

logger.info(f"Using device: {device}")
logger.info(f"Loading model: {MODEL_ID}")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)
model.to(device)
model.eval()

# Decoder prompt ids (forces language/task for Whisper-family models)
forced_decoder_ids = None
try:
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
    logger.info(f"Forced decoder prompt: language={LANGUAGE} task={TASK}")
except Exception as e:
    logger.warning(f"Could not set forced_decoder_ids ({e}). Continuing without it.")

logger.info("Model loaded (NO pipeline, NO torchcodec)")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def pcm16_to_float32(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / 32768.0

def rms_energy(x: np.ndarray) -> float:
    # x float32 [-1,1]
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def now_ms() -> int:
    return int(time.time() * 1000)

def transcribe_audio_float32_mono(audio: np.ndarray, sample_rate: int) -> str:
    """
    audio: float32 mono [-1,1]
    """
    if audio.size == 0:
        return ""

    if sample_rate != 16000:
        logger.warning(f"Unexpected sample_rate={sample_rate}. Expected 16000.")
        return ""

    # return_attention_mask removes the warning you saw
    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_features = inputs.input_features.to(device, dtype=dtype)
    attention_mask = None
    if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
        attention_mask = inputs.attention_mask.to(device)

    gen_kwargs = dict(
        max_new_tokens=128,
        num_beams=NUM_BEAMS,
        temperature=TEMPERATURE,
    )
    if forced_decoder_ids is not None:
        gen_kwargs["forced_decoder_ids"] = forced_decoder_ids

    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return (text or "").strip()

# --------------------------------------------------
# PER-CONNECTION STATE
# --------------------------------------------------

@dataclass
class ConnState:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS

    pcm_parts: deque = None
    buffered_samples: int = 0
    silence_samples: int = 0
    speech_started: bool = False

    last_texts: deque = None
    last_send_ms: int = 0

    def __post_init__(self):
        self.pcm_parts = deque()
        self.last_texts = deque(maxlen=DEDUP_WINDOW)

# --------------------------------------------------
# WS HANDLER
# --------------------------------------------------

async def maybe_flush(state: ConnState, websocket, reason: str):
    """Transcribe and send if we have enough audio."""
    min_samples = int(state.sample_rate * MIN_CHUNK_SECONDS)
    if state.buffered_samples < min_samples:
        return

    # concat and trim trailing silence if we tracked it
    pcm = np.concatenate(list(state.pcm_parts)) if state.pcm_parts else np.zeros(0, dtype=np.int16)

    if state.silence_samples > 0 and state.silence_samples < pcm.shape[0]:
        pcm = pcm[: pcm.shape[0] - state.silence_samples]

    state.pcm_parts.clear()
    state.buffered_samples = 0
    state.silence_samples = 0
    state.speech_started = False

    audio = pcm16_to_float32(pcm)

    text = await asyncio.to_thread(transcribe_audio_float32_mono, audio, state.sample_rate)

    # basic filtering
    if not text or len(text.strip()) < MIN_TEXT_CHARS:
        return

    # dedupe: same text recently or too frequent
    t = text.strip()
    now = now_ms()
    if t in state.last_texts:
        # if it's repeating and too soon, drop it
        if now - state.last_send_ms < DEDUP_MIN_INTERVAL_MS:
            return

    state.last_texts.append(t)
    state.last_send_ms = now

    logger.info(f"TRANSCRIPT ({reason}): {t}")
    await websocket.send(json.dumps({
        "type": "transcriber-response",
        "transcription": t,
        "channel": "customer",
    }))

async def handle_vapi_connection(websocket):
    logger.info("Vapi connection opened")
    state = ConnState()

    max_samples = int(DEFAULT_SAMPLE_RATE * MAX_CHUNK_SECONDS)
    silence_flush_samples = int(DEFAULT_SAMPLE_RATE * (SILENCE_FLUSH_MS / 1000.0))

    try:
        async for message in websocket:
            # ---- control JSON ----
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                except Exception:
                    continue

                if data.get("type") == "start":
                    state.sample_rate = int(data.get("sampleRate", state.sample_rate))
                    state.channels = int(data.get("channels", state.channels))

                    # recompute thresholds if sr changed
                    max_samples = int(state.sample_rate * MAX_CHUNK_SECONDS)
                    silence_flush_samples = int(state.sample_rate * (SILENCE_FLUSH_MS / 1000.0))

                    logger.info(f"START received | sample_rate={state.sample_rate} channels={state.channels}")
                continue

            # ---- audio PCM16 ----
            if not isinstance(message, (bytes, bytearray)):
                continue

            pcm = np.frombuffer(message, dtype=np.int16)

            # stereo interleaved -> take customer channel 0
            if state.channels == 2 and pcm.size >= 2:
                pcm = pcm.reshape(-1, 2)[:, 0]

            audio = pcm16_to_float32(pcm)
            e = rms_energy(audio)

            is_speech = e >= VAD_RMS_THRESHOLD

            # start buffering only when speech begins (prevents random hallucinations on silence)
            if is_speech:
                state.speech_started = True

            if not state.speech_started:
                # ignore pure silence before speech starts
                continue

            state.pcm_parts.append(pcm)
            state.buffered_samples += pcm.shape[0]

            # silence tracking (for flush-on-silence)
            if is_speech:
                state.silence_samples = 0
            else:
                state.silence_samples += pcm.shape[0]

            # flush conditions
            if state.buffered_samples >= max_samples:
                await maybe_flush(state, websocket, reason="max_chunk")
                continue

            if state.silence_samples >= silence_flush_samples:
                await maybe_flush(state, websocket, reason="silence_flush")
                continue

    except websockets.exceptions.ConnectionClosed:
        logger.info("Vapi connection closed")

    except Exception:
        logger.exception("Connection handler error")

    finally:
        # final flush if any leftover
        try:
            await maybe_flush(state, websocket, reason="final_flush")
        except Exception:
            pass

# --------------------------------------------------
# SERVER
# --------------------------------------------------

async def main():
    logger.info(f"Starting Custom Transcriber WebSocket server on port {PORT}")
    await websockets.serve(
        handle_vapi_connection,
        "0.0.0.0",
        PORT,
        origins=None,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
    )
    logger.info(f"Server listening on ws://0.0.0.0:{PORT}/api/custom-transcriber")
    logger.info(
        "Tuning: "
        f"VAD_RMS_THRESHOLD={VAD_RMS_THRESHOLD} "
        f"MIN_CHUNK_SECONDS={MIN_CHUNK_SECONDS} "
        f"MAX_CHUNK_SECONDS={MAX_CHUNK_SECONDS} "
        f"SILENCE_FLUSH_MS={SILENCE_FLUSH_MS} "
        f"NUM_BEAMS={NUM_BEAMS} TEMP={TEMPERATURE} "
        f"LANGUAGE={LANGUAGE} TASK={TASK}"
    )
    logger.info("Waiting for connections from Vapi...")
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
