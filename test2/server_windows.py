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

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 2

LANGUAGE = os.getenv("LANGUAGE", "hr")   # "hr" / "en" / etc
TASK = os.getenv("TASK", "transcribe")   # "transcribe" / "translate"

MIN_CHUNK_SECONDS = float(os.getenv("MIN_CHUNK_SECONDS", "1.0"))
MAX_CHUNK_SECONDS = float(os.getenv("MAX_CHUNK_SECONDS", "4.0"))
SILENCE_FLUSH_MS = int(os.getenv("SILENCE_FLUSH_MS", "450"))

VAD_RMS_THRESHOLD = float(os.getenv("VAD_RMS_THRESHOLD", "0.008"))

NUM_BEAMS = int(os.getenv("NUM_BEAMS", "1"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", "2"))
DEDUP_WINDOW = int(os.getenv("DEDUP_WINDOW", "3"))
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
# DEVICE + MODEL
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

logger.info("Model loaded (NO pipeline, NO torchcodec)")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def pcm16_to_float32(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / 32768.0

def rms_energy(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def now_ms() -> int:
    return int(time.time() * 1000)

# --------------------------------------------------
# TRANSCRIPTION (robust across Whisper variants)
# --------------------------------------------------

def transcribe_audio_float32_mono(audio: np.ndarray, sample_rate: int) -> str:
    if audio.size == 0:
        return ""

    if sample_rate != 16000:
        logger.warning(f"Unexpected sample_rate={sample_rate}. Expected 16000.")
        return ""

    # Some processors accept language/task here, some don't.
    # We'll try them, and fall back safely.
    proc_kwargs = dict(
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Try to pass language/task (works for many Whisper processors)
    tried_lang_task = False
    try:
        inputs = processor(audio, language=LANGUAGE, task=TASK, **proc_kwargs)
        tried_lang_task = True
    except TypeError:
        inputs = processor(audio, **proc_kwargs)

    input_features = getattr(inputs, "input_features", None)
    if input_features is None:
        # very rare fallback
        input_values = getattr(inputs, "input_values", None)
        if input_values is None:
            return ""
        input_tensor = input_values.to(device, dtype=dtype)
    else:
        input_tensor = input_features.to(device, dtype=dtype)

    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    gen_kwargs = dict(
        max_new_tokens=128,
        num_beams=NUM_BEAMS,
        temperature=TEMPERATURE,
    )

    # IMPORTANT:
    # Do NOT pass forced_decoder_ids here because some Whisper wrappers reject it,
    # causing: "model_kwargs not used: forced_decoder_ids".
    # Language/task is handled via processor() above when supported.

    with torch.no_grad():
        try:
            generated_ids = model.generate(
                input_tensor,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        except Exception as e:
            # Fallback if language/task attempt caused issues indirectly
            if tried_lang_task:
                logger.warning(f"Generate failed after language/task attempt; retrying without them. ({e})")
                inputs = processor(audio, **proc_kwargs)
                input_features = getattr(inputs, "input_features", None)
                if input_features is not None:
                    input_tensor = input_features.to(device, dtype=dtype)
                else:
                    input_values = getattr(inputs, "input_values", None)
                    if input_values is None:
                        return ""
                    input_tensor = input_values.to(device, dtype=dtype)

                attention_mask = getattr(inputs, "attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                generated_ids = model.generate(
                    input_tensor,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            else:
                raise

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return (text or "").strip()

# --------------------------------------------------
# CONNECTION STATE
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
# FLUSH / SEND
# --------------------------------------------------

async def maybe_flush(state: ConnState, websocket, reason: str):
    min_samples = int(state.sample_rate * MIN_CHUNK_SECONDS)
    if state.buffered_samples < min_samples:
        return

    pcm = np.concatenate(list(state.pcm_parts)) if state.pcm_parts else np.zeros(0, dtype=np.int16)

    # trim trailing silence
    if state.silence_samples > 0 and state.silence_samples < pcm.shape[0]:
        pcm = pcm[: pcm.shape[0] - state.silence_samples]

    state.pcm_parts.clear()
    state.buffered_samples = 0
    state.silence_samples = 0
    state.speech_started = False

    audio = pcm16_to_float32(pcm)

    text = await asyncio.to_thread(transcribe_audio_float32_mono, audio, state.sample_rate)

    if not text or len(text.strip()) < MIN_TEXT_CHARS:
        return

    t = text.strip()
    now = now_ms()

    # dedupe
    if t in state.last_texts and (now - state.last_send_ms) < DEDUP_MIN_INTERVAL_MS:
        return

    state.last_texts.append(t)
    state.last_send_ms = now

    logger.info(f"TRANSCRIPT ({reason}): {t}")
    await websocket.send(json.dumps({
        "type": "transcriber-response",
        "transcription": t,
        "channel": "customer",
    }))

# --------------------------------------------------
# WS HANDLER
# --------------------------------------------------

async def handle_vapi_connection(websocket):
    logger.info("Vapi connection opened")
    state = ConnState()

    max_samples = int(state.sample_rate * MAX_CHUNK_SECONDS)
    silence_flush_samples = int(state.sample_rate * (SILENCE_FLUSH_MS / 1000.0))

    try:
        async for message in websocket:
            # JSON control
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                except Exception:
                    continue

                if data.get("type") == "start":
                    state.sample_rate = int(data.get("sampleRate", state.sample_rate))
                    state.channels = int(data.get("channels", state.channels))

                    max_samples = int(state.sample_rate * MAX_CHUNK_SECONDS)
                    silence_flush_samples = int(state.sample_rate * (SILENCE_FLUSH_MS / 1000.0))

                    logger.info(f"START received | sample_rate={state.sample_rate} channels={state.channels}")
                continue

            # binary PCM16
            if not isinstance(message, (bytes, bytearray)):
                continue

            pcm = np.frombuffer(message, dtype=np.int16)

            # stereo interleaved -> take customer ch0
            if state.channels == 2 and pcm.size >= 2:
                pcm = pcm.reshape(-1, 2)[:, 0]

            audio = pcm16_to_float32(pcm)
            e = rms_energy(audio)
            is_speech = e >= VAD_RMS_THRESHOLD

            if is_speech:
                state.speech_started = True

            if not state.speech_started:
                continue  # ignore pre-speech silence

            state.pcm_parts.append(pcm)
            state.buffered_samples += pcm.shape[0]

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
