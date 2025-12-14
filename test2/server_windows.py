import asyncio
import json
import logging
import os
from collections import deque

import numpy as np
import torch
import websockets
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

PORT = int(os.getenv("PORT", "8765"))
MODEL_ID = os.getenv("MODEL_ID", "GoranS/whisper-large-v3-turbo-hr-parla")

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 2

# Koliko audio-a skupimo prije transkripcije (sekunde)
CHUNK_SECONDS = float(os.getenv("CHUNK_SECONDS", "1.2"))
MIN_FLUSH_SECONDS = float(os.getenv("MIN_FLUSH_SECONDS", "0.6"))

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

logger.info("Model loaded (NO pipeline, NO torchcodec)")

# --------------------------------------------------
# TRANSCRIPTION
# --------------------------------------------------

def transcribe_audio_float32_mono(audio: np.ndarray, sample_rate: int) -> str:
    """
    audio: float32 mono in [-1, 1]
    """
    if audio.size == 0:
        return ""

    # Whisper expects 16kHz; Vapi obično šalje 16000.
    if sample_rate != 16000:
        logger.warning(f"Unexpected sample_rate={sample_rate}. Expected 16000.")
        return ""

    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")

    # neki procesori daju input_features, neki input_values; za Whisper je input_features
    if hasattr(inputs, "input_features") and inputs.input_features is not None:
        feats = inputs.input_features.to(device, dtype=dtype)
        with torch.no_grad():
            generated_ids = model.generate(feats, max_new_tokens=128)
    else:
        # fallback (rijetko)
        vals = inputs.input_values.to(device, dtype=dtype)
        with torch.no_grad():
            generated_ids = model.generate(vals, max_new_tokens=128)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return (text or "").strip()

# --------------------------------------------------
# VAPI WS HANDLER
# --------------------------------------------------

async def handle_vapi_connection(websocket):
    logger.info("Vapi connection opened")

    sample_rate = DEFAULT_SAMPLE_RATE
    channels = DEFAULT_CHANNELS

    # Buffer za mono int16 sampleove
    pcm_parts = deque()
    buffered_samples = 0

    target_samples = int(DEFAULT_SAMPLE_RATE * CHUNK_SECONDS)
    min_flush_samples = int(DEFAULT_SAMPLE_RATE * MIN_FLUSH_SECONDS)

    try:
        async for message in websocket:
            # ---- JSON control ----
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                except Exception:
                    continue

                if data.get("type") == "start":
                    sample_rate = int(data.get("sampleRate", sample_rate))
                    channels = int(data.get("channels", channels))

                    target_samples = int(sample_rate * CHUNK_SECONDS)
                    min_flush_samples = int(sample_rate * MIN_FLUSH_SECONDS)

                    logger.info(
                        f"START received | sample_rate={sample_rate} channels={channels}"
                    )
                continue

            # ---- binary audio PCM16 ----
            if not isinstance(message, (bytes, bytearray)):
                continue

            pcm = np.frombuffer(message, dtype=np.int16)

            # Stereo interleaved -> uzmi customer kanal (0)
            if channels == 2 and pcm.size >= 2:
                pcm = pcm.reshape(-1, 2)[:, 0]

            pcm_parts.append(pcm)
            buffered_samples += pcm.shape[0]

            if buffered_samples >= target_samples:
                chunk = np.concatenate(list(pcm_parts))
                pcm_parts.clear()
                buffered_samples = 0

                audio = chunk.astype(np.float32) / 32768.0

                text = await asyncio.to_thread(
                    transcribe_audio_float32_mono, audio, sample_rate
                )

                if text:
                    logger.info(f"TRANSCRIPT: {text}")
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "transcriber-response",
                                "transcription": text,
                                "channel": "customer",
                            }
                        )
                    )

    except websockets.exceptions.ConnectionClosed:
        logger.info("Vapi connection closed")

    except Exception:
        logger.exception("Connection handler error")

    finally:
        # Flush ako je nešto ostalo
        try:
            if buffered_samples >= min_flush_samples:
                chunk = np.concatenate(list(pcm_parts))
                audio = chunk.astype(np.float32) / 32768.0
                text = await asyncio.to_thread(
                    transcribe_audio_float32_mono, audio, sample_rate
                )
                if text:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "transcriber-response",
                                "transcription": text,
                                "channel": "customer",
                            }
                        )
                    )
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
    logger.info("Waiting for connections from Vapi...")
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
