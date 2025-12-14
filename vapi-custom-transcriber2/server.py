#!/usr/bin/env python3
"""
Vapi Custom Transcriber using a Hugging Face Whisper model (Croatian by default).

- Accepts WebSocket connections from Vapi
- Receives interleaved stereo PCM16 audio frames
- Transcribes each channel separately (customer=left, assistant=right)
- Responds with Vapi's `transcriber-response` messages

Notes:
- Model is loaded ONCE at startup (not per connection) for performance.
- Transcription runs in a worker thread so the asyncio event loop stays responsive.
"""

import asyncio
import json
import logging
import os
from typing import Optional

import numpy as np
import torch
import websockets
from dotenv import load_dotenv
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vapi-custom-transcriber")

# Configuration
PORT = int(os.getenv("PORT", "8765"))
MODEL_ID = os.getenv("MODEL_ID", "lovremastelic/bva")

# Allow forcing device (useful for debugging)
FORCE_DEVICE = os.getenv("DEVICE", "").strip().lower()
if FORCE_DEVICE in {"cpu", "cuda"}:
    DEVICE = FORCE_DEVICE
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Audio configuration from Vapi
SAMPLE_RATE = 16000
CHANNELS = 2
BYTES_PER_SAMPLE = 2  # 16-bit PCM

logger.info(f"Device selected: {DEVICE}")
logger.info(f"Model: {MODEL_ID}")


class WhisperTranscriber:
    """Handles audio transcription using a Hugging Face Whisper(-style) model."""

    def __init__(self):
        self.pipe = None
        self._lock = asyncio.Lock()

    def load(self):
        """Load model + processor and create pipeline."""
        logger.info("Loading ASR model (this can take a while on first boot)...")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(DEVICE)

        processor = AutoProcessor.from_pretrained(MODEL_ID)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=TORCH_DTYPE,
            device=0 if DEVICE == "cuda" else -1,  # pipeline expects int device index
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=8,
            return_timestamps=False,
        )

        logger.info("ASR model loaded.")

    @staticmethod
    def pcm_to_numpy(pcm_data: bytes, channel: int) -> np.ndarray:
        """Convert interleaved stereo PCM16 to float32 mono array for `channel`."""
        bytes_per_frame = CHANNELS * BYTES_PER_SAMPLE
        if len(pcm_data) < bytes_per_frame:
            return np.array([], dtype=np.float32)

        # Trim to full frames
        if len(pcm_data) % bytes_per_frame != 0:
            pcm_data = pcm_data[: (len(pcm_data) // bytes_per_frame) * bytes_per_frame]

        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        audio_stereo = audio_int16.reshape(-1, CHANNELS)
        audio_mono = audio_stereo[:, channel]

        # Normalize to [-1, 1]
        return (audio_mono.astype(np.float32) / 32768.0).copy()

    async def transcribe(self, audio_data: bytes, channel: int) -> str:
        """Transcribe PCM bytes for a channel; runs in a thread to avoid blocking."""
        if self.pipe is None:
            return ""

        audio = self.pcm_to_numpy(audio_data, channel)
        if audio.size == 0:
            return ""

        async with self._lock:
            # Transformers pipelines can use a lot of GPU memory; locking is safer for multi-connection use.
            result = await asyncio.to_thread(self.pipe, audio)

        text = (result.get("text") or "").strip()
        return text


class AudioBuffer:
    """Buffers stereo audio before transcription."""

    def __init__(self, min_duration_seconds: float = 1.5):
        self.buffer = bytearray()
        self.min_samples = int(SAMPLE_RATE * min_duration_seconds)
        self.min_bytes = self.min_samples * BYTES_PER_SAMPLE  # per-channel bytes (approx)

    def add_audio(self, data: bytes):
        self.buffer.extend(data)

    def should_transcribe(self) -> bool:
        return len(self.buffer) >= (self.min_bytes * CHANNELS)

    def get_and_clear(self) -> bytes:
        data = bytes(self.buffer)
        self.buffer.clear()
        return data

    def clear(self):
        self.buffer.clear()


TRANSCRIBER: Optional[WhisperTranscriber] = None


async def handle_vapi_connection(websocket):
    """Handle one WebSocket connection from Vapi."""
    logger.info("New Vapi connection")

    buffer = AudioBuffer(min_duration_seconds=float(os.getenv("MIN_CHUNK_SECONDS", "1.5")))

    try:
        async for message in websocket:
            if isinstance(message, str):
                # Vapi sends JSON control frames (e.g. start)
                try:
                    data = json.loads(message)
                    if data.get("type") == "start":
                        logger.info(f"Start: sampleRate={data.get('sampleRate')} channels={data.get('channels')}")
                except json.JSONDecodeError:
                    logger.warning("Received non-JSON text frame")
                continue

            if not isinstance(message, (bytes, bytearray)):
                continue

            # Binary audio data
            buffer.add_audio(message)

            if not buffer.should_transcribe():
                continue

            audio_data = buffer.get_and_clear()

            # Transcribe both channels
            for channel in (0, 1):
                text = await TRANSCRIBER.transcribe(audio_data, channel)  # type: ignore[union-attr]
                if not text:
                    continue

                channel_name = "customer" if channel == 0 else "assistant"
                payload = {"type": "transcriber-response", "transcription": text, "channel": channel_name}
                await websocket.send(json.dumps(payload))
                logger.info(f"{channel_name}: {text}")

    except websockets.exceptions.ConnectionClosed:
        logger.info("Connection closed")
    except Exception:
        logger.exception("Connection error")
    finally:
        buffer.clear()


async def main():
    global TRANSCRIBER
    TRANSCRIBER = WhisperTranscriber()
    TRANSCRIBER.load()

    logger.info(f"Listening on 0.0.0.0:{PORT} (ws)")
    async with websockets.serve(
        lambda ws, _path=None: handle_vapi_connection(ws),
        "0.0.0.0",
        PORT,
        ping_interval=20,
        ping_timeout=20,
        max_size=8 * 1024 * 1024,
    ):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
