import asyncio
import json
import logging
import os

import numpy as np
import torch
import websockets
from transformers import pipeline

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

PORT = int(os.getenv("PORT", 8765))
MODEL_ID = os.getenv(
    "MODEL_ID",
    "GoranS/whisper-large-v3-turbo-hr-parla",
)

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 2

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
logger.info(f"Using device: {device}")
logger.info(f"Loading model: {MODEL_ID}")

transcriber = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    device=0 if device == "cuda" else -1,
)

logger.info("Model loaded")

# --------------------------------------------------
# TRANSCRIPTION FUNCTION
# --------------------------------------------------

def transcribe_audio(audio: np.ndarray, sample_rate: int) -> str:
    """
    audio: float32 mono [-1, 1]
    """
    if audio.size == 0:
        return ""

    result = transcriber(
        audio,
        sampling_rate=sample_rate,
        return_timestamps=False,
    )

    if isinstance(result, dict):
        return result.get("text", "").strip()

    return ""

# --------------------------------------------------
# WEBSOCKET HANDLER (VAPI)
# --------------------------------------------------

async def handle_vapi_connection(websocket):
    logger.info("Vapi connection opened")

    sample_rate = DEFAULT_SAMPLE_RATE
    channels = DEFAULT_CHANNELS

    try:
        async for message in websocket:

            # ----------------------------
            # START / CONTROL (JSON)
            # ----------------------------
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                except Exception:
                    continue

                msg_type = data.get("type")

                if msg_type == "start":
                    sample_rate = int(data.get("sampleRate", sample_rate))
                    channels = int(data.get("channels", channels))
                    logger.info(
                        f"START received | sample_rate={sample_rate} channels={channels}"
                    )

                continue

            # ----------------------------
            # AUDIO (BINARY PCM16)
            # ----------------------------
            if not isinstance(message, (bytes, bytearray)):
                continue

            logger.debug(f"AUDIO BYTES: {len(message)}")

            pcm = np.frombuffer(message, dtype=np.int16)

            # Vapi šalje stereo interleaved: L,R,L,R...
            if channels == 2 and pcm.size >= 2:
                pcm = pcm.reshape(-1, 2)[:, 0]  # customer channel

            audio = pcm.astype(np.float32) / 32768.0

            # Transcribe (off event loop)
            text = await asyncio.to_thread(
                transcribe_audio, audio, sample_rate
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

    except Exception as e:
        logger.exception("Connection handler error")

# --------------------------------------------------
# SERVER
# --------------------------------------------------

async def main():
    logger.info(
        f"Starting Custom Transcriber WebSocket server on port {PORT}"
    )

    server = await websockets.serve(
        handle_vapi_connection,
        "0.0.0.0",
        PORT,
        origins=None,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
    )

    logger.info(
        f"Server is listening on ws://0.0.0.0:{PORT}/api/custom-transcriber"
    )
    logger.info("Waiting for connections from Vapi...")

    await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
