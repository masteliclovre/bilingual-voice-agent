import asyncio, json, os, time
from typing import Optional
import numpy as np
from fastapi import FastAPI, WebSocket
from dotenv import load_dotenv
from model import model


load_dotenv()


SAMPLE_RATE = 16000
CHUNK_SECONDS = float(os.getenv("CHUNK_SECONDS", "1.0"))
VAD_FILTER = os.getenv("VAD_FILTER", "true").lower() == "true"


app = FastAPI()


@app.websocket('/stream')
async def stream(ws: WebSocket):
    await ws.accept()
    buf = bytearray()
    last_emit = time.time()


    async def transcribe_and_emit(data: bytes, final=False):
        if not data:
            return
        # Convert bytes (PCM16 mono) to float32 [-1,1]
        pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        segments, info = model.transcribe(pcm, language='hr', vad_filter=VAD_FILTER, word_timestamps=False, beam_size=1)
        text = ''.join([seg.text for seg in segments]).strip()
        if not text:
            return
    # crude confidence proxy: average no_speech_prob complement
    conf = max(0.0, min(1.0, 1.0 - (getattr(info, 'no_speech_prob', 0.0) or 0.0)))
    msg = { 'type': 'final' if final else 'partial', 'text': text, 'confidence': conf }
    await ws.send_text(json.dumps(msg, ensure_ascii=False))


    try:
        while True:
            data = await ws.receive_bytes()
            buf.extend(data)
            # every CHUNK_SECONDS, emit partial
            samples_needed = int(SAMPLE_RATE * CHUNK_SECONDS) * 2 # bytes
            if len(buf) >= samples_needed:
                chunk = bytes(buf[:samples_needed]); del buf[:samples_needed]
                await transcribe_and_emit(chunk, final=False)
    except Exception:
        # flush remainder as final
        if buf:
            await transcribe_and_emit(bytes(buf), final=True)