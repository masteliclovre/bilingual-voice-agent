from __future__ import annotations
import asyncio
import base64
import json
import time
from typing import Callable, Literal
import numpy as np


from audio.mu_law import mulaw_to_pcm16

class Leg:
    """One call leg (a single Twilio Media Stream WebSocket)."""


    def __init__(self, name: str, send_json: Callable[[dict], None]):
        self.name = name
        self._send_json = send_json
        self.in_pcm_q: asyncio.Queue[bytes] = asyncio.Queue() # PCM16 8kHz
        self._tts_playing = False
        self._tts_lock = asyncio.Lock()
        self._last_voice_ts = 0.0


    async def on_twilio_event(self, evt: dict, on_barge_in: Callable[[], None] | None = None):
        if evt.get("event") == "media":
            payload_b64 = evt["media"]["payload"]
            mulaw = np.frombuffer(base64.b64decode(payload_b64), dtype=np.uint8)
            pcm16 = mulaw_to_pcm16(mulaw)
            # Simple VAD: if RMS > threshold, trigger barge-in
            rms = float(np.sqrt(np.mean((pcm16.astype(np.float32)) ** 2)))
            if rms > 600: # empirical threshold for μ-law-decoded PCM16
                now = time.time()
                if now - self._last_voice_ts > 0.15: # debounce
                    if on_barge_in:
                        await on_barge_in()
                self._last_voice_ts = now
            await self.in_pcm_q.put(pcm16.tobytes())
    
    async def clear_audio(self):
        await self._send_json({"event": "clear"})


    async def send_mulaw_stream(self, mulaw_bytes: bytes, chunk_ms: int = 20):
        """Send mu-law audio to Twilio in paced 20ms chunks (160 bytes/chunk @ 8kHz)."""
        if not mulaw_bytes:
            return
        chunk_size = int(8_000 * (chunk_ms / 1000.0)) # bytes, since 8-bit mu-law
        for i in range(0, len(mulaw_bytes), chunk_size):
            payload = base64.b64encode(mulaw_bytes[i:i+chunk_size]).decode()
            await self._send_json({"event": "media", "media": {"payload": payload}})
            await asyncio.sleep(chunk_ms / 1000.0)