"""Microphone streaming client that talks to the realtime_proxy via WebSockets."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from elevenlabs import ElevenLabs


TARGET_SR = 16000
CHANNELS = 1
FRAME_DURATION_MS = int(os.getenv("FRAME_DURATION_MS", "20"))
SILENCE_TIMEOUT_SECS = float(os.getenv("SILENCE_TIMEOUT_SECS", "1.0"))
MIN_SPEECH_SECS = float(os.getenv("MIN_SPEECH_SECS", "0.35"))
RMS_THRESH = float(os.getenv("RMS_THRESH", "0.0025"))
RMS_HANGOVER = float(os.getenv("RMS_HANGOVER", "0.18"))
MAX_UTTERANCE_SECS = float(os.getenv("MAX_UTTERANCE_SECS", "45"))

PROXY_URL = os.getenv("REALTIME_PROXY_URL", "ws://localhost:8081")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "vFQACl5nAIV0owAavYxE")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

ASSISTANT_INSTRUCTIONS = os.getenv(
    "ASSISTANT_INSTRUCTIONS",
    "You are a friendly bilingual (Croatian and English) voice assistant."
)


def pcm16_from_float32(chunk: np.ndarray) -> bytes:
    """Convert float32 PCM (-1..1) to 16-bit little-endian bytes."""

    scaled = np.clip(chunk, -1.0, 1.0)
    return (scaled * 32767.0).astype(np.int16).tobytes()


def elevenlabs_tts(text: str) -> None:
    if not text:
        return

    if not ELEVENLABS_API_KEY:
        print("[warn] ELEVENLABS_API_KEY missing – skipping TTS playback.")
        return

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio_gen = client.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        optimize_streaming_latency="0",
        output_format="pcm_16000",
        text=text,
        model_id=ELEVENLABS_MODEL_ID,
    )
    pcm_bytes = b"".join(audio_gen)
    if not pcm_bytes:
        print("[warn] Empty audio from ElevenLabs")
        return

    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(pcm.reshape(-1, 1), TARGET_SR)
    sd.wait()


@dataclass
class UtteranceResult:
    transcript: str
    reply: str


async def receive_realtime(ws: "websockets.WebSocketClientProtocol") -> UtteranceResult:
    """Collect transcript/reply events for a single response."""

    transcript_parts: list[str] = []
    reply_parts: list[str] = []

    while True:
        message = await ws.recv()
        data = json.loads(message)
        event_type = data.get("type")

        if event_type == "input_audio_buffer.cleared":
            continue

        if event_type == "response.delta":
            delta = data.get("delta", {})
            if isinstance(delta, dict):
                maybe_text = delta.get("text") or ""
                if maybe_text:
                    reply_parts.append(maybe_text)
                if delta.get("transcript"):
                    transcript_parts.append(delta["transcript"])
            elif isinstance(delta, list):
                for item in delta:
                    if item.get("type") == "output_text_delta":
                        reply_parts.append(item.get("text", ""))
                    if item.get("type") == "transcript_delta":
                        transcript_parts.append(item.get("text", ""))
            continue

        if event_type == "response.output_text.delta":
            reply_parts.append(data.get("delta", {}).get("text", ""))
            continue

        if event_type == "response.refusal.delta":
            reply_parts.append(data.get("delta", {}).get("text", ""))
            continue

        if event_type == "response.completed":
            break

        if event_type == "error":
            raise RuntimeError(f"Realtime API error: {data}")

        # Some responses embed transcript chunks under conversation events
        if event_type == "conversation.item.input_audio_transcription.delta":
            transcript_parts.append(data.get("delta", {}).get("text", ""))

    return UtteranceResult(
        transcript="".join(transcript_parts).strip(),
        reply="".join(reply_parts).strip(),
    )


async def stream_utterance(ws: "websockets.WebSocketClientProtocol") -> Optional[UtteranceResult]:
    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
    finished = asyncio.Event()
    samples_streamed = 0

    def audio_callback(indata, frames, time_info, status):  # noqa: ARG001
        mono = indata.reshape(-1)
        rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-12)
        now = time.time()

        if not hasattr(audio_callback, "started"):
            audio_callback.started = False  # type: ignore[attr-defined]
            audio_callback.last_voice = now  # type: ignore[attr-defined]
            audio_callback.first_chunk = now  # type: ignore[attr-defined]

        if rms > RMS_THRESH:
            audio_callback.started = True  # type: ignore[attr-defined]
            audio_callback.last_voice = now  # type: ignore[attr-defined]

        if audio_callback.started:  # type: ignore[attr-defined]
            loop.call_soon_threadsafe(audio_queue.put_nowait, mono.copy())

        elapsed = now - getattr(audio_callback, "first_chunk", now)
        if (
            audio_callback.started  # type: ignore[attr-defined]
            and (now - audio_callback.last_voice) > max(SILENCE_TIMEOUT_SECS, RMS_HANGOVER)
        ) or elapsed > MAX_UTTERANCE_SECS:
            loop.call_soon_threadsafe(finished.set)

    device_index = sd.default.device[0]
    if device_index is None:
        device_index = 0

    start_time = time.time()

    with sd.InputStream(
        samplerate=TARGET_SR,
        blocksize=int(TARGET_SR * FRAME_DURATION_MS / 1000),
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
    ):
        while not finished.is_set():
            try:
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                if time.time() - start_time > MAX_UTTERANCE_SECS:
                    break
                continue

            b16 = pcm16_from_float32(chunk)
            payload = {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(b16).decode("ascii"),
            }
            await ws.send(json.dumps(payload))
            samples_streamed += len(chunk)

        while not audio_queue.empty():
            chunk = await audio_queue.get()
            b16 = pcm16_from_float32(chunk)
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(b16).decode("ascii"),
                    }
                )
            )
            samples_streamed += len(chunk)

    if samples_streamed == 0 or not getattr(audio_callback, "started", False):
        print("[info] No speech detected.")
        await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
        return None

    duration = samples_streamed / TARGET_SR
    if duration < MIN_SPEECH_SECS:
        print("[info] Speech shorter than MIN_SPEECH_SECS; skipping.")
        await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
        return None

    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
    await ws.send(
        json.dumps(
            {
                "type": "response.create",
                "response": {
                    "modalities": ["text"],
                    "instructions": ASSISTANT_INSTRUCTIONS,
                },
            }
        )
    )

    return await receive_realtime(ws)


async def realtime_loop() -> None:
    import websockets

    async with websockets.connect(PROXY_URL) as ws:
        print("Connected to realtime proxy. Press Ctrl+C to stop.")

        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "input_audio_format": {
                            "sample_rate_hz": TARGET_SR,
                            "channels": CHANNELS,
                        },
                        "modalities": ["text"],
                        "instructions": ASSISTANT_INSTRUCTIONS,
                    },
                }
            )
        )

        while True:
            print("\n🎙️ Speak now…")
            result = await stream_utterance(ws)
            if not result:
                continue

            print(f"📝 You: {result.transcript}")
            print(f"🤖 Bot: {result.reply}")
            elevenlabs_tts(result.reply)


def main() -> None:
    try:
        sd.check_output_settings(samplerate=TARGET_SR, channels=1)
    except Exception:
        pass

    asyncio.run(realtime_loop())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Bye!")
