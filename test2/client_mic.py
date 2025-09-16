# client_mic.py
# Mic -> send WAV to GPU server -> get transcript+reply -> speak reply via ElevenLabs (local playback).
# Works on Windows/macOS/Linux (run outside WSL if you want audio output on Windows).

import io, os, time, wave, requests
import numpy as np
import sounddevice as sd
from scipy.signal import resample

# =========================
# Config (env-first)
# =========================
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8765/transcribe_and_reply")

# ElevenLabs

ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "vFQACl5nAIV0owAavYxE")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

# VAD / audio
TARGET_SR = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30
SILENCE_TIMEOUT_SECS = 1.7
MIN_SPEECH_SECS = 0.4
RMS_THRESH = float(os.getenv("RMS_THRESH", "0.003"))
RMS_HANGOVER = 0.25
MAX_UTTERANCE_SECS = 45

# =========================
# Helpers
# =========================
def resample_to_16k(audio_np: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return audio_np
    target_len = int(len(audio_np) * TARGET_SR / src_sr)
    if target_len <= 0:
        return np.zeros(1, dtype=np.float32)
    return resample(audio_np, target_len).astype(np.float32)

def float32_to_wav_bytes(audio_np: np.ndarray, sr: int) -> io.BytesIO:
    audio_16k = resample_to_16k(audio_np, sr)
    int16 = np.clip(audio_16k * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(int16.tobytes())
    buf.seek(0)
    return buf

def pick_input_device():
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d.get('max_input_channels', 0) > 0:
            return i
    raise RuntimeError("No input audio devices.")

# =========================
# Capture one utterance (RMS VAD)
# =========================
def record_utterance():
    device_index = pick_input_device()
    dev_info = sd.query_devices(device_index)
    device_sr = int(dev_info.get("default_samplerate") or TARGET_SR)

    print("\n🎙️ Speak (short pause ends)…")
    chunks = []
    start_time = time.time()
    last_above = None
    frame_samples = max(256, int(device_sr * FRAME_DURATION_MS / 1000))
    started = False

    def callback(indata, frames, time_info, status):
        nonlocal last_above, started
        if status:  # over/under-runs etc.
            print(status)
        mono = indata.copy().reshape(-1)
        rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
        chunks.append(mono.tobytes())
        if rms > RMS_THRESH:
            last_above = time.time()
            started = True

    try:
        with sd.InputStream(device=device_index, channels=CHANNELS, samplerate=device_sr,
                            dtype='float32', blocksize=frame_samples, callback=callback):
            while True:
                time.sleep(0.04)
                now = time.time()
                if now - start_time > MAX_UTTERANCE_SECS:
                    break
                if started and last_above and (now - last_above) > max(SILENCE_TIMEOUT_SECS, RMS_HANGOVER):
                    break
    except Exception as e:
        print("Stream error:", e)
        return io.BytesIO(), 0

    if not chunks:
        print("⏹️ No audio.")
        return io.BytesIO(), 0

    audio_float = b"".join(chunks)
    audio_np = np.frombuffer(audio_float, dtype=np.float32)
    dur = len(audio_np) / float(device_sr)
    if dur < MIN_SPEECH_SECS:
        print("⏹️ Too short.")
        return io.BytesIO(), 0

    print("⏹️ End.")
    return float32_to_wav_bytes(audio_np, device_sr), dur

# =========================
# ElevenLabs TTS (local playback)
# =========================
def speak_elevenlabs(text: str, lang_hint: str | None):
    """
    Generate 16 kHz PCM with ElevenLabs and play via sounddevice.
    Falls back to pyttsx3 if ElevenLabs fails or API key missing.
    """
    if not ELEVENLABS_API_KEY:
        return speak_offline_tts(text, lang_hint)

    try:
        from elevenlabs import ElevenLabs
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        # Streaming generator of raw 16k PCM bytes
        audio_gen = client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            optimize_streaming_latency="0",
            output_format="pcm_16000",
            text=text,
            model_id=ELEVENLABS_MODEL_ID,
        )
        pcm_bytes = b"".join(audio_gen)
        if not pcm_bytes:
            raise RuntimeError("Empty audio from ElevenLabs")

        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(pcm.reshape(-1, 1), TARGET_SR)
        sd.wait()
    except Exception as e:
        print("ElevenLabs TTS error:", e)
        speak_offline_tts(text, lang_hint)

def speak_offline_tts(text: str, lang_hint: str | None):
    """Basic offline TTS (pyttsx3) as fallback."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        try:
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 1.0)
        except Exception:
            pass
        # Try to select HR/EN voice if available
        try:
            voices = engine.getProperty('voices') or []
            want_hr = (lang_hint or "").startswith("hr")
            hr_hints = ["hr", "croat", "hrv", "hr-hr", "croatian"]
            en_hints = ["en", "eng", "en-us", "english"]

            def choose(hints):
                for v in voices:
                    blob = f"{getattr(v,'name','')} {getattr(v,'id','')} {getattr(v,'languages',[])}".lower()
                    if any(h in blob for h in hints):
                        return v.id
                return None
            vid = choose(hr_hints) if want_hr else choose(en_hints)
            if not vid and want_hr:
                vid = choose(en_hints)
            if vid:
                engine.setProperty('voice', vid)
        except Exception:
            pass

        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Offline TTS error:", e)

# =========================
# Main loop
# =========================
def main():
    summary = ""  # (optional) možeš držati rolling summary lokalno i slati serveru
    while True:
        wav_buf, dur = record_utterance()
        if wav_buf.getbuffer().nbytes < 32000:
            continue

        files = {"audio": ("in.wav", wav_buf, "audio/wav")}
        data = {"summary": summary}
        t0 = time.time()
        r = requests.post(SERVER_URL, files=files, data=data, timeout=120)
        dt = time.time() - t0
        r.raise_for_status()
        out = r.json()

        transcript = out.get("transcript", "")
        reply = out.get("reply", "")
        lang = out.get("language", None)
        lat = out.get("latency", {}) or {}

        print(f"📝 You: {transcript}")
        print(f"🤖 Bot: {reply}")
        print(
            "⏱ Latency:"
            f" whisper={lat.get('whisper_sec','?')}s"
            f"  llm={lat.get('llm_sec','?')}s"
            f"  total={lat.get('total_sec','?')}s"
            f"  +network={dt - float(lat.get('total_sec', 0) or 0):.2f}s"
        )

        # Speak reply
        if reply:
            speak_elevenlabs(reply, lang)

if __name__ == "__main__":
    try:
        sd.check_output_settings(samplerate=TARGET_SR, channels=1)
    except Exception:
        # Some drivers need an initial dummy open to init backend
        try:
            sd.play(np.zeros((TARGET_SR//10, 1), dtype=np.float32), TARGET_SR)
            sd.wait()
        except Exception:
            pass
    main()
