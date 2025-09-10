# voice_agent_continuous.py
# Continuous bilingual (HR/EN) voice agent:
# - Always-listening RMS VAD turn-taking (no click)
# - Local transcription (faster-whisper)
# - OpenAI for reasoning
# - ElevenLabs TTS (pcm_16000)
# - Conversation memory: rolling history + auto summary compression
# Version: 2025-09-10

import os
import io
import sys
import time
import wave
import threading
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import OpenAI
from elevenlabs import ElevenLabs
from scipy.signal import resample

# =========================
# Config
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "vFQACl5nAIV0owAavYxE")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "GoranS/whisper-large-v3-turbo-hr-parla-ctranslate2")

PREFERRED_INPUT_NAME = os.getenv("PREFERRED_INPUT_NAME", "").strip() or None
INPUT_DEVICE_INDEX = os.getenv("INPUT_DEVICE_INDEX", "").strip()
INPUT_DEVICE_INDEX = int(INPUT_DEVICE_INDEX) if INPUT_DEVICE_INDEX.isdigit() else None

# Audio
TARGET_SR = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30
MAX_UTTERANCE_SECS = 45
SILENCE_TIMEOUT_SECS = 1.7     # end-of-utterance gap
MIN_SPEECH_SECS = 0.4          # ignore ultra-short blips
RMS_THRESH = float(os.getenv("RMS_THRESH", "0.003"))  # lower if your mic is quiet (e.g. 0.002)
RMS_HANGOVER = 0.25            # how long to keep "speaking" after last above-threshold frame

# OpenAI
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

# Memory
MAX_TURNS_IN_WINDOW = 12       # keep last N user/assistant turns verbatim
SUMMARY_UPDATE_EVERY = 4       # after this many user turns, compress into summary

# Optional wake word (say e.g. "Hej asistente" / "Hey assistant")
WAKE_WORD = os.getenv("WAKE_WORD", "").strip() or None   # set to e.g. "hej asistent" to require wake word

# =========================
# Utilities
# =========================

def list_audio_devices():
    print("Audio devices:")
    for i, d in enumerate(sd.query_devices()):
        name = d.get('name', '?')
        in_ch = d.get('max_input_channels', 0)
        out_ch = d.get('max_output_channels', 0)
        sr = d.get('default_samplerate', None)
        print(f"[{i:02d}] {name}  in:{in_ch} out:{out_ch} sr:{sr}")

def pick_input_device(prefer_name_substr=None, prefer_index=None):
    devices = sd.query_devices()
    # prefer index
    if isinstance(prefer_index, int) and 0 <= prefer_index < len(devices):
        if devices[prefer_index]['max_input_channels'] > 0:
            return prefer_index
    # prefer name
    if prefer_name_substr:
        p = prefer_name_substr.lower()
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0 and p in d.get('name', '').lower():
                return i
    # first input-capable
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            return i
    raise RuntimeError("No input audio devices with capture channels.")

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

# =========================
# Engines (Whisper, OpenAI, ElevenLabs)
# =========================

def load_whisper():
    print("Loading Whisper model (first run may take a bit)…")
    return WhisperModel(
        WHISPER_MODEL,
        device="auto",
        compute_type="auto",
    )

def init_openai():
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    return OpenAI(api_key=OPENAI_API_KEY)

def init_elevenlabs():
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("Missing ELEVENLABS_API_KEY in .env")
    return ElevenLabs(api_key=ELEVENLABS_API_KEY)

def whisper_transcribe(whisper: WhisperModel, wav_buf: io.BytesIO):
    if wav_buf.getbuffer().nbytes < 32000:
        return "", None
    tmp = "temp_in.wav"
    with open(tmp, "wb") as f:
        f.write(wav_buf.read())
    wav_buf.seek(0)
    segments, info = whisper.transcribe(
        tmp,
        beam_size=3,
        vad_filter=False,
        temperature=0.0,
        language=None,
        without_timestamps=True,
    )
    text = "".join(seg.text for seg in segments).strip()
    lang = getattr(info, "language", None)
    return text, lang

def openai_complete(oa: OpenAI, messages, temperature=OPENAI_TEMPERATURE):
    resp = oa.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()

def tts_elevenlabs_and_play(el: ElevenLabs, text: str):
    # Use raw PCM at 16k (no WAV header)
    audio_gen = el.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        optimize_streaming_latency="0",
        output_format="pcm_16000",
        text=text,
        model_id="eleven_multilingual_v2",
    )
    pcm_bytes = b"".join(audio_gen)
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    pcm = pcm.reshape(-1, 1)
    sd.play(pcm, TARGET_SR)
    sd.wait()

# =========================
# Conversation Memory
# =========================

class Memory:
    """
    Keeps rolling verbatim turns + an accumulated summary.
    We summarize every N user turns to keep context compact.
    """
    def __init__(self, oa: OpenAI):
        self.oa = oa
        self.summary = ""           # long-term compressed memory
        self.window = []            # recent messages: [{'role':'user'|'assistant','content':...}, ...]
        self.user_turns_since_summary = 0

    def add_user(self, content: str):
        self.window.append({"role": "user", "content": content})
        self.user_turns_since_summary += 1
        # keep size in check
        self._trim_window()

    def add_assistant(self, content: str):
        self.window.append({"role": "assistant", "content": content})
        self._trim_window()

    def _trim_window(self):
        # keep only last N turns
        roles = [m["role"] for m in self.window]
        # Each user/assistant is a "turn" here, so keep 2*N messages-ish:
        # We'll simply cap messages length
        if len(self.window) > MAX_TURNS_IN_WINDOW * 2:
            self.window = self.window[-MAX_TURNS_IN_WINDOW*2:]

    def maybe_summarize(self):
        if self.user_turns_since_summary < SUMMARY_UPDATE_EVERY:
            return
        # Build a brief summary with salient facts, preferences, open tasks
        sys_prompt = (
            "You are a memory compressor. Summarize the following conversation "
            "into concise bullet points capturing user preferences, facts, goals, and unresolved tasks. "
            "Keep neutral tone. Max ~150 words."
        )
        msgs = [{"role": "system", "content": sys_prompt}]
        if self.summary:
            msgs.append({"role": "system", "content": f"Existing summary memory:\n{self.summary}"})
        # add the recent window
        for m in self.window:
            msgs.append(m)
        try:
            new_summary = openai_complete(self.oa, msgs, temperature=0.2)
            # merge
            self.summary = new_summary
            # reset counter
            self.user_turns_since_summary = 0
        except Exception as e:
            print("Memory summarize error:", e)

    def build_prompt(self, user_lang_hint: str | None):
        system = (
            "You are a concise, helpful bilingual assistant for Croatian and English. "
            "Detect the user's language (Croatian or English) and ALWAYS reply in that language. "
            "If the user mixes languages, keep their dominant language. "
            "Keep answers short for voice (2–5 sentences). "
            "Be context-aware and remember prior details from the summary."
        )
        if (user_lang_hint or "").startswith("hr"):
            system += " Prefer Croatian if the user speaks Croatian."
        else:
            system += " Prefer English if the user speaks English."

        msgs = [{"role": "system", "content": system}]
        if self.summary:
            msgs.append({"role": "system", "content": f"Conversation summary memory:\n{self.summary}"})
        # include recent window
        msgs.extend(self.window)
        return msgs

# =========================
# Turn-taking capture (RMS VAD)
# =========================

class ContinuousListener:
    """
    Opens and closes the input stream for each utterance to avoid TTS echo.
    Capture ends on silence timeout and minimum speech checks.
    """

    def __init__(self):
        self.device_index = pick_input_device(PREFERRED_INPUT_NAME, INPUT_DEVICE_INDEX)
        self.dev_info = sd.query_devices(self.device_index)
        self.device_sr = int(self.dev_info.get("default_samplerate") or TARGET_SR)

    def record_utterance(self):
        """Capture one utterance based on RMS VAD with hangover and silence timeout."""
        print("\n🎙️ Listening… (speak; short pause = end)")
        chunks = []
        start_time = time.time()
        last_above = None
        frame_samples = max(256, int(self.device_sr * FRAME_DURATION_MS / 1000))
        started_speaking = False

        def callback(indata, frames, time_info, status):
            nonlocal last_above, started_speaking
            if status:
                # Non-fatal info/overflow/underflow
                print(status, file=sys.stderr)
            mono = indata.copy().reshape(-1)
            rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
            chunks.append(mono.tobytes())
            if rms > RMS_THRESH:
                last_above = time.time()
                started_speaking = True

        try:
            with sd.InputStream(device=self.device_index,
                                channels=CHANNELS,
                                samplerate=self.device_sr,
                                dtype='float32',
                                blocksize=frame_samples,
                                callback=callback):
                while True:
                    time.sleep(0.04)
                    now = time.time()
                    # too long safeguard
                    if now - start_time > MAX_UTTERANCE_SECS:
                        break
                    # if we have begun speaking…
                    if started_speaking:
                        # ensure min speech before allowing to end
                        if last_above is not None and (now - last_above) > max(SILENCE_TIMEOUT_SECS, RMS_HANGOVER):
                            # silence tail exceeded
                            break
                    else:
                        # haven't started; keep waiting
                        pass
        except Exception as e:
            print("Stream error:", e)
            return io.BytesIO()

        if not chunks:
            print("⏹️ End (no audio).")
            return io.BytesIO()

        audio_float = b"".join(chunks)
        audio_np = np.frombuffer(audio_float, dtype=np.float32)

        # Reject ultra-short utterances
        duration_sec = len(audio_np) / float(self.device_sr)
        if duration_sec < MIN_SPEECH_SECS:
            print("⏹️ End (too short).")
            return io.BytesIO()

        print("⏹️ End of utterance.")
        return float32_to_wav_bytes(audio_np, self.device_sr)

# =========================
# Main loop
# =========================

def main():
    list_audio_devices()

    oa = init_openai()
    el = init_elevenlabs()
    whisper = load_whisper()
    mem = Memory(oa)
    listener = ContinuousListener()

    print("\nBilingual voice agent ready. (HR/EN)")
    print("Tips:")
    print("- Speak naturally; a short pause ends your turn.")
    print("- Set RMS_THRESH lower in .env if it misses quiet speech (e.g. 0.002).")
    if WAKE_WORD:
        print(f"- Wake word enabled: say “{WAKE_WORD}” to start a turn.")

    while True:
        try:
            # 1) Capture one utterance
            wav_buf = listener.record_utterance()
            if wav_buf.getbuffer().nbytes < 32000:
                continue

            # 2) Transcribe
            user_text, lang = whisper_transcribe(whisper, wav_buf)
            if not user_text:
                continue

            # 2a) Wake word (optional)
            if WAKE_WORD:
                if user_text.lower().strip().startswith(WAKE_WORD.lower()):
                    # strip wake word
                    user_text = user_text[len(WAKE_WORD):].lstrip(" ,.-:") or "Hej!"
                else:
                    # ignore non-wake speech
                    print(f"(Ignored — missing wake word '{WAKE_WORD}')")
                    continue

            flag = "🇭🇷" if (lang or "").startswith("hr") else "🇬🇧"
            print(f"{flag} You: {user_text}")

            # 3) Build contextful prompt with memory
            mem.add_user(user_text)
            messages = mem.build_prompt(user_lang_hint=lang)

            # 4) LLM reply
            assistant_text = openai_complete(oa, messages)
            print(f"🤖 Assistant: {assistant_text}")

            # 5) Add to memory and maybe summarize
            mem.add_assistant(assistant_text)
            mem.maybe_summarize()

            # 6) Speak reply (mute mic by closing input stream — we open per utterance)
            tts_elevenlabs_and_play(el, assistant_text)

            # loop continues automatically (always-listening)

        except KeyboardInterrupt:
            print("\nExit. Bye!")
            break
        except Exception as e:
            print("Error:", e)
            time.sleep(0.4)

if __name__ == "__main__":
    main()
