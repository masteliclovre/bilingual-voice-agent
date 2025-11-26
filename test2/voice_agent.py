"""
# Bilingual voice agent - Remote-only version
# - Audio capture with VAD
# - Audio feedback (beep)
# - Remote agent communication
# - Enhanced latency diagnostics
# - Audio playback
"""

# =========================
# Imports
# =========================

import os
import io
import sys
import time
import wave
import base64
import contextlib
import threading
import multiprocessing
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:
    print("ERROR: sounddevice not installed. Run: pip install sounddevice", file=sys.stderr)
    raise SystemExit(1) from exc

from dotenv import load_dotenv
from scipy.signal import resample
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()


# =========================
# Configuration
# =========================

# Remote Agent
REMOTE_AGENT_URL = os.getenv("REMOTE_AGENT_URL", "").strip() or None
REMOTE_AGENT_TOKEN = os.getenv("REMOTE_AGENT_TOKEN", "").strip() or None
REMOTE_AGENT_OPENAI_KEY = os.getenv("REMOTE_AGENT_OPENAI_KEY", "").strip() or None

# Audio Settings
TARGET_SR = 16000
FRAME_DURATION_MS = int(os.getenv("FRAME_DURATION_MS", "15"))
MAX_UTTERANCE_SECS = 45
SILENCE_TIMEOUT_SECS = float(os.getenv("SILENCE_TIMEOUT_SECS", "0.2"))
MIN_SPEECH_SECS = float(os.getenv("MIN_SPEECH_SECS", "0.3"))
RMS_THRESH = float(os.getenv("RMS_THRESH", "0.003"))
RMS_HANGOVER = float(os.getenv("RMS_HANGOVER", "0.12"))

# Audio Feedback
BEEP_DELAY_MS = int(os.getenv("BEEP_DELAY_MS", "2000"))

# Device Selection
PREFERRED_INPUT_NAME = os.getenv("PREFERRED_INPUT_NAME", "").strip() or None
INPUT_DEVICE_INDEX = os.getenv("INPUT_DEVICE_INDEX", "").strip()
INPUT_DEVICE_INDEX = int(INPUT_DEVICE_INDEX) if INPUT_DEVICE_INDEX.isdigit() else None

# HTTP Timeouts
HTTP_CONNECT_TIMEOUT = float(os.getenv("HTTP_CONNECT_TIMEOUT", "4.0"))
HTTP_READ_TIMEOUT = float(os.getenv("HTTP_READ_TIMEOUT", "60.0"))
HTTP_TIMEOUT = (HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT)


# =========================
# HTTP Session
# =========================

def _configure_http_session(session: requests.Session) -> requests.Session:
    """Configure HTTP session with retries and connection pooling."""
    adapter = HTTPAdapter(
        pool_connections=8,
        pool_maxsize=16,
        max_retries=Retry(
            total=2,
            backoff_factor=0.1,
            status_forcelist=[429, 502, 503, 504]
        ),
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


_http_session_lock = __import__('threading').Lock()
_shared_http_session: Optional[requests.Session] = None


def _get_shared_http_session() -> requests.Session:
    """Get or create a shared HTTP session with thread safety."""
    global _shared_http_session
    with _http_session_lock:
        if _shared_http_session is None:
            _shared_http_session = _configure_http_session(requests.Session())
    return _shared_http_session


# =========================
# Audio Feedback
# =========================

def generate_rising_beep(start_freq=200, end_freq=300, duration_ms=600):
    """Generate a rising frequency beep (swoosh up)."""
    duration_sec = duration_ms / 1000.0
    samples = int(TARGET_SR * duration_sec)
    t = np.linspace(0, duration_sec, samples, False)
    
    # Linear frequency sweep
    freq = np.linspace(start_freq, end_freq, samples)
    phase = 2 * np.pi * np.cumsum(freq) / TARGET_SR
    tone = np.sin(phase).astype(np.float32)
    
    # Envelope with fade in/out
    envelope = np.ones_like(tone)
    fade_samples = int(TARGET_SR * 0.01)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    
    return tone * envelope * 0.3


# Pre-generate beep sound
BEEP_SOUND = generate_rising_beep()


# =========================
# Audio Output
# =========================

class OutputAudio:
    """Persistent audio output stream."""
    
    def __init__(self, samplerate=TARGET_SR, channels=1):
        stream_kwargs = dict(
            samplerate=samplerate,
            channels=channels,
            dtype='float32',
        )
        try:
            self.stream = sd.OutputStream(latency='low', **stream_kwargs)
        except Exception:
            self.stream = sd.OutputStream(**stream_kwargs)
        self.stream.start()

    def write_int16_bytes(self, pcm_bytes: bytes):
        """Write PCM16 audio bytes to output stream."""
        if not pcm_bytes:
            return
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.stream.write(pcm.reshape(-1, 1))

    def write_float_np(self, pcm: np.ndarray):
        """Write float32 numpy audio to output stream."""
        if pcm.size == 0:
            return
        self.stream.write(pcm.reshape(-1, 1))

    def close(self):
        """Close the output stream."""
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


# =========================
# Latency Tracker
# =========================

class LatencyTracker:
    """Track and report latency for different operations."""
    
    def __init__(self):
        self.events: list[tuple[str, float]] = []
        self.turn_start = time.perf_counter()

    @contextlib.contextmanager
    def track(self, label: str):
        """Context manager to track timing of an operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self.events.append((label, max(0.0, end - start)))

    def clear(self):
        """Clear all tracked events and reset timer."""
        self.events.clear()
        self.turn_start = time.perf_counter()

    def report(self, title: str = "⏱️ Latency breakdown"):
        """Print a detailed latency report."""
        if not self.events:
            return
        
        total_turn = time.perf_counter() - self.turn_start
        sum_tracked = sum(d for _, d in self.events)
        untracked = max(0.0, total_turn - sum_tracked)
        
        print()
        print("=" * 60)
        print(title)
        print("=" * 60)
        
        for label, duration in self.events:
            pct_of_total = (duration / total_turn * 100.0) if total_turn > 0 else 0.0
            bar_width = int(pct_of_total / 2)
            bar = "█" * bar_width
            print(f"  {label:<25} {duration*1000:7.1f}ms  {pct_of_total:5.1f}%  {bar}")
        
        if untracked > 0.01:
            pct_untracked = (untracked / total_turn * 100.0)
            bar_width = int(pct_untracked / 2)
            bar = "░" * bar_width
            print(f"  {'[untracked overhead]':<25} {untracked*1000:7.1f}ms  {pct_untracked:5.1f}%  {bar}")
        
        print("-" * 60)
        print(f"  {'TOTAL TURN TIME':<25} {total_turn*1000:7.1f}ms  100.0%")
        print("=" * 60)
        print()


# =========================
# Audio Utilities
# =========================

def resample_to_16k(audio_np: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample audio to 16kHz."""
    if src_sr == TARGET_SR:
        return audio_np
    target_len = int(len(audio_np) * TARGET_SR / src_sr)
    if target_len <= 0:
        return np.zeros(1, dtype=np.float32)
    return resample(audio_np, target_len).astype(np.float32)


def float32_to_wav_bytes(audio_np: np.ndarray, sr: int) -> io.BytesIO:
    """Convert float32 audio to WAV format bytes."""
    audio_16k = resample_to_16k(audio_np, sr)
    int16 = np.clip(audio_16k * 32767, -32768, 32767).astype(np.int16)
    
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(int16.tobytes())
    buf.seek(0)
    return buf


def select_input_device():
    """Select the best available input device."""
    devices = sd.query_devices()
    
    if INPUT_DEVICE_INDEX is not None:
        dev = devices[INPUT_DEVICE_INDEX]
        print(f"Using device {INPUT_DEVICE_INDEX}: {dev['name']}")
        return INPUT_DEVICE_INDEX, int(dev['default_samplerate'])
    
    if PREFERRED_INPUT_NAME:
        for idx, dev in enumerate(devices):
            if PREFERRED_INPUT_NAME.lower() in dev['name'].lower():
                print(f"Using device {idx}: {dev['name']}")
                return idx, int(dev['default_samplerate'])
    
    default = sd.default.device[0]
    if default is None:
        default = 0
    dev = devices[default]
    print(f"Using default device {default}: {dev['name']}")
    return default, int(dev['default_samplerate'])


def play_audio_feedback_process(delay_ms: int, stop_flag):
    """
    Play audio feedback in a separate process.
    
    This runs in its own process to avoid Python's GIL blocking audio playback
    while the main process waits for remote agent response.
    """
    time.sleep(delay_ms / 1000.0)
    
    if stop_flag.value:
        return
    
    try:
        stream_kwargs = dict(
            samplerate=TARGET_SR,
            channels=1,
            dtype='float32',
        )
        try:
            stream = sd.OutputStream(latency='low', **stream_kwargs)
        except Exception:
            stream = sd.OutputStream(**stream_kwargs)
        
        stream.start()
        stream.write(BEEP_SOUND.reshape(-1, 1))
        stream.stop()
        stream.close()
    except Exception:
        pass


# =========================
# Voice Activity Detection
# =========================

class ContinuousListener:
    """Continuous audio listener with VAD."""
    
    def __init__(self):
        self.device_idx, self.device_sr = select_input_device()
        self.frame_size = int(self.device_sr * FRAME_DURATION_MS / 1000.0)

    def record_utterance(self) -> io.BytesIO:
        """Record a single utterance using simple RMS-based VAD."""
        print("🎙️  Listening...")
        
        stream_kwargs = dict(
            device=self.device_idx,
            samplerate=self.device_sr,
            channels=1,
            dtype='float32',
            blocksize=self.frame_size,
        )
        
        try:
            stream = sd.InputStream(latency='low', **stream_kwargs)
        except Exception:
            stream = sd.InputStream(**stream_kwargs)
        
        stream.start()
        
        chunks = []
        triggered = False
        silent_frames = 0
        hangover_frames = int(RMS_HANGOVER * 1000.0 / FRAME_DURATION_MS)
        silence_limit = int(SILENCE_TIMEOUT_SECS * 1000.0 / FRAME_DURATION_MS)
        max_frames = int(MAX_UTTERANCE_SECS * 1000.0 / FRAME_DURATION_MS)
        
        try:
            for _ in range(max_frames):
                frame, overflow = stream.read(self.frame_size)
                if overflow:
                    print("⚠️  Audio overflow")
                
                rms = np.sqrt(np.mean(frame ** 2))
                chunks.append(frame.copy())
                
                if rms > RMS_THRESH:
                    if not triggered:
                        print("🗣️  Speech detected...")
                        triggered = True
                    silent_frames = 0
                else:
                    if triggered:
                        silent_frames += 1
                        if silent_frames > (silence_limit + hangover_frames):
                            break
            else:
                print(f"⏹️  Max utterance length reached ({MAX_UTTERANCE_SECS}s)")
        
        finally:
            stream.stop()
            stream.close()
        
        if not triggered:
            print("⏹️  No speech detected.")
            return io.BytesIO()
        
        audio_np = np.concatenate(chunks, axis=0).flatten()
        duration_sec = len(audio_np) / float(self.device_sr)
        
        if duration_sec < MIN_SPEECH_SECS:
            print("⏹️  End (too short).")
            return io.BytesIO()
        
        print("⏹️  End of utterance.")
        return float32_to_wav_bytes(audio_np, self.device_sr)


# =========================
# Remote Agent Client
# =========================

@dataclass
class RemoteAgentResult:
    """Result from remote agent processing."""
    user_text: str
    assistant_text: str
    lang: Optional[str]
    audio_pcm16: bytes
    sample_rate: int
    session_id: Optional[str] = None
    skipped: bool = False
    reason: Optional[str] = None


class RemoteAgentClient:
    """Client for communicating with remote voice agent server."""
    
    def __init__(self, base_url: str, token: Optional[str] = None, openai_api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/api"):
            self.base_url = f"{self.base_url}/api"
        self.token = token.strip() if token else None
        self.openai_api_key = openai_api_key.strip() if openai_api_key else None
        self.session_id: Optional[str] = None
        self.session = _configure_http_session(requests.Session())

    def _headers(self):
        """Build request headers."""
        headers = {}
        if self.token:
            headers["X-Auth"] = self.token
        if self.openai_api_key:
            headers["X-OpenAI-Key"] = self.openai_api_key
        return headers

    def ensure_session(self):
        """Ensure we have a valid session ID."""
        if self.session_id:
            return
        
        resp = self.session.post(
            f"{self.base_url}/session",
            headers=self._headers(),
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        sid = data.get("session_id")
        
        if not sid:
            raise RuntimeError("Remote agent did not return a session_id")
        
        self.session_id = sid

    def process(self, wav_buf: io.BytesIO) -> Optional[RemoteAgentResult]:
        """Send audio to remote agent and get response."""
        payload = wav_buf.getvalue()
        if not payload:
            return None
        
        attempts = 0
        while attempts < 2:
            attempts += 1
            self.ensure_session()
            
            files = {"audio": ("audio.wav", payload, "audio/wav")}
            data = {"session_id": self.session_id}
            if self.openai_api_key:
                data["openai_api_key"] = self.openai_api_key
            
            resp = self.session.post(
                f"{self.base_url}/process",
                headers=self._headers(),
                files=files,
                data=data,
                timeout=HTTP_TIMEOUT,
            )
            
            if resp.status_code in (401, 403):
                raise RuntimeError("Remote agent authentication failed")
            
            if resp.status_code in (404, 410):
                self.session_id = None
                continue
            
            resp.raise_for_status()
            body = resp.json()
            
            if body.get("error"):
                raise RuntimeError(body["error"])
            
            sid = body.get("session_id")
            if sid:
                self.session_id = sid
            
            user_text = body.get("text", "") or ""
            assistant_text = body.get("assistant_text", "") or ""
            lang = body.get("lang") or None
            audio_b64 = body.get("tts_audio_b64")
            audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""
            sr = int(body.get("tts_sample_rate", TARGET_SR) or TARGET_SR)
            skipped = bool(body.get("skipped"))
            
            return RemoteAgentResult(
                user_text=user_text,
                assistant_text=assistant_text,
                lang=lang,
                audio_pcm16=audio_bytes,
                sample_rate=sr,
                session_id=self.session_id,
                skipped=skipped,
                reason=body.get("reason"),
            )
        
        raise RuntimeError("Remote agent unavailable after retries")

    def process_stream(self, wav_buf: io.BytesIO, out: OutputAudio) -> Optional[RemoteAgentResult]:
        """Send audio to remote agent and stream TTS response back."""
        payload = wav_buf.getvalue()
        if not payload:
            return None

        attempts = 0
        while attempts < 2:
            attempts += 1
            self.ensure_session()

            files = {"audio": ("audio.wav", payload, "audio/wav")}
            data = {"session_id": self.session_id}
            if self.openai_api_key:
                data["openai_api_key"] = self.openai_api_key

            resp = self.session.post(
                f"{self.base_url}/process_stream",
                headers=self._headers(),
                files=files,
                data=data,
                timeout=HTTP_TIMEOUT,
                stream=True,  # Enable streaming
            )

            if resp.status_code in (401, 403):
                raise RuntimeError("Remote agent authentication failed")

            if resp.status_code in (404, 410):
                self.session_id = None
                continue

            resp.raise_for_status()

            # Extract metadata from headers
            session_id = resp.headers.get("X-Session-ID")
            user_text = resp.headers.get("X-User-Text", "")
            assistant_text = resp.headers.get("X-Assistant-Text", "")
            lang = resp.headers.get("X-Lang") or None
            rag_used = resp.headers.get("X-RAG-Used", "false") == "true"

            if session_id:
                self.session_id = session_id

            # Stream audio chunks and play immediately
            audio_chunks = []
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    # Play immediately for low latency
                    out.write_int16_bytes(chunk)
                    # Also collect for return value
                    audio_chunks.append(chunk)

            # Combine all chunks
            audio_bytes = b"".join(audio_chunks)

            return RemoteAgentResult(
                user_text=user_text,
                assistant_text=assistant_text,
                lang=lang,
                audio_pcm16=audio_bytes,
                sample_rate=TARGET_SR,
                session_id=self.session_id,
                skipped=False,
                reason=None,
            )

        raise RuntimeError("Remote agent unavailable after retries")


# =========================
# Main Loop
# =========================

def main():
    """Main application loop."""
    
    # Validate configuration
    if not REMOTE_AGENT_URL:
        print("\n" + "=" * 60)
        print("❌ ERROR: REMOTE_AGENT_URL is not set!")
        print("=" * 60)
        print("\nThis minimal version requires a remote agent server.")
        print("Please set REMOTE_AGENT_URL in your .env file.")
        print("\nExample:")
        print("  REMOTE_AGENT_URL=https://your-runpod-url.proxy.runpod.net/")
        print("=" * 60)
        sys.exit(1)

    print(f"\n✅ Using remote voice agent backend at: {REMOTE_AGENT_URL}")
    
    remote_client = RemoteAgentClient(
        REMOTE_AGENT_URL,
        REMOTE_AGENT_TOKEN,
        openai_api_key=REMOTE_AGENT_OPENAI_KEY,
    )

    out = OutputAudio(samplerate=TARGET_SR, channels=1)
    listener = ContinuousListener()

    print("\n🎙️ Bilingual voice agent ready. (HR/EN)")
    print("Tips:")
    print("- Speak naturally; a short pause ends your turn.")
    
    try:
        while True:
            try:
                tracker = LatencyTracker()
                
                # Capture audio
                with tracker.track("capture"):
                    wav_buf = listener.record_utterance()
                
                if wav_buf.getbuffer().nbytes < 32000:
                    tracker.report("⏱️ Turn skipped (no usable audio)")
                    continue

                # Create stop flag for audio feedback process
                stop_flag = multiprocessing.Value('i', 0)
                result = None
                exception = None
                
                # Start audio feedback in separate PROCESS (not thread)
                feedback_process = multiprocessing.Process(
                    target=play_audio_feedback_process,
                    args=(BEEP_DELAY_MS, stop_flag),
                    daemon=True
                )
                feedback_process.start()
                
                # Send to remote agent and stream response (happens in parallel, no GIL interference)
                try:
                    with tracker.track("remote_agent_and_playback"):
                        result = remote_client.process_stream(wav_buf, out)
                except Exception as e:
                    exception = e
                finally:
                    # Signal audio feedback to stop
                    stop_flag.value = 1
                    feedback_process.join(timeout=0.5)
                    if feedback_process.is_alive():
                        feedback_process.terminate()

                # Handle any exception from remote agent
                if exception:
                    print(f"❌ Remote agent error: {exception}")
                    tracker.report()
                    continue

                if not result or not result.user_text:
                    tracker.report("⏱️ Turn skipped (remote empty)")
                    continue

                # Display results
                flag = "🇭🇷" if (result.lang or "").startswith("hr") else "🇬🇧"
                print(f"{flag} You: {result.user_text}")
                print(f"🤖 Assistant: {result.assistant_text}")

                # Audio already played during streaming - no need to play again
                if not result.audio_pcm16:
                    print("⚠️ (Remote agent returned no audio)")
                
                tracker.report()

            except KeyboardInterrupt:
                print("\n👋 Exit. Bye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                tracker.report("⏱️ Turn errored")
                time.sleep(0.2)
    finally:
        out.close()


if __name__ == "__main__":
    main()