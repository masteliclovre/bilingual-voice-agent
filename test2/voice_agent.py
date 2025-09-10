# voice_agent.py
# Bilingvalni (HR/EN) voice agent s lokalnom transkripcijom (faster-whisper),
# OpenAI za razum/odgovor i ElevenLabs TTS.
# Verzija: 2025-09-10 (stabilna, bez webrtcvad — koristi RMS VAD)

import os
import io
import sys
import time
import wave
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import OpenAI
from elevenlabs import ElevenLabs
from scipy.signal import resample  # za resampling u 16 kHz

# =========================
# Konfiguracija / Parametri
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "vFQACl5nAIV0owAavYxE")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "GoranS/whisper-large-v3-turbo-hr-parla-ctranslate2")

# Preferirani ulaz (opcionalno):
#   - PREFERRED_INPUT_NAME: dio naziva npr. "Microphone", "USB", "Realtek"
#   - INPUT_DEVICE_INDEX: konkretan indeks iz ispisa uređaja (npr. 1, 3, 8)
PREFERRED_INPUT_NAME = os.getenv("PREFERRED_INPUT_NAME", "").strip() or None
INPUT_DEVICE_INDEX = os.getenv("INPUT_DEVICE_INDEX", "").strip()
INPUT_DEVICE_INDEX = int(INPUT_DEVICE_INDEX) if INPUT_DEVICE_INDEX.isdigit() else None

# Audio
TARGET_SR = 16000       # Whisperu paše 16k
CHANNELS = 1
FRAME_DURATION_MS = 30  # koristi se samo za VAD "okvire" (RMS), ne mora biti strogo
FRAME_SAMPLES = int(TARGET_SR * FRAME_DURATION_MS / 1000)

# Snimanje / VAD
MAX_RECORD_SECS = 60          # maksimalno trajanje jedne snimke
SILENCE_TIMEOUT_SECS = 2.5    # koliko tišine označava kraj govora
RMS_THRESH = 0.003            # osjetljivost na govor (spuštaj prema 0.002/0.0015 ako si tih)

# Alternativa: fiksno snimanje bez VAD-a (za dijagnostiku)
USE_FIXED_SECONDS = os.getenv("USE_FIXED_SECONDS", "").strip()
USE_FIXED_SECONDS = int(USE_FIXED_SECONDS) if USE_FIXED_SECONDS.isdigit() else None  # npr. 5

# OpenAI model
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

# =========================
# Pomoćne funkcije (Audio)
# =========================

def list_audio_devices():
    """Ispiši sve dostupne audio uređaje (debug/info)."""
    print("Audio uređaji:")
    for i, d in enumerate(sd.query_devices()):
        name = d.get('name', '?')
        in_ch = d.get('max_input_channels', 0)
        out_ch = d.get('max_output_channels', 0)
        sr = d.get('default_samplerate', None)
        print(f"[{i:02d}] {name}  in:{in_ch}  out:{out_ch}  sr:{sr}")

def pick_input_device(prefer_name_substr=None, prefer_index=None):
    """
    Vrati indeks input uređaja:
      - ako je zadan prefer_index, koristi njega ako je valjan
      - inače ako je zadan prefer_name_substr, pokušaj prvo to po imenu
      - u suprotnom prvi uređaj s input kanalima
    """
    devices = sd.query_devices()

    # Preferirani indeks
    if isinstance(prefer_index, int) and 0 <= prefer_index < len(devices):
        if devices[prefer_index]['max_input_channels'] > 0:
            return prefer_index

    # Pretraga po nazivu
    if prefer_name_substr:
        p = prefer_name_substr.lower()
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0 and p in d.get('name', '').lower():
                return i

    # Prvi s input kanalima
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            return i

    raise RuntimeError("Nema dostupnih input audio uređaja (mikrofona).")


def resample_to_16k(audio_np: np.ndarray, src_sr: int) -> np.ndarray:
    """Resamplaj mono float32 -1..1 u 16 kHz, ako treba."""
    if src_sr == TARGET_SR:
        return audio_np
    target_len = int(len(audio_np) * TARGET_SR / src_sr)
    if target_len <= 0:
        return np.zeros(1, dtype=np.float32)
    return resample(audio_np, target_len).astype(np.float32)


def float32_to_wav_bytes(audio_np: np.ndarray, sr: int) -> io.BytesIO:
    """Spremi mono float32 -1..1 u 16-bit PCM WAV u memoriji (na TARGET_SR)."""
    audio_16k = resample_to_16k(audio_np, sr)
    int16 = np.clip(audio_16k * 32767, -32768, 32767).astype(np.int16)

    wav_buf = io.BytesIO()
    with wave.open(wav_buf, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(TARGET_SR)
        wf.writeframes(int16.tobytes())
    wav_buf.seek(0)
    return wav_buf


# =========================
# Snimanje (RMS VAD ili fiksno)
# =========================

def record_fixed(seconds=5, device_index=None):
    """Snimaj fiksno 'seconds' sekundi bez VAD-a (za dijagnostiku / fallback)."""
    print(f"🎙️ Snimam {seconds}s… (fiksno)")
    dev_index = device_index if device_index is not None else pick_input_device(PREFERRED_INPUT_NAME, INPUT_DEVICE_INDEX)
    dev_info = sd.query_devices(dev_index)
    device_sr = int(dev_info.get("default_samplerate") or TARGET_SR)

    audio = sd.rec(int(seconds * device_sr),
                   samplerate=device_sr,
                   channels=CHANNELS,
                   dtype='float32',
                   device=dev_index)
    sd.wait()

    audio_np = audio.reshape(-1).astype(np.float32)
    print("⛳ Kraj fiksnog snimanja.")
    return float32_to_wav_bytes(audio_np, device_sr)


def record_until_silence():
    """
    Snimaj 1 kanal float32 na SR uređaja; prati RMS i prekini nakon SILENCE_TIMEOUT_SECS tišine.
    Nema vanjskih C++ ovisnosti (nema webrtcvad).
    """
    import threading

    dev_index = pick_input_device(PREFERRED_INPUT_NAME, INPUT_DEVICE_INDEX)
    dev_info = sd.query_devices(dev_index)
    device_sr = int(dev_info.get("default_samplerate") or TARGET_SR)

    print("🎙️  Pritisni Enter pa govori (CTRL+C za izlaz).")
    input()
    print("Snimam... pričaj. (pauza = kraj)")

    last_voice_time = time.time()
    start_time = time.time()
    q = []          # spremamo raw float32 bytes okvira
    q_lock = threading.Lock()
    stopped = False

    frame_samples_dev = max(256, int(device_sr * FRAME_DURATION_MS / 1000))

    def callback(indata, frames, time_info, status):
        nonlocal last_voice_time
        if status:
            # ispiši ali nastavi raditi
            print(status, file=sys.stderr)

        mono = indata.copy().reshape(-1)  # float32 -1..1
        # RMS po okviru
        rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
        # Debug: otkomentiraj da vidiš tipične RMS vrijednosti
        # print(f"rms={rms:.5f}")

        with q_lock:
            q.append(mono.tobytes())

        if rms > RMS_THRESH:
            last_voice_time = time.time()

    try:
        with sd.InputStream(device=dev_index,
                            channels=CHANNELS,
                            samplerate=device_sr,
                            dtype='float32',
                            blocksize=frame_samples_dev,
                            callback=callback):
            while True:
                time.sleep(0.05)
                if time.time() - last_voice_time > SILENCE_TIMEOUT_SECS:
                    break
                if time.time() - start_time > MAX_RECORD_SECS:
                    break
    except Exception as e:
        print("Greška pri otvaranju audio streama:", e)
        # Ako ne uspije, pokušaj fiksno 5s kao fallback
        return record_fixed(5, device_index=dev_index)

    # Spoji sve float32 okvire
    with q_lock:
        if not q:
            print("⏹️  Kraj snimanja (nema prikupljenih okvira).")
            return io.BytesIO()  # prazno
        audio_float = b"".join(q)
    audio_np = np.frombuffer(audio_float, dtype=np.float32)

    print("⏹️  Kraj snimanja.")
    return float32_to_wav_bytes(audio_np, device_sr)


# =========================
# Transkripcija (faster-whisper)
# =========================

def load_whisper():
    print("Učitavam Whisper model (ovo može potrajati samo prvi put)...")
    return WhisperModel(
        WHISPER_MODEL,
        device="auto",        # "cuda" ako imaš GPU
        compute_type="auto",  # npr. "int8_float16" na GPU, "int8" na CPU
    )

def transcribe(whisper: WhisperModel, wav_buf: io.BytesIO):
    """Whisper transkripcija s auto detekcijom jezika (hr/en)."""
    if wav_buf.getbuffer().nbytes < 32000:  # ~1 s
        return "", None

    tmp_path = "temp_in.wav"
    with open(tmp_path, "wb") as f:
        f.write(wav_buf.read())
    wav_buf.seek(0)

    segments, info = whisper.transcribe(
        tmp_path,
        beam_size=3,
        vad_filter=False,          # naš RMS VAD je već odradio
        temperature=0.0,
        language=None,             # auto
        without_timestamps=True,
    )
    text = "".join(seg.text for seg in segments).strip()
    lang = info.language if hasattr(info, "language") else None  # npr. 'hr' ili 'en'
    return text, lang


# =========================
# OpenAI (razum/odgovor)
# =========================

def init_openai():
    if not OPENAI_API_KEY:
        raise RuntimeError("Nedostaje OPENAI_API_KEY u .env")
    return OpenAI(api_key=OPENAI_API_KEY)

def openai_chat(oa: OpenAI, text: str, lang_hint: str | None):
    """Odgovori istim jezikom koji korisnik koristi (HR/EN)."""
    system = (
        "You are a concise, helpful bilingual assistant for Croatian and English. "
        "Detect the user's language (Croatian or English) and always reply in that language. "
        "If the user mixes languages, keep their dominant language. "
        "Keep answers short for voice (2–4 sentences max)."
    )
    if (lang_hint or "").startswith("hr"):
        system += " Prefer Croatian if the user speaks Croatian."
    else:
        system += " Prefer English if the user speaks English."

    resp = oa.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content.strip()


# =========================
# ElevenLabs (TTS)
# =========================

def init_elevenlabs():
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("Nedostaje ELEVENLABS_API_KEY u .env")
    return ElevenLabs(api_key=ELEVENLABS_API_KEY)

def tts_and_play(el: ElevenLabs, text: str, voice_id: str = ELEVENLABS_VOICE_ID):
    """ElevenLabs TTS i reprodukcija (pcm_16000 raw)."""
    audio_gen = el.text_to_speech.convert(
        voice_id=voice_id,
        optimize_streaming_latency="0",
        output_format="pcm_16000",   # raw 16kHz PCM
        text=text,
        model_id="eleven_multilingual_v2"
    )
    pcm_bytes = b"".join(audio_gen)

    # Pretvori u float32 -1..1
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    pcm = pcm.reshape(-1, 1)  # mono kanal

    print("🔊 Reproduciram odgovor...")
    sd.play(pcm, TARGET_SR)
    sd.wait()



# =========================
# Main petlja
# =========================

def main():
    # Ispis uređaja (korisno pri prvom pokretanju)
    list_audio_devices()

    oa = init_openai()
    el = init_elevenlabs()
    whisper = load_whisper()

    print("Bilingvalni voice agent spreman. (HR/EN)")

    while True:
        try:
            # Odaberi način snimanja
            if USE_FIXED_SECONDS:
                wav_buf = record_fixed(USE_FIXED_SECONDS)
            else:
                wav_buf = record_until_silence()

            if wav_buf.getbuffer().nbytes < 32000:
                print("Nisam ništa čuo. Pokušaj opet.")
                continue

            user_text, lang = transcribe(whisper, wav_buf)
            if not user_text:
                print("Nisam uspio transkribirati. Pokušaj opet.")
                continue

            flag = "🇭🇷" if (lang or "").startswith("hr") else "🇬🇧"
            print(f"{flag} Ti: {user_text}")

            reply = openai_chat(oa, user_text, lang)
            print(f"🤖 Asistent: {reply}")

            tts_and_play(el, reply)

        except KeyboardInterrupt:
            print("\nIzlaz. Hvala!")
            break
        except Exception as e:
            print("Greška:", e)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
