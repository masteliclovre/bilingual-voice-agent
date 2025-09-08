# server.py
import asyncio
import json
import time
import numpy as np

from fastapi import FastAPI, WebSocket
from fastapi.responses import PlainTextResponse
from starlette.websockets import WebSocketState, WebSocketDisconnect

from transformers import pipeline
import torch

# ---------- ASR PIPELINE ----------
device = 0 if torch.cuda.is_available() else -1
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("Using device:", torch.cuda.get_device_name(0))
    except Exception:
        pass

asr = pipeline(
    "automatic-speech-recognition",
    model="GoranS/whisper-large-v3-turbo-hr-parla",
    chunk_length_s=8,           # manji prozor -> brže
    stride_length_s=(3, 1),     # preklapanje
    return_timestamps=False,
    torch_dtype=torch.float16 if device == 0 else None,
    device=device,
)

# ---------- APP ----------
app = FastAPI()

@app.get("/")
def root():
    return PlainTextResponse("HR Transcriber OK")

@app.websocket("/api/custom-transcriber")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    # Vapi šalje "start" s realnim parametrima; ovo su defaulti
    sample_rate = 16000
    channels = 2

    BYTES_PER_SAMPLE = 2
    SILENCE_FINAL_S = 0.4        # koliko tišine znači "završi segment"
    MAX_SEGMENT_S = 6.0          # sigurnosna granica: ako govor traje dugo, svejedno pošalji final

    frame_bytes = bytearray()
    last_audio_ts = time.time()
    closed = False

    # Debounce stanje
    final_sent_for_current_segment = False
    flush_lock = asyncio.Lock()

    async def safe_send_text(payload: dict):
        """Šalji samo ako je socket živ; ignoriraj race-eve."""
        try:
            if (ws.application_state == WebSocketState.CONNECTED
                and ws.client_state == WebSocketState.CONNECTED):
                await ws.send_text(json.dumps(payload))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print("safe_send_text error:", repr(e))

    async def safe_close():
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            if (ws.application_state != WebSocketState.DISCONNECTED
                and ws.client_state != WebSocketState.DISCONNECTED):
                await ws.close(code=1000)
        except Exception:
            pass

    def seconds_in_buffer() -> float:
        return len(frame_bytes) / (BYTES_PER_SAMPLE * channels * sample_rate)

    async def do_asr_and_maybe_send_final(force_final: bool):
        """
        Pretvori buffer -> mono float32 -> ASR.
        Šalje Vapi-ju SAMO final (nikakve partijale).
        """
        nonlocal frame_bytes, sample_rate, channels, final_sent_for_current_segment

        async with flush_lock:
            # Ako nema ničega ili smo već poslali final za ovaj segment, izađi.
            if not frame_bytes or final_sent_for_current_segment:
                return

            secs = seconds_in_buffer()
            # Pošalji final ako je tišina dosta duga ili je segment predug ili je forsirano.
            if not (force_final or (time.time() - last_audio_ts) >= SILENCE_FINAL_S or secs >= MAX_SEGMENT_S):
                return

            # int16 -> float32 [-1,1]
            pcm = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if channels == 2:
                if (len(pcm) % 2) != 0:
                    print("WARN: misaligned stereo frame; dropping chunk")
                    frame_bytes = bytearray()
                    return
                pcm = pcm.reshape(-1, 2).mean(axis=1)

            try:
                out = asr(
                    {"array": pcm, "sampling_rate": sample_rate},
                    generate_kwargs={"language": "hr", "task": "transcribe"},
                )
                text = (out.get("text") or "").strip()
                print(f"[ASR-FINAL] {len(pcm)} samples @ {sample_rate}Hz -> '{text}'")
                if text:
                    print(f"[SEND] final: {text}")
                    # Vapi očekuje baš ovaj oblik poruke
                    await safe_send_text({
                        "type": "transcriber-response",
                        "transcription": text,
                        "channel": "customer"
                    })
                    final_sent_for_current_segment = True
            except Exception as e:
                print("ASR error:", repr(e))
            finally:
                # Očisti buffer za novi segment
                frame_bytes = bytearray()

    # Pozadinski task: promatra tišinu i, kad je vrijeme, šalje final (jednom)
    stop_flag = {"stop": False}
    async def idle_finalizer():
        while not stop_flag["stop"]:
            try:
                await do_asr_and_maybe_send_final(force_final=False)
            except Exception as e:
                print("idle_finalizer error:", repr(e))
            await asyncio.sleep(0.1)

    bg = asyncio.create_task(idle_finalizer())

    try:
        while True:
            msg = await ws.receive()

            if msg.get("text") is not None:
                # očekujemo {"type":"start", "sampleRate":..., "channels":..., "encoding":"linear16","container":"raw"}
                try:
                    payload = json.loads(msg["text"])
                    if payload.get("type") == "start":
                        sample_rate = int(payload.get("sampleRate", 16000))
                        channels = int(payload.get("channels", 2))
                        print("[START]",
                              f"SR={sample_rate}Hz, ch={channels},",
                              f"enc={payload.get('encoding')}, cont={payload.get('container')}")
                        # Ako koristiš shared secret, provjeri ovdje:
                        # if payload.get("secret") != "YOUR_SECRET": await safe_close(); return
                except Exception as e:
                    print("Start msg parse error:", e)

            elif msg.get("bytes") is not None:
                data = msg["bytes"]
                frame_size = BYTES_PER_SAMPLE * channels
                cut = len(data) - (len(data) % frame_size)
                if cut > 0:
                    frame_bytes.extend(data[:cut])
                    last_audio_ts = time.time()
                    # stigao novi govor -> dopušten je final za novi segment
                    final_sent_for_current_segment = False

                # Ovdje NE radimo partial ASR (radi brzine).
                # Final ide iz idle_finalizer-a (tišina / max trajanje) ili u finally bloku.

            else:
                # client zatražio zatvaranje
                break

    except WebSocketDisconnect:
        # normalno kad Vapi prekine poziv (tišina, kraj, itd.)
        pass
    except Exception as e:
        print("WS loop error:", repr(e))
    finally:
        stop_flag["stop"] = True
        try:
            # Pošalji final samo ako nije poslan i još ima nešto u bufferu.
            if frame_bytes and not final_sent_for_current_segment:
                await do_asr_and_maybe_send_final(force_final=True)
        except Exception as e:
            print("Final flush error:", repr(e))
        await safe_close()
        try:
            await bg
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=3002, reload=False)
