import asyncio
import json
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from config import Config
from bridge.leg import Leg
from bridge.bridge import Bridge
from translate.gcp_translate import Translator
from tts.azure import AzureTTS
from asr.google_hr import GoogleHRStream
from asr.deepgram_en import DeepgramENStream

app = FastAPI()

app.state.leg_hr: Optional[Leg] = None
app.state.leg_en: Optional[Leg] = None
app.state.bridge: Optional[Bridge] = None


async def _send_json_ws(ws: WebSocket, obj: dict):
    await ws.send_text(json.dumps(obj))


def _both_legs_connected():
    return (app.state.leg_hr is not None) and (app.state.leg_en is not None)



async def _ensure_bridge_started():
    if app.state.bridge is not None:
        return
    if not _both_legs_connected():
        return
    translator = Translator(Config.GCP_PROJECT_ID)
    tts = AzureTTS(Config.AZURE_TTS_KEY, Config.AZURE_TTS_REGION)
    hr_asr = GoogleHRStream(Config.GCP_PROJECT_ID)
    en_asr = DeepgramENStream(Config.DG_API_KEY, model=Config.DG_EN_MODEL, sample_rate=8000)
    bridge = Bridge(
        leg_hr=app.state.leg_hr,
        leg_en=app.state.leg_en,
        hr_asr=hr_asr,
        en_asr=en_asr,
        translator=translator,
        tts=tts,
        azure_hr_voice=Config.AZURE_TTS_HR_VOICE,
        azure_en_voice=Config.AZURE_TTS_EN_VOICE,
    )
    app.state.bridge = bridge
    asyncio.create_task(bridge.start())

@app.websocket("/twilio/hr")
async def twilio_hr(ws: WebSocket):
    await ws.accept()


    async def send(obj: dict):
        await _send_json_ws(ws, obj)


    leg = Leg("HR", send_json=send)
    app.state.leg_hr = leg


    async def barge_in_other():
        if app.state.leg_en:
            await app.state.leg_en.clear_audio()


    await _ensure_bridge_started()


    try:
        while True:
            msg = await ws.receive_text()
            evt = json.loads(msg)
        await leg.on_twilio_event(evt, on_barge_in=barge_in_other)
    except WebSocketDisconnect:
        pass

@app.websocket("/twilio/en")
async def twilio_en(ws: WebSocket):
    await ws.accept()


    async def send(obj: dict):
        await _send_json_ws(ws, obj)


    leg = Leg("EN", send_json=send)
    app.state.leg_en = leg


    async def barge_in_other():
        if app.state.leg_hr:
            await app.state.leg_hr.clear_audio()


    await _ensure_bridge_started()


    try:
        while True:
            msg = await ws.receive_text()
            evt = json.loads(msg)
        await leg.on_twilio_event(evt, on_barge_in=barge_in_other)
    except WebSocketDisconnect:
        pass

    @app.get("/twiml")
    async def twiml(request: Request):
        """Return TwiML with <Connect><Stream> to either /twilio/hr or /twilio/en based on ?leg= param."""
        base_ws = str(request.url).split("/twiml")[0]
        leg = request.query_params.get("leg", "hr").lower()
        if leg not in {"hr", "en"}:
            leg = "hr"
        ws_url = base_ws.replace("http", "ws").replace("https", "wss") + f"/twilio/{leg}"
        twiml = f"""
        <?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Connect>
                <Stream url="{ws_url}" />
            </Connect>
        </Response>
        """.strip()
        return Response(content=twiml, media_type="text/xml")




@app.post("/bridge/start")
async def start_bridge():
    await _ensure_bridge_started()
    return {"status": "ok" if app.state.bridge else "waiting_for_legs"}


