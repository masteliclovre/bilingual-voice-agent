import asyncio
import json
from typing import Optional, Tuple
import websockets


class DeepgramENStream:
    """Minimal Deepgram Live client over websockets for EN @ 8kHz linear16.


    Usage:
        dg = DeepgramENStream(api_key, model="nova-3", sample_rate=8000)
        await dg.start()
        await dg.send_pcm16(pcm_bytes)
        text, is_final = await dg.get_transcript()
        await dg.stop()
    """

    def __init__(self, api_key: str, model: str = "nova-3", sample_rate: int = 8000):
        self.api_key = api_key
        self.model = model
        self.sample_rate = sample_rate
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._send_q: asyncio.Queue[bytes] = asyncio.Queue()
        self._recv_q: asyncio.Queue[Tuple[str, bool]] = asyncio.Queue()
        self._sender_task: Optional[asyncio.Task] = None
        self._receiver_task: Optional[asyncio.Task] = None

    async def start(self):
        qs = (
            f"language=en&model={self.model}&encoding=linear16&"
            f"sample_rate={self.sample_rate}&smart_format=true&interim_results=true"
        )
        url = f"wss://api.deepgram.com/v1/listen?{qs}"
        self._ws = await websockets.connect(
            url,
            extra_headers={"Authorization": f"Token {self.api_key}"},
            ping_interval=20,
            ping_timeout=20,
            max_size=5 * 1024 * 1024,
        )
        self._sender_task = asyncio.create_task(self._sender())
        self._receiver_task = asyncio.create_task(self._receiver())

    async def _sender(self):
        assert self._ws
        while True:
            chunk = await self._send_q.get()
            await self._ws.send(chunk)


    async def _receiver(self):
        assert self._ws
        async for msg in self._ws:
            try:
                data = json.loads(msg)
            except Exception:
                continue
            if data.get("type") == "results":
                ch = data.get("channel", {})
                alts = ch.get("alternatives", [])
                if not alts:
                    continue
                txt = alts[0].get("transcript", "")
                is_final = data.get("is_final", False)
                if txt:
                    await self._recv_q.put((txt, is_final))
    
    async def send_pcm16(self, pcm16_bytes: bytes):
        await self._send_q.put(pcm16_bytes)


    async def get_transcript(self) -> Tuple[str, bool]:
        return await self._recv_q.get()


    async def stop(self):
        if self._ws:
            try:
                await self._ws.close()
            finally:
                self._ws = None
        for t in (self._sender_task, self._receiver_task):
            if t:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t