"""Minimal proxy that bridges local clients to the OpenAI Realtime WebSocket API.

The proxy keeps your OpenAI API key on the server while letting untrusted
clients push microphone audio in real time.  Each incoming client connection
spawns a fresh Realtime session with OpenAI and relays events in both
directions until either side disconnects.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import websockets


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("realtime_proxy")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in the proxy environment.")

OPENAI_REALTIME_MODEL = os.getenv(
    "OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17"
)
PROXY_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8081"))


async def relay_messages(
    client_ws: websockets.WebSocketServerProtocol,
    openai_ws: websockets.WebSocketClientProtocol,
) -> None:
    """Relay messages between the end user and OpenAI Realtime."""

    async def forward(src: websockets.WebSocketCommonProtocol,
                      dst: websockets.WebSocketCommonProtocol,
                      direction: str) -> None:
        try:
            async for message in src:
                await dst.send(message)
        except websockets.ConnectionClosed:
            LOGGER.debug("%s connection closed", direction)
        finally:
            await asyncio.gather(
                asyncio.create_task(dst.close()), return_exceptions=True
            )

    await asyncio.gather(
        forward(client_ws, openai_ws, "client"),
        forward(openai_ws, client_ws, "openai"),
    )


async def handle_client(client_ws: websockets.WebSocketServerProtocol) -> None:
    peer = getattr(client_ws, "remote_address", ("?", "?"))
    LOGGER.info("Client connected from %s:%s", peer[0], peer[1])

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }
    url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"

    try:
        async with websockets.connect(url, additional_headers=headers) as openai_ws:
            await relay_messages(client_ws, openai_ws)
    except Exception as exc:  # noqa: BLE001 - we want to log and notify the client
        LOGGER.exception("Proxy error: %s", exc)
        try:
            await client_ws.send(json.dumps({
                "type": "proxy_error",
                "error": str(exc),
            }))
        finally:
            await client_ws.close()


async def main() -> None:
    LOGGER.info("Starting realtime proxy on %s:%s", PROXY_HOST, PROXY_PORT)
    async with websockets.serve(handle_client, PROXY_HOST, PROXY_PORT):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("Proxy stopped by user.")
