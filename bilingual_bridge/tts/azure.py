import httpx


AZURE_TTS_URL = "https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"


class AzureTTS:
    def __init__(self, key: str, region: str, voice_default: str):
        self.key = key
        self.region = region
        self.voice_default = voice_default


    async def synthesize_mulaw8k(self, text: str, voice: str) -> bytes:
        # Directly synthesize in mu-law 8k to avoid resampling/encoding
        ssml = f"""
        <speak version='1.0' xml:lang='en-US'>
            <voice name='{voice}'>{text}</voice>
        </speak>
        """.strip()
        url = AZURE_TTS_URL.format(region=self.region)
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "raw-8khz-8bit-mulaw-mono",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, headers=headers, content=ssml)
            r.raise_for_status()
            return r.content