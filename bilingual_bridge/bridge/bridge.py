from __future__ import annotations
import asyncio
from typing import Callable, Literal

from translate.gcp_translate import Translator
from tts.azure import AzureTTS
from asr.google_hr import GoogleHRStream
from asr.deepgram_en import DeepgramENStream


class Bridge:
    """Binds two legs with ASR->Translate->TTS pipelines in both directions."""


    def __init__(self,
                leg_hr: Leg,
                leg_en: Leg,
                hr_asr: GoogleHRStream,
                en_asr: DeepgramENStream,
                translator: Translator,
                tts: AzureTTS,
                azure_hr_voice: str,
                azure_en_voice: str):
        self.leg_hr = leg_hr
        self.leg_en = leg_en
        self.hr_asr = hr_asr
        self.en_asr = en_asr
        self.translator = translator
        self.tts = tts
        self.azure_hr_voice = azure_hr_voice
        self.azure_en_voice = azure_en_voice
        self._tasks: list[asyncio.Task] = []
        self._running = False
    
    async def start(self):
        if self._running:
            return
        self._running = True
        await self.hr_asr.start()
        await self.en_asr.start()
        self._tasks = [
            asyncio.create_task(self._feed_asr(self.leg_hr, self.hr_asr)),
            asyncio.create_task(self._feed_asr(self.leg_en, self.en_asr)),
            asyncio.create_task(self._consume_hr_to_en()),
            asyncio.create_task(self._consume_en_to_hr()),
        ]
    
    async def stop(self):
        for t in self._tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
        await self.hr_asr.stop()
        await self.en_asr.stop()
        self._running = False


    async def _feed_asr(self, leg: Leg, asr):
        while True:
            pcm = await leg.in_pcm_q.get()
            await asr.send_pcm16(pcm)

    async def _consume_hr_to_en(self):
        """HR speech -> HR ASR -> translate HR→EN -> TTS EN -> play on EN leg"""
        while True:
            txt, is_final = await self.hr_asr.get_transcript()
            if not txt:
                continue
            if not is_final:
                # For low-latency you can stream partials; here we only speak finals
                continue
            translated = self.translator.translate(txt, src="hr", tgt="en")
            audio = await self.tts.synthesize_mulaw8k(translated, voice=self.azure_en_voice)
            await self.leg_en.send_mulaw_stream(audio)
    
    async def _consume_en_to_hr(self):
        """EN speech -> EN ASR -> translate EN→HR -> TTS HR -> play on HR leg"""
        while True:
            txt, is_final = await self.en_asr.get_transcript()
            if not txt:
                continue
            if not is_final:
                continue
            translated = self.translator.translate(txt, src="en", tgt="hr")
            audio = await self.tts.synthesize_mulaw8k(translated, voice=self.azure_hr_voice)
            await self.leg_hr.send_mulaw_stream(audio)