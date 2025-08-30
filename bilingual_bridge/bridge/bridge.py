from __future__ import annotations
import asyncio
from typing import Callable


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
        # state for interim speaking
        self._en_spoken = ""
        self._hr_spoken = ""
        self._en_tts_task: asyncio.Task | None = None
        self._hr_tts_task: asyncio.Task | None = None


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

    # ---- helpers for incremental TTS ----
    def _extract_new_segment(self, total: str, already_spoken: str) -> tuple[str, str, bool]:
        """Return (segment_to_say, new_already_spoken, should_speak_now).
        Strategy: speak only when we see sentence-ending punctuation OR length >= 12 tokens.
        """
        if not total:
            return "", already_spoken, False
        if total.startswith(already_spoken):
            remainder = total[len(already_spoken):].lstrip()
        else:
            # Hypothesis changed; rollback to common prefix
            prefix_len = 0
            for a, b in zip(total, already_spoken):
                if a != b:
                    break
                prefix_len += 1
            remainder = total[prefix_len:]
            already_spoken = total[:prefix_len]
        # trigger rules
        trigger = False
        seg = remainder
        if any(seg.endswith(p) for p in ".!?“”\""):
            trigger = True
        elif len(seg.split()) >= 12:
            trigger = True
        return seg, total if trigger else already_spoken, trigger
    
    async def _speak_cancelable(self, leg: Leg, voice: str, text: str, existing_task: asyncio.Task | None) -> asyncio.Task:
        if existing_task and not existing_task.done():
            existing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await existing_task
        audio = await self.tts.synthesize_mulaw8k(text, voice=voice)
        return asyncio.create_task(leg.send_mulaw_stream(audio))


    async def _consume_hr_to_en(self):
        while True:
            txt, is_final = await self.hr_asr.get_transcript()
            if not txt:
                continue
            seg, new_total, should = self._extract_new_segment(txt, self._en_spoken)
            if should and seg.strip():
                translated = self.translator.translate(seg, src="hr", tgt="en")
                self._en_tts_task = await self._speak_cancelable(self.leg_en, self.azure_en_voice, translated, self._en_tts_task)
                self._en_spoken = new_total
            if is_final:
                # flush any remainder
                remainder = txt[len(self._en_spoken):]
                if remainder.strip():
                    translated = self.translator.translate(remainder, src="hr", tgt="en")
                    self._en_tts_task = await self._speak_cancelable(self.leg_en, self.azure_en_voice, translated, self._en_tts_task)
                self._en_spoken = txt
    
    async def _consume_en_to_hr(self):
        while True:
            txt, is_final = await self.en_asr.get_transcript()
            if not txt:
                continue
            seg, new_total, should = self._extract_new_segment(txt, self._hr_spoken)
            if should and seg.strip():
                translated = self.translator.translate(seg, src="en", tgt="hr")
                self._hr_tts_task = await self._speak_cancelable(self.leg_hr, self.azure_hr_voice, translated, self._hr_tts_task)
                self._hr_spoken = new_total
            if is_final:
                remainder = txt[len(self._hr_spoken):]
                if remainder.strip():
                    translated = self.translator.translate(remainder, src="en", tgt="hr")
                    self._hr_tts_task = await self._speak_cancelable(self.leg_hr, self.azure_hr_voice, translated, self._hr_tts_task)
                self._hr_spoken = txt