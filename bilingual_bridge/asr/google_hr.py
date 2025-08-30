import asyncio
import threading
from queue import SimpleQueue
from typing import Optional, Tuple


from google.cloud import speech


class GoogleHRStream:
    """Streaming HR ASR using Google Speech-to-Text v1 (LINEAR16 @ 8 kHz).


    This class hides the blocking gRPC stream behind async queues.
    """

    def __init__(self, project_id: str):
        self.client = speech.SpeechClient()
        self.project_id = project_id
        self._audio_q: SimpleQueue[Optional[bytes]] = SimpleQueue()
        self._out_q: asyncio.Queue[Tuple[str, bool]] = asyncio.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()
    
    def _requests(self):
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=8000,
            language_code="hr-HR",
            enable_automatic_punctuation=True,
            model="latest_long",
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
            single_utterance=False,
        )
        # First message with config
        yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
        # Then raw audio chunks
        while not self._stopped.is_set():
            b = self._audio_q.get()
            if b is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=b)
    
    def _run(self, loop: asyncio.AbstractEventLoop):
        responses = self.client.streaming_recognize(requests=self._requests())
        for resp in responses:
            for result in resp.results:
                if result.alternatives:
                    txt = result.alternatives[0].transcript
                    is_final = bool(result.is_final)
                    if txt:
                        asyncio.run_coroutine_threadsafe(
                            self._out_q.put((txt, is_final)), loop
                        )
    
    async def start(self):
        loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._run, args=(loop,), daemon=True)
        self._thread.start()


async def send_pcm16(self, pcm16_bytes: bytes):
    self._audio_q.put(pcm16_bytes)


async def get_transcript(self) -> Tuple[str, bool]:
    return await self._out_q.get()


async def stop(self):
    self._stopped.set()
    self._audio_q.put(None)
    if self._thread:
        self._thread.join(timeout=2.0)