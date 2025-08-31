import { createClient, LiveTranscriptionEvents } from '@deepgram/sdk';

export type DGEvent = {
  text: string;
  isFinal: boolean;
  channel: 'customer' | 'assistant';
  confidence?: number;
};

export function makeDeepgramClient(apiKey: string, sampleRate: number) {
  const client = createClient(apiKey);
  const live = client.listen.live({
    encoding: 'linear16',
    channels: 2,
    sample_rate: sampleRate,
    model: 'nova-3',
    smart_format: true,
    interim_results: true,
    multichannel: true,
    endpointing: 800,
  });

  const onEvents = (cb: (e: DGEvent) => void) => {
    live.on(LiveTranscriptionEvents.Transcript, (ev: any) => {
      const alt = ev?.channel?.alternatives?.[0];
      const text = alt?.transcript || '';
      if (!text) return;
      const isFinal = !!ev?.is_final;
      const chIndex = ev?.channel_index?.[0] ?? 0;
      const channel: 'customer'|'assistant' = chIndex === 0 ? 'customer' : 'assistant';
      const confidence = alt?.confidence as number | undefined;
      cb({ text, isFinal, channel, confidence });
    });
  };

  // Deepgram Node klijent očekuje SocketDataLike: ArrayBuffer | SharedArrayBuffer | Blob
  // Pretvori Buffer u ArrayBuffer (bez kopije ili sa malom kopijom)
  const send = (pcm: Buffer) => {
    // bez kopije:
    const ab = pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength);
    // ili s kopijom:
    // const ab = new Uint8Array(pcm).buffer;
    live.send(ab as ArrayBuffer);
  };

  const close = () => live.finish();

  return { live, onEvents, send, close };
}
