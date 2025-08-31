import { createClient, LiveTranscriptionEvents } from '@deepgram/sdk';


export type DGEvent = { text: string; isFinal: boolean; channel: 'customer'|'assistant'; confidence?: number };


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
     endpointing: 800
});
return {
    live,
    onEvents(cb: (e: DGEvent) => void) {
        live.on(LiveTranscriptionEvents.Transcript, (ev: any) => {
            const alt = ev.channel.alternatives?.[0];
            const text = alt?.transcript || '';
            if (!text) return;
            const fin = !!ev.is_final;
            const chIndex = ev.channel_index?.[0] ?? 0;
            const channel = chIndex === 0 ? 'customer' : 'assistant';
            const conf = alt?.confidence;
            cb({ text, isFinal: fin, channel, confidence: conf });
        });
    },
    send(pcm: Buffer) { live.send(pcm); },
    close() { live.finish(); }
};
}