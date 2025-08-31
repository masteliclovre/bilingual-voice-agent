import WebSocket from 'ws';

export type HREvent = { text: string; isFinal: boolean; confidence?: number };

export function connectHR(wssUrl: string) {
  const ws = new WebSocket(wssUrl);
  const listeners: Array<(e: HREvent) => void> = [];
  ws.on('message', (data, isBinary) => {
    if (isBinary) return;
    try {
      const msg = JSON.parse(data.toString());
      if (msg.type === 'partial' || msg.type === 'final') {
        listeners.forEach(cb => cb({
          text: msg.text || '',
          isFinal: msg.type === 'final',
          confidence: msg.confidence
        }));
      }
    } catch {}
  });
  return {
    ws,
    onEvents(cb: (e: HREvent) => void) { listeners.push(cb); },
    ready(): Promise<void> { return new Promise(res => ws.once('open', () => res())); },
    send(pcmMono16k: Buffer) { ws.send(pcmMono16k); },
    close() { ws.close(); }
  };
}
