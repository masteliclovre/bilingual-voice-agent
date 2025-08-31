import 'dotenv/config';
import http from 'http';
import express from 'express';
import { WebSocketServer, WebSocket } from 'ws';
import { makeDeepgramClient, DGEvent } from './deepgramClient.js';
import { connectHR, HREvent } from './hrClient.js';
import { stereoToMono16le } from './audio.js';

// ---- env ----
const PORT = Number(process.env.PORT || 3001);
const DG_KEY = process.env.DEEPGRAM_API_KEY || '';
const HR_WSS = process.env.HR_STT_WSS || 'ws://localhost:8081/stream';
const SECRET = process.env.SECRET;
const LANGLOCK_MS = Number(process.env.LANGLOCK_MS || 2500);
const HR_LOCK_CONF = Number(process.env.HR_CONFIDENCE_LOCK || 0.75);

// ---- http + ws server ----
const app = express();
app.get('/', (_req, res) => res.send('Vapi Custom Transcriber Router OK'));
const server = http.createServer(app);

type LockLang = 'HR' | 'EN' | null;

const wss = new WebSocketServer({ server, path: '/router' });

wss.on('connection', (ws: WebSocket, req) => {
  // Optional simple auth
  if (SECRET && req.headers['x-vapi-secret'] !== SECRET) {
    try {
      ws.close(1008, 'unauthorized');
    } catch {}
    return;
  }

  let started = false;
  let sampleRate = 16000;
  let channels = 2;
  let locked: LockLang = null;

  // Create HR client immediately (mono 16k path), Deepgram after we know sampleRate
  const hr = connectHR(HR_WSS);
  let dg: ReturnType<typeof makeDeepgramClient> | null = null;

  // Helpers
  const sendResp = (text: string, channel: 'customer' | 'assistant') => {
    const msg = { type: 'transcriber-response', transcription: text, channel };
    try {
      ws.send(JSON.stringify(msg));
    } catch {}
  };

  let hrBestConf = 0;

  // Forward HR events (only when locked === 'HR')
  hr.onEvents((e: HREvent) => {
    const { text, isFinal, confidence } = e;
    if (!locked) {
      // Update best HR confidence during race
      if ((confidence ?? 0) > hrBestConf) hrBestConf = confidence ?? 0;
      if ((confidence ?? 0) >= HR_LOCK_CONF && text && text.length > 2) {
        locked = 'HR';
        console.log('Language locked early:', locked);
        clearTimeout(lockTimer);
      }
      return;
    }
    if (locked !== 'HR') return;
    if (text) sendResp(text, 'customer'); // HR stream treated as caller channel
  });

  // Forward DG events (only when locked === 'EN')
  const attachDeepgramHandlers = () => {
    if (!dg) return;
    dg.onEvents((e: DGEvent) => {
      const { text, isFinal, channel, confidence } = e;
      if (!locked) {
        // During race we let timeout decide unless HR already locked
        return;
      }
      if (locked !== 'EN') return;
      if (text) sendResp(text, channel);
    });
  };

  // Race timeout → decide lock if not decided yet
  const lockTimer = setTimeout(() => {
    if (!locked) {
      locked = hrBestConf >= HR_LOCK_CONF ? 'HR' : 'EN';
      console.log('Language locked by timeout:', locked, `(hrBestConf=${hrBestConf})`);
    }
  }, LANGLOCK_MS);

  // Incoming messages: Vapi sends a JSON "start" then binary PCM frames
  ws.on('message', (data: Buffer, isBinary) => {
    if (!isBinary) {
      // likely the "start" message
      try {
        const msg = JSON.parse(data.toString());
        if (msg?.type === 'start') {
          started = true;
          sampleRate = Number(msg.sampleRate || 16000);
          channels = Number(msg.channels || 2);
          console.log('Vapi start', { sampleRate, channels });

          // Initialize Deepgram live using the negotiated sample rate
          dg = makeDeepgramClient(DG_KEY, sampleRate);
          attachDeepgramHandlers();

          // Ensure HR ws is open (fire-and-forget)
          hr.ready().catch(() => {});
        }
      } catch {
        // ignore non-JSON control messages
      }
      return;
    }

    // Binary PCM frame from Vapi
    if (!started) return;

    const stereoOrMono = data;
    const mono16 = channels === 2 ? stereoToMono16le(stereoOrMono) : stereoOrMono;

    if (!locked) {
      // During race: feed both
      if (dg) dg.send(stereoOrMono);
      hr.send(mono16);
      return;
    }

    // After lock: send only to the chosen engine
    if (locked === 'EN') {
      if (dg) dg.send(stereoOrMono);
    } else {
      hr.send(mono16);
    }
  });

  ws.on('close', () => {
    clearTimeout(lockTimer);
    try {
      dg?.close();
    } catch {}
    try {
      hr.close();
    } catch {}
  });
});

server.listen(PORT, () => {
  console.log('Router listening on', PORT);
});
