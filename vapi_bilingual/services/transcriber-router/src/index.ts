import 'dotenv/config';
import http from 'http';
import express from 'express';
import { WebSocketServer } from 'ws';
import { makeDeepgramClient } from './deepgramClient';
import { connectHR } from './hrClient';
import { stereoToMono16le } from './audio';

const PORT = Number(process.env.PORT || 3001);
const DG_KEY = process.env.DEEPGRAM_API_KEY!;
const HR_WSS = process.env.HR_STT_WSS!;
const SECRET = process.env.SECRET;
const LANGLOCK_MS = Number(process.env.LANGLOCK_MS || 2500);
const HR_LOCK_CONF = Number(process.env.HR_CONFIDENCE_LOCK || 0.75);

const app = express();
app.get('/', (_, res) => res.send('Vapi Custom Transcriber Router OK'));
const server = http.createServer(app);


const wss = new WebSocketServer({ server, path: '/router' });


wss.on('connection', (ws, req) => {
    if (SECRET && req.headers['x-vapi-secret'] !== SECRET) {
        ws.close(1008, 'unauthorized');
        return;
    }
    let sampleRate = 16000; let channels = 2;
    let started = false; let locked: 'HR'|'EN'|null = null;


    // Deepgram and HR connections
    const dg = makeDeepgramClient(DG_KEY, sampleRate);
    const hr = connectHR(HR_WSS);

    // Forward transcripts back to Vapi
    function sendResp(text: string, channel: 'customer'|'assistant') {
        const msg = { type: 'transcriber-response', transcription: text, channel };
        ws.send(JSON.stringify(msg));
    }


    let hrBestConf = 0; let hrLastText = '';


    dg.onEvents(({ text, isFinal, channel, confidence }) => {
        if (!locked) return; // We only emit EN transcripts after lock
        if (locked !== 'EN') return;
        if (text) sendResp(text, channel);
    });

    hr.onEvents(({ text, isFinal, confidence }) => {
        if (!locked) return;
        if (locked !== 'HR') return;
        if (text) sendResp(text, 'customer'); // HR model is single-channel (assume caller)
    });


    hr.ready().catch(()=>{});


    // Race & lock timer
    const lockTimer = setTimeout(() => {
        if (!locked) {
            // Pick by best seen HR confidence, else EN default
            locked = hrBestConf >= HR_LOCK_CONF ? 'HR' : 'EN';
            console.log('Language locked by timeout:', locked);
        }
    }, LANGLOCK_MS);

    ws.on('message', (data, isBinary) => {
    if (!isBinary) {
        try {
            const msg = JSON.parse(data.toString());
            if (msg.type === 'start') {
                started = true;
                sampleRate = msg.sampleRate; channels = msg.channels;
                console.log('Vapi start', { sampleRate, channels });
            }
        } catch {}
        return;
    }
    if (!started) return;


    const stereo = data as Buffer;
    const mono = stereoToMono16le(stereo);


    // Feed both until locked
    if (!locked) {
        dg.send(stereo);
        hr.send(mono);
        // update heuristics for HR confidence via last partials
        // (We rely on HR partial events setting hrBestConf)
    } else if (locked === 'EN') {
        dg.send(stereo);
    } else if (locked === 'HR') {
        hr.send(mono);
    }
    });

    // Update lock based on earliest strong signal
    hr.onEvents(({ text, isFinal, confidence }) => {
        if (!locked && (confidence ?? 0) > hrBestConf) hrBestConf = confidence ?? 0;
        if (!locked && (confidence ?? 0) >= HR_LOCK_CONF && text && text.length > 2) {
            locked = 'HR'; console.log('Language locked:', locked); clearTimeout(lockTimer);
        }
    });


    dg.onEvents(({ text, isFinal, channel, confidence }) => {
        if (!locked && text && text.length > 2) {
        // Prefer EN if HR not confident enough in time window
        // Leave lock to timeout unless HR already exceeded HR_LOCK_CONF
        }
    });


    ws.on('close', () => { clearTimeout(lockTimer); try { dg.close(); } catch {}; try { hr.close(); } catch {}; });
});


server.listen(PORT, () => console.log('Router listening on', PORT));