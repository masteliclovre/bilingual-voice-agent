import "dotenv/config";
import axios from "axios";

const ROUTER_URL = process.env.ROUTER_URL;
const VAPI_SECRET_KEY = process.env.VAPI_SECRET_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID;

if (!VAPI_SECRET_KEY || !ROUTER_URL) {
  console.error("Missing VAPI_SECRET_KEY or ROUTER_URL in .env");
  process.exit(1);
}

const assistantConfig = {
  name: "Bilingual Croatian Agent",
  transcriber: {
    provider: "custom-transcriber",
    server: { url: ROUTER_URL }
  },
  voice: {
    provider: "11labs",
    voiceId: ELEVENLABS_VOICE_ID
  },
  model: {
    provider: "openai",
    model: "gpt-4o-mini"
  },
  systemPrompt: "You are a polite bilingual receptionist. Default to Croatian; switch to English if caller uses English. In Croatian always use formal 'Vi'. Keep answers short and clear."
};

async function upsertAssistant() {
  const vapi = axios.create({
    baseURL: "https://api.vapi.ai",
    headers: { Authorization: `Bearer ${VAPI_SECRET_KEY}` }
  });

  const res = await vapi.post("/assistants", assistantConfig);
  console.log("Assistant created:", res.data);
}

upsertAssistant().catch(e => {
  console.error(e?.response?.data || e.message);
  process.exit(1);
});
