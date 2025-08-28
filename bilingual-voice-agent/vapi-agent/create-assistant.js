// c:\Users\barto\bilingual-voice-agent\vapi-agent\create-assistant.js

require('dotenv').config();
const axios = require('axios');

const VAPI_SECRET_KEY = process.env.VAPI_SECRET_KEY;

// --- IMPORTANT: YOUR PUBLIC NGROK URLS ---
// You will get these URLs after running the servers and ngrok.
// Note that the transcriber now uses port 8000.
const TRANSCRIBER_NGROK_URL = 'https://778443757bf2.ngrok-free.app'; // The URL for port 8000
const VAPI_AGENT_NGROK_URL = 'https://54a00e72b5dc.ngrok-free.app';  // The URL for port 4000

// --- Validation ---
if (!VAPI_SECRET_KEY || VAPI_SECRET_KEY.includes('REPLACE')) {
  console.error("FATAL ERROR: VAPI_SECRET_KEY is not set in the .env file.");
  process.exit(1);
}
if (TRANSCRIBER_NGROK_URL.includes('REPLACE') || VAPI_AGENT_NGROK_URL.includes('REPLACE')) {
    console.error("FATAL ERROR: You must replace the placeholder ngrok URLs in this script.");
    process.exit(1);
}

// --- Assistant Configuration ---
const assistantConfig = {
  name: "Bilingual Croatian Agent",
  transcriber: {
    provider: "custom-transcriber",
    server: {
      url: `${TRANSCRIBER_NGROK_URL}/transcribe`,
    },
  },
  model: {
    provider: "custom-llm",
    model: "custom-llm-model",
    url: `${VAPI_AGENT_NGROK_URL}/api/vapi`,
  },
  voice: {
    provider: "11labs",
    voiceId: "vFQACl5nAIV0owAavYxE", // Using a standard, reliable voice
  },
  serverUrl: `${VAPI_AGENT_NGROK_URL}/api/vapi`,
};

// --- API Call to Create/Update the Assistant ---
const createOrUpdateAssistant = async () => {
  try {
    const { data: assistants } = await axios.get('https://api.vapi.ai/assistant', {
      headers: { 'Authorization': `Bearer ${VAPI_SECRET_KEY}` },
    });
    const existingAssistant = assistants.find(a => a.name === assistantConfig.name);
    
    let updatedAssistant;
    if (existingAssistant) {
      console.log(`Updating existing assistant (ID: ${existingAssistant.id})...`);
      const response = await axios.patch(`https://api.vapi.ai/assistant/${existingAssistant.id}`, assistantConfig, { headers: { 'Authorization': `Bearer ${VAPI_SECRET_KEY}` } });
      updatedAssistant = response.data;
    } else {
      console.log('Creating new assistant...');
      const response = await axios.post('https://api.vapi.ai/assistant', assistantConfig, { headers: { 'Authorization': `Bearer ${VAPI_SECRET_KEY}` } });
      updatedAssistant = response.data;
    }
    console.log('✅ Assistant configured successfully in Vapi!');
    console.log(`➡️ Assistant ID: ${updatedAssistant.id}`);
    console.log(`➡️ Use this ID to make a test call.`);
  } catch (error) {
    console.error('❌ Error configuring assistant:', error.response ? error.response.data : error.message);
  }
};

createOrUpdateAssistant();
