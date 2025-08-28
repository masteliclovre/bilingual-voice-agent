// c:\Users\barto\bilingual-voice-agent\vapi-agent\index.js

// Load environment variables from .env file
require('dotenv').config();
const express = require('express');
const { default: Vapi } = require('@vapi-ai/web');
const OpenAI = require('openai');

// --- Basic Setup ---
const app = express();
app.use(express.json()); // Middleware to parse JSON bodies

const port = 4000; // The port our local server will run on

// Check for the Vapi and OpenAI secret keys
const VAPI_SECRET_KEY = process.env.VAPI_SECRET_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

if (!VAPI_SECRET_KEY || !OPENAI_API_KEY) {
  console.error("FATAL ERROR: VAPI_SECRET_KEY or OPENAI_API_KEY is not set in the .env file.");
  process.exit(1);
}

// Initialize the Vapi SDK
const vapi = new Vapi(VAPI_SECRET_KEY);
// Initialize the OpenAI client
const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
});

// --- Vapi Webhook Endpoints ---

// Vapi sends a GET request to this endpoint to check if the server is live.
app.get('/api/vapi', (req, res) => {
  console.log('✅ Received Vapi GET health check. Responding with 200 OK.');
  res.sendStatus(200);
});

// Vapi sends all call-related events to this POST endpoint.
app.post('/api/vapi', async (req, res) => {
  const { message } = req.body;

  // Immediately acknowledge the request so Vapi doesn't time out.
  res.sendStatus(200);

  if (!message) {
    return;
  }

  // This is where we act as the "Custom LLM".
  if (message.type === 'function-call' && message.functionCall.name === 'llm') {
    const conversation = message.functionCall.parameters.messages;

    console.log(`🤖 Vapi wants a response. Sending conversation to OpenAI...`);

    try {
      const completion = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          { role: "system", content: "You are a helpful assistant for a company in Croatia. Be concise and professional." },
          ...conversation,
        ],
      });

      const assistantResponse = completion.choices[0].message.content;

      await vapi.call.send({
        callId: message.call.id,
        message: { type: 'add-message', role: 'assistant', content: assistantResponse },
      });
      console.log(`💬 Sent response: "${assistantResponse}"`);
    } catch (error) {
      console.error('❌ Error communicating with OpenAI or Vapi:', error);
    }
  }
});

// --- Start the Server ---
app.listen(port, () => {
  console.log(`✅ Vapi Agent server listening on http://localhost:${port}`);
});
