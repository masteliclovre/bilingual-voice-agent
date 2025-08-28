import express from "express";
import dotenv from "dotenv";
dotenv.config();

const app = express();
app.use(express.json());

app.get("/health", (_, res) => res.json({ ok: true }));

// Optional endpoints for Vapi session hooks
app.post("/session/start", (req, res) => res.json({ ok: true }));
app.post("/session/stop", (req, res) => res.json({ ok: true }));

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`LLM Agent running on port ${PORT}`));
