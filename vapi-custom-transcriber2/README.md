# Vapi Custom Transcriber - Croatian Whisper Model

A custom transcriber for Vapi that uses the Croatian Whisper model (`lovremastelic/bva`) for real-time speech-to-text transcription.

## Features

- ✅ Real-time audio transcription using Croatian Whisper model
- ✅ Dual-channel support (customer + assistant)
- ✅ WebSocket-based communication with Vapi
- ✅ Automatic audio buffering and processing
- ✅ GPU acceleration support (CUDA)
- ✅ Production-ready error handling

## Model Information

This transcriber uses:
- **Primary Model**: `lovremastelic/bva`
- **Alternative**: `GoranS/whisper-large-v3-turbo-hr-parla`

Both are fine-tuned Croatian versions of OpenAI's Whisper Large V3 Turbo model, achieving excellent WER (Word Error Rate) for Croatian language transcription.

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, but CPU also works)
- Minimum 4GB RAM (8GB+ recommended)
- Internet connection for initial model download

## Installation

### 1. Clone or create project directory

```bash
mkdir vapi-croatian-transcriber
cd vapi-croatian-transcriber
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv

# Activate on Linux/Mac:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you have CUDA installed, make sure to install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Configure environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings
nano .env  # or use your preferred editor
```

**`.env` configuration:**
```bash
PORT=8765
MODEL_ID=lovremastelic/bva
```

## Usage

### 1. Start the server

```bash
python server.py
```

You should see:
```
INFO - Using device: cuda  # or 'cpu' if no GPU
INFO - Loading model: lovremastelic/bva
INFO - Loading Whisper model...
INFO - Model loaded successfully!
INFO - Starting Custom Transcriber WebSocket server on port 8765
INFO - Server is listening on ws://0.0.0.0:8765/api/custom-transcriber
INFO - Waiting for connections from Vapi...
```

### 2. Expose your server (for testing)

Use a tunneling service to make your local server accessible:

**Using ngrok:**
```bash
ngrok http 8765
```

This will give you a public URL like: `https://abc123.ngrok.io`

Your WebSocket URL will be: `wss://abc123.ngrok.io/api/custom-transcriber`

### 3. Test with Vapi

Create a test call using the Vapi API:

```bash
curl -X POST https://api.vapi.ai/call \
     -H "Authorization: Bearer YOUR_VAPI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
  "phoneNumberId": "YOUR_PHONE_NUMBER_ID",
  "customer": {
    "number": "+385XXXXXXXXX"
  },
  "assistant": {
    "transcriber": {
      "provider": "custom-transcriber",
      "server": {
        "url": "wss://abc123.ngrok.io/api/custom-transcriber"
      }
    },
    "model": {
      "provider": "openai",
      "model": "gpt-4",
      "messages": [{
        "role": "system",
        "content": "You are a helpful assistant that speaks Croatian."
      }]
    },
    "voice": {
      "provider": "11labs",
      "voiceId": "21m00Tcm4TlvDq8ikWAM"
    },
    "firstMessage": "Bok! Kako vam mogu pomoći?"
  },
  "name": "Croatian Transcriber Test"
}'
```

## How It Works

### 1. Connection Flow

```
Vapi → WebSocket Connection → Your Server
                ↓
         Audio Streaming (PCM)
                ↓
         Channel Separation
                ↓
    Customer (Ch 0)  |  Assistant (Ch 1)
                ↓
         Audio Buffering
                ↓
         Whisper Model
                ↓
         Transcription
                ↓
         Response to Vapi
```

### 2. Audio Processing

- **Input Format**: Stereo PCM, 16kHz, 16-bit
- **Channel 0**: Customer audio
- **Channel 1**: Assistant audio
- **Buffering**: 1.5 seconds minimum before transcription
- **Processing**: Real-time with automatic chunking

### 3. Response Format

The server sends transcriptions back to Vapi as JSON:

```json
{
  "type": "transcriber-response",
  "transcription": "Transcribed text in Croatian",
  "channel": "customer"  // or "assistant"
}
```

## Configuration Options

### Model Selection

You can use either model by changing the `MODEL_ID` in `.env`:

```bash
# Option 1: lovremastelic/bva (recommended)
MODEL_ID=lovremastelic/bva

# Option 2: GoranS/whisper-large-v3-turbo-hr-parla
MODEL_ID=GoranS/whisper-large-v3-turbo-hr-parla
```

### Buffer Duration

To adjust transcription latency, modify `AudioBuffer` initialization in `server.py`:

```python
# More frequent transcriptions (lower latency, higher cost)
buffer = AudioBuffer(min_duration_seconds=1.0)

# Less frequent transcriptions (higher latency, lower cost)
buffer = AudioBuffer(min_duration_seconds=3.0)
```

### Batch Size

Adjust batch size for better GPU utilization in `server.py`:

```python
self.pipe = pipeline(
    ...
    batch_size=8,  # Increase for better GPU usage (if you have memory)
    ...
)
```

## Deployment

### Docker Deployment (Recommended)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY server.py .
COPY .env .

# Expose port
EXPOSE 8765

# Run server
CMD ["python", "server.py"]
```

Build and run:
```bash
docker build -t vapi-croatian-transcriber .
docker run -p 8765:8765 --gpus all vapi-croatian-transcriber
```

### Production Deployment

For production, consider:

1. **Use a process manager**: PM2, systemd, or supervisord
2. **Set up SSL/TLS**: Use a reverse proxy (nginx) with Let's Encrypt
3. **Scale horizontally**: Deploy multiple instances behind a load balancer
4. **Monitor performance**: Set up logging and metrics
5. **Use a dedicated GPU**: For optimal performance

Example with systemd:

```ini
# /etc/systemd/system/vapi-transcriber.service
[Unit]
Description=Vapi Croatian Transcriber
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/vapi-croatian-transcriber
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## Troubleshooting

### Model Download Issues

If the model fails to download:
```bash
# Pre-download the model
python -c "from transformers import AutoModelForSpeechSeq2Seq; AutoModelForSpeechSeq2Seq.from_pretrained('lovremastelic/bva')"
```

### CUDA Out of Memory

If you get CUDA OOM errors:
1. Reduce `batch_size` in the pipeline
2. Use `torch.float32` instead of `torch.float16`
3. Reduce `chunk_length_s` to 15 or 20 seconds

### Connection Issues

If Vapi can't connect:
1. Check firewall rules
2. Verify WebSocket URL (must use `wss://` for HTTPS)
3. Check server logs for errors
4. Ensure ngrok or your tunnel is running

### Poor Transcription Quality

If transcriptions are inaccurate:
1. Verify audio quality and sample rate
2. Ensure correct channel separation
3. Try increasing buffer duration
4. Check if Croatian language is being detected correctly

## Performance Notes

- **CPU Mode**: ~2-5x slower than real-time
- **GPU Mode (CUDA)**: ~0.5-1x real-time (faster than audio playback)
- **Memory Usage**: 2-4GB RAM + 2-6GB VRAM (GPU mode)
- **First transcription**: May take longer due to model warm-up

## Contributing

Feel free to submit issues or pull requests for improvements!

## License

This implementation is provided as-is. The underlying model (`lovremastelic/bva`) is licensed under Apache 2.0.

## Credits

- Model: [lovremastelic/bva](https://huggingface.co/lovremastelic/bva)
- Alternative: [GoranS/whisper-large-v3-turbo-hr-parla](https://huggingface.co/GoranS/whisper-large-v3-turbo-hr-parla)
- Based on: OpenAI Whisper Large V3 Turbo
- Framework: Hugging Face Transformers
