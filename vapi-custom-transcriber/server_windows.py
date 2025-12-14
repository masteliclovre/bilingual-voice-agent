#!/usr/bin/env python3
"""
Vapi Custom Transcriber using Croatian Whisper Model
Windows-compatible version - uses direct model inference instead of pipeline
"""

import asyncio
import websockets
import json
import logging
import os
import numpy as np
from dotenv import load_dotenv
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PORT = int(os.getenv("PORT", 8765))
MODEL_ID = os.getenv("MODEL_ID", "GoranS/whisper-large-v3-turbo-hr-parla")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Audio configuration from Vapi
SAMPLE_RATE = 16000
CHANNELS = 2
BYTES_PER_SAMPLE = 2  # 16-bit audio

logger.info(f"Using device: {DEVICE}")
logger.info(f"Loading model: {MODEL_ID}")


class WhisperTranscriber:
    """Handles audio transcription using Hugging Face Whisper model"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.load_model()
        
    def load_model(self):
        """Load the Croatian Whisper model"""
        try:
            logger.info("Loading Whisper model...")
            
            # Load model
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            self.model.to(DEVICE)
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(MODEL_ID)
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def transcribe_audio(self, audio_data: bytes, channel: int) -> str:
        """
        Transcribe audio data for a specific channel using direct model inference
        
        Args:
            audio_data: Raw PCM audio bytes
            channel: 0 for customer, 1 for assistant
            
        Returns:
            Transcribed text
        """
        try:
            # Convert PCM bytes to numpy array
            audio_array = self.pcm_to_numpy(audio_data, channel)
            
            if len(audio_array) == 0:
                return ""
            
            # Process audio with feature extractor
            inputs = self.processor(
                audio_array,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            )
            inputs = inputs.to(DEVICE)
            
            # Generate transcription using model.generate()
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs["input_features"],
                    max_new_tokens=128
                )
            
            # Decode
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0].strip()
            
            logger.info(f"Channel {channel} transcription: {transcription}")
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    def pcm_to_numpy(self, pcm_data: bytes, channel: int) -> np.ndarray:
        """
        Convert PCM bytes to numpy array and extract specific channel
        
        Args:
            pcm_data: Raw PCM audio bytes (stereo, 16-bit)
            channel: 0 for left (customer), 1 for right (assistant)
            
        Returns:
            Mono audio as numpy array
        """
        try:
            # Ensure data length is valid (multiple of 4 bytes for stereo 16-bit)
            bytes_per_frame = CHANNELS * BYTES_PER_SAMPLE
            if len(pcm_data) % bytes_per_frame != 0:
                # Trim to valid length
                valid_length = (len(pcm_data) // bytes_per_frame) * bytes_per_frame
                pcm_data = pcm_data[:valid_length]
            
            if len(pcm_data) == 0:
                return np.array([], dtype=np.float32)
            
            # Convert bytes to numpy array of int16
            audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
            
            # Reshape to separate channels (interleaved stereo)
            audio_stereo = audio_int16.reshape(-1, CHANNELS)
            
            # Extract the specific channel
            audio_mono = audio_stereo[:, channel]
            
            # Convert to float32 and normalize to [-1.0, 1.0]
            audio_float = audio_mono.astype(np.float32) / 32768.0
            
            return audio_float
            
        except Exception as e:
            logger.error(f"Error converting PCM to numpy: {e}")
            return np.array([], dtype=np.float32)


class AudioBuffer:
    """Buffers audio data for each channel before transcription"""
    
    def __init__(self, min_duration_seconds=2.0):
        self.customer_buffer = bytearray()
        self.assistant_buffer = bytearray()
        self.min_samples = int(SAMPLE_RATE * min_duration_seconds)
        self.min_bytes = self.min_samples * BYTES_PER_SAMPLE
        
    def add_audio(self, data: bytes):
        """Add audio data to buffers"""
        # Audio is interleaved stereo, we keep both channels
        self.customer_buffer.extend(data)
        self.assistant_buffer.extend(data)
    
    def should_transcribe(self, channel: int) -> bool:
        """Check if we have enough audio to transcribe"""
        buffer = self.customer_buffer if channel == 0 else self.assistant_buffer
        return len(buffer) >= self.min_bytes
    
    def get_and_clear(self, channel: int) -> bytes:
        """Get buffer contents and clear it"""
        if channel == 0:
            data = bytes(self.customer_buffer)
            self.customer_buffer.clear()
        else:
            data = bytes(self.assistant_buffer)
            self.assistant_buffer.clear()
        return data
    
    def clear_all(self):
        """Clear all buffers"""
        self.customer_buffer.clear()
        self.assistant_buffer.clear()


async def handle_vapi_connection(websocket, path):
    """
    Handle WebSocket connection from Vapi
    
    Args:
        websocket: WebSocket connection
        path: WebSocket path
    """
    logger.info(f"New connection from Vapi on path: {path}")
    
    # Initialize transcriber and buffer
    transcriber = WhisperTranscriber()
    buffer = AudioBuffer(min_duration_seconds=1.5)
    
    try:
        async for message in websocket:
            # Handle text messages (JSON)
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    
                    if data.get("type") == "start":
                        logger.info(f"Received start message: {data}")
                        logger.info(f"Audio config - Sample rate: {data.get('sampleRate')}, Channels: {data.get('channels')}")
                        
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON message")
                    
            # Handle binary messages (audio data)
            elif isinstance(message, bytes):
                # Add to buffer
                buffer.add_audio(message)
                
                # Check if we should transcribe each channel
                for channel in [0, 1]:
                    if buffer.should_transcribe(channel):
                        audio_data = buffer.get_and_clear(channel)
                        
                        # Transcribe in background
                        transcription = transcriber.transcribe_audio(audio_data, channel)
                        
                        if transcription:
                            # Send response back to Vapi
                            channel_name = "customer" if channel == 0 else "assistant"
                            response = {
                                "type": "transcriber-response",
                                "transcription": transcription,
                                "channel": channel_name
                            }
                            
                            await websocket.send(json.dumps(response))
                            logger.info(f"Sent transcription for {channel_name}: {transcription}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info("Connection closed by Vapi")
    except Exception as e:
        logger.error(f"Error handling connection: {e}", exc_info=True)
    finally:
        # Clear buffers
        buffer.clear_all()
        logger.info("Connection handler finished")


async def main():
    """Start the WebSocket server"""
    logger.info(f"Starting Custom Transcriber WebSocket server on port {PORT}")
    
    # Start WebSocket server
    async with websockets.serve(
        handle_vapi_connection,
        "0.0.0.0",
        PORT,
        ping_interval=20,
        ping_timeout=20
    ):
        logger.info(f"Server is listening on ws://0.0.0.0:{PORT}/api/custom-transcriber")
        logger.info("Waiting for connections from Vapi...")
        
        # Keep server running
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
