import base64
import json
import requests

# Read audio and encode to Base64
with open("audio.wav", "rb") as audio_file:
    audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")

# Prepare the request
payload = {
    "callId": "test-1",
    "audio": audio_base64
}

# Send POST request
response = requests.post("http://localhost:8000/transcribe",
                         headers={"Content-Type": "application/json"},
                         data=json.dumps(payload))

print("Response:", response.json())
