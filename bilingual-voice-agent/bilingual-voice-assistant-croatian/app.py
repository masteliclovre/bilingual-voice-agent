import os
import base64
import tempfile
from flask import Flask, request, jsonify
from transformers import pipeline

# --- Model Loading ---
# This happens once when the server starts for maximum performance.
MODEL_PATH = "./models/"
DEVICE = "cpu"

print(f"--- Loading ASR pipeline with CTranslate2 model from {MODEL_PATH} ---")
try:
    # Use the high-level pipeline API from transformers.
    # With 'optimum' installed, it automatically detects and uses the fast CTranslate2 model.
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_PATH,
        device=DEVICE
    )
    print("--- Pipeline loaded successfully ---")
except Exception as e:
    print(f"!!! FATAL ERROR: Could not load pipeline. !!!")
    print(f"Error: {e}")
    exit(1)

# --- Flask Application ---
app = Flask(__name__)

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

# --- Transcription Endpoint (Vapi Compatible) ---
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    data = request.get_json()
    if not data or 'audio' not in data:
        return jsonify({"error": "Missing 'audio' in request body"}), 400

    call_id = data.get('callId', 'unknown-call')
    print(f"[{call_id}] Received transcription request.")

    temp_audio_path = None
    try:
        audio_data = base64.b64decode(data['audio'])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_audio_file.write(audio_data)
            temp_audio_path = temp_audio_file.name

        # Transcribe using the pipeline. We must specify the language for generation.
        # The pipeline handles all the complex pre- and post-processing.
        result = pipe(temp_audio_path, generate_kwargs={"language": "croatian"})
        text = result["text"].strip()

        print(f"[{call_id}] Transcription successful. Transcript: \"{text}\"")

        return jsonify({"text": text})

    except Exception as e:
        print(f"[{call_id}] !!! ERROR during transcription: {e} !!!")
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

