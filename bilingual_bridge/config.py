import os
from dotenv import load_dotenv


load_dotenv()


class Config:
    # Twilio validation is optional for dev (you can disable)
    TWILIO_WS_AUTH_TOKEN = os.getenv("TWILIO_WS_AUTH_TOKEN")


    # Deepgram
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    DG_EN_MODEL = os.getenv("DEEPGRAM_EN_MODEL", "nova-3")


    # Google Cloud
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    GCP_LOCATION = os.getenv("GCP_LOCATION", "europe-west4")


    # Azure TTS
    AZURE_TTS_KEY = os.getenv("AZURE_TTS_KEY")
    AZURE_TTS_REGION = os.getenv("AZURE_TTS_REGION", "westeurope")
    AZURE_TTS_EN_VOICE = os.getenv("AZURE_TTS_EN_VOICE", "en-GB-RyanNeural")
    AZURE_TTS_HR_VOICE = os.getenv("AZURE_TTS_HR_VOICE", "hr-HR-GabrijelaNeural")


    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")