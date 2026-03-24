from groq import Groq
import os

class TranscriptionService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None

    def transcribe(self, audio_path: str, api_key: str = None):
        client = self.client
        if api_key:
            client = Groq(api_key=api_key)
            
        if not client:
            raise ValueError("Groq API Key missing. Please provide it in the settings.")
            
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} not found")
        
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
            
        return transcription.text

transcription_service = TranscriptionService()
