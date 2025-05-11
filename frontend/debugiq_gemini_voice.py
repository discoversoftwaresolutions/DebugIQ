from pathlib import Path

# Clean and working debugiq_gemini_voice.py file
gemini_voice_code = '''
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------
# 🎙️ Gemini Voice Processing
# -----------------------------
def process_voice_file(audio_bytes: bytes) -> dict:
    """
    Process a .wav file using Gemini:
    1. Transcribes voice to text
    2. Sends text to Gemini for reasoning
    """
    try:
        transcript = transcribe_audio_with_gemini(audio_bytes)
        if not transcript:
            return {"error": "Transcription failed."}
        return query_gemini(transcript)
    except Exception as e:
        return {"error": str(e)}


def transcribe_audio_with_gemini(audio_bytes: bytes) -> str:
    """
    Uses Gemini-compatible endpoint (replace this stub with a working transcription endpoint).
    """
    try:
        headers = {
            "Authorization": f"Bearer {GEMINI_API_KEY}",
            "Content-Type": "application/octet-stream"
        }

        # Replace with your own Gemini-compatible endpoint
        response = requests.post(
            "https://your-transcription-endpoint.com/v1/audio:transcribe",
            headers=headers,
            data=audio_bytes
        )
        transcript = response.json().get("transcript", "")
        return transcript

    except Exception as e:
        return f"[Gemini transcription error] {str(e)}"

# -----------------------------
# ✍️ GPT-4o Text Prompt Processing
# -----------------------------
def process_text_command(prompt: str) -> dict:
    return query_openai(prompt)

def query_openai(prompt: str) -> dict:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        reply = response.choices[0].message.content
        return {"model": "gpt-4o", "response": reply}
    except Exception as e:
        return {"error": f"OpenAI error: {str(e)}"}

# -----------------------------
# 🧠 Gemini Reasoning from Transcript
# -----------------------------
def query_gemini(prompt: str) -> dict:
    try:
        headers = {
            "Authorization": f"Bearer {GEMINI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7}
        }

        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
            headers=headers,
            params={"key": GEMINI_API_KEY},
            json=payload
        )
        result = response.json()
        reply = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return {"model": "gemini-pro", "response": reply, "input": prompt}
    except Exception as e:
        return {"error": f"Gemini error: {str(e)}"}
