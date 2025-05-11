import os
import openai
import requests
import base64

# ✅ Load API keys from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Configure OpenAI
openai.api_key = OPENAI_API_KEY

# -----------------------------
# 🔊 Voice Processing
# -----------------------------
def process_voice_file(audio_bytes: bytes) -> dict:
    try:
        # Step 1: Transcribe using Whisper
        transcript = transcribe_audio_with_openai(audio_bytes)

        if not transcript:
            return {"error": "Transcription failed."}

        # Step 2: Send to Gemini or fallback model
        return process_text_command(transcript, fallback_model="gemini-pro")

    except Exception as e:
        return {"error": str(e)}


def transcribe_audio_with_openai(audio_bytes: bytes) -> str:
    try:
        response = openai.Audio.transcribe("whisper-1", file=audio_bytes)
        return response.get("text", "")
    except Exception as e:
        return f"[Transcription Error] {str(e)}"


# -----------------------------
# 🧠 Text Command Processing
# -----------------------------
def process_text_command(text: str, fallback_model: str = "gpt-4o") -> dict:
    """
    Process a command using Gemini API, fallback to OpenAI GPT-4o.
    """
    try:
        if fallback_model == "gemini-pro":
            return query_gemini(text)
        else:
            return query_openai(text)
    except Exception as e:
        return {"error": str(e)}


def query_gemini(prompt: str) -> dict:
    # Replace with actual Gemini API call once endpoint is stable
    try:
        # This is a placeholder Gemini call structure
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
        return {"model": "gemini-pro", "response": reply}
    except Exception as e:
        return {"error": f"Gemini error: {str(e)}"}


def query_openai(prompt: str) -> dict:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        reply = response["choices"][0]["message"]["content"]
        return {"model": "gpt-4o", "response": reply}
    except Exception as e:
        return {"error": f"OpenAI error: {str(e)}"}
