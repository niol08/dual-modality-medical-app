import requests
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

if os.getenv("GEMINI_API_KEY"):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
else:
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        GEMINI_API_KEY = ""

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def query_gemini_flash(modality: str, label: str, confidence: float) -> str:
    if not GEMINI_API_KEY:
        return "Gemini API key missing – no explanation generated."

    prompt = (
        f"Explain the meaning of a {modality} signal classified as '{label}' "
        f"with a confidence of {confidence:.1%} in a medical diagnostic context."
    )

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY,
    }

    try:
        resp = requests.post(GEMINI_ENDPOINT, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Gemini API error: {str(e)}"
