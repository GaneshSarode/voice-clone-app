import streamlit as st

import tempfile
import soundfile as sf
import os
import google.generativeai as genai
# -------------------- GEMINI CONFIG --------------------

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

st.title("Multilingual Voice Translator")

audio_file = st.audio_input("Record your voice")

target_lang = st.selectbox(
    "Translate to",
    ["English", "Hindi", "Marathi"]
)

if audio_file and st.button("Translate & Speak"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_file.read())
        audio_path = f.name

    # 1️⃣ Speech → Text
    with open(audio_path, "rb") as f:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )

    text = transcript.text
    st.text_area("Transcribed Text", text)

    # 2️⃣ Translate
    prompt = f"Translate this to {target_lang}: {text}"
    translation = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    translated_text = translation.choices[0].message.content
    st.text_area("Translated Text", translated_text)

    # 3️⃣ Text → Speech
    speech = openai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=translated_text
    )

    audio_bytes = speech.read()
    st.audio(audio_bytes, format="audio/wav")
