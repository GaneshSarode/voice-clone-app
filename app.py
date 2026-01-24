import streamlit as st
import os
import tempfile
from gtts import gTTS
import google.generativeai as genai

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Voice Translator",
    page_icon="🌍",
    layout="centered"
)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY is not set")
    st.stop()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.0-pro")

# ---------------- UI ----------------
st.markdown(
    """
    <h1 style="text-align:center;">🌍 Voice Translator</h1>
    <p style="text-align:center;color:gray;">
        Speak → Translate → Listen (Stable Version)
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# Input text (voice-to-text can be added later)
input_text = st.text_area(
    "🗣️ Enter text in English",
    placeholder="Example: Hello, how are you?",
    height=120
)

target_lang = st.selectbox(
    "🌐 Translate to",
    ["Hindi", "Marathi", "French"]
)

translate_btn = st.button("🚀 Translate & Speak", use_container_width=True)

st.divider()

# ---------------- LOGIC ----------------
if translate_btn:
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Translating..."):
            try:
                prompt = f"Translate the following English text to {target_lang}:\n\n{input_text}"
                response = model.generate_content(prompt)

                translated_text = response.text.strip()

                st.success("✅ Translation Successful")
                st.text_area(
                    "📘 Translated Text",
                    value=translated_text,
                    height=120
                )

                # -------- TEXT TO SPEECH --------
                lang_map = {
                    "Hindi": "hi",
                    "Marathi": "mr",
                    "French": "fr"
                }

                tts = gTTS(
                    text=translated_text,
                    lang=lang_map[target_lang]
                )

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
                    tts.save(audio_file.name)
                    st.audio(audio_file.name, format="audio/mp3")

            except Exception as e:
                st.error("❌ Something went wrong")
                st.code(str(e))
