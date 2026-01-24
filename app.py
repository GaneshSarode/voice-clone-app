import streamlit as st
import os
import google.generativeai as genai

# ---------- CONFIG ----------
st.set_page_config(page_title="Multilingual Voice Translator", layout="centered")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY not set in Streamlit Secrets")
    st.stop()

genai.configure(api_key=API_KEY)

# IMPORTANT: correct model
model = genai.GenerativeModel("gemini-1.0-pro")

# ---------- UI ----------
st.title("🌍 Multilingual Voice Translator")
st.caption("Speech → Meaningful Translation (Text only for now)")

text = st.text_area("Enter text to translate", height=150)

target_lang = st.selectbox(
    "Translate to",
    ["English", "Hindi", "Marathi"]
)

if st.button("Translate"):
    if not text.strip():
        st.warning("Enter some text")
    else:
        prompt = f"Translate this into {target_lang}:\n\n{text}"
        response = model.generate_content(
            prompt,
            request_options={"timeout": 20}
        )
        st.success("Translated Text")
        st.write(response.text)
