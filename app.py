import streamlit as st
import google.generativeai as genai
import os

# ---------------- CONFIG ----------------
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("GEMINI_API_KEY is not set")
    st.stop()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------- UI ----------------
st.set_page_config(page_title="Multilingual Voice Translator", layout="centered")
st.title("Multilingual Voice Translator")

st.markdown("### Step 1: Enter text (voice later)")

input_text = st.text_area(
    "Text to translate",
    placeholder="Type something here...",
    height=120
)

target_lang = st.selectbox(
    "Translate to",
    ["English", "Hindi", "Marathi"]
)

if st.button("Translate"):
    if not input_text.strip():
        st.warning("Please enter some text")
    else:
        prompt = f"Translate the following text to {target_lang}:\n\n{input_text}"
        response = model.generate_content(prompt)

        st.markdown("### Translated Text")
        st.success(response.text)
