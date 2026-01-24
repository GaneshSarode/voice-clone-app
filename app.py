import streamlit as st
import google.generativeai as genai
import os
import time

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Multilingual Voice Translator",
    page_icon="🌍",
    layout="centered"
)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY is not set in environment variables")
    st.stop()

genai.configure(api_key=API_KEY)

# Use a stable model
model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------- UI --------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🌍 Multilingual Translator</h1>
    <p style='text-align:center;color:gray;'>
        Fast • Clean • No GPU • No OpenAI
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

input_text = st.text_area(
    "✍️ Enter text",
    placeholder="Type any sentence you want to translate...",
    height=140
)

target_lang = st.selectbox(
    "🌐 Translate to",
    ["English", "Hindi", "Marathi"]
)

translate_btn = st.button("🚀 Translate", use_container_width=True)

st.divider()

# -------------------- LOGIC --------------------
if translate_btn:
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Translating..."):
            try:
                prompt = f"""
Translate the following text to {target_lang}.
Return only the translated text.

Text:
{input_text}
"""
                response = model.generate_content(
                    prompt,
                    request_options={"timeout": 20}
                )

                time.sleep(0.3)  # avoid retry crash

                st.success("✅ Translation successful")
                st.text_area(
                    "📘 Translated Text",
                    value=response.text,
                    height=140
                )

            except Exception as e:
                st.error("❌ Translation failed")
                st.code(str(e))
