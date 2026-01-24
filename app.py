import streamlit as st
import pyttsx3
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Offline Multilingual Voice Translator",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Offline Multilingual Voice Translator")
st.caption("Local model • No API • No Internet after download")

# ---------------- LANGUAGE MAP (NLLB CODES) ----------------
LANG_MAP = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Marathi": "mar_Deva",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Tamil": "tam_Taml",
    "Bengali": "ben_Beng"
}

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# ---------------- LOAD MODEL (ONCE) ----------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- UI ----------------
input_text = st.text_area(
    "🗣️ Enter text (English)",
    placeholder="Hello, how are you?",
    height=120
)

target_lang = st.selectbox(
    "🌐 Translate to",
    list(LANG_MAP.keys())
)

if st.button("🚀 Translate & Speak", use_container_width=True):

    if not input_text.strip():
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Translating (first time may take a while)..."):
        src_lang = "eng_Latn"
        tgt_lang = LANG_MAP[target_lang]

        tokenizer.src_lang = src_lang
        encoded = tokenizer(input_text, return_tensors="pt")
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
        )
        translated_text = tokenizer.decode(
            generated[0],
            skip_special_tokens=True
        )

    st.success("✅ Translation Successful")
    st.text_area(
        "📘 Translated Text",
        value=translated_text,
        height=120
    )

    # ---------------- OFFLINE TEXT TO SPEECH ----------------
    engine = pyttsx3.init()
    engine.say(translated_text)
    engine.runAndWait()
