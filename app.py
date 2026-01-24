import streamlit as st
import torch
import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import soundfile as sf

st.set_page_config(
    page_title="Offline Multilingual Translator",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Offline Multilingual Translator")
st.caption("No API • Local model • Streamlit-safe")

MODEL_NAME = "facebook/nllb-200-distilled-600M"

LANG_MAP = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Marathi": "mar_Deva",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
}

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

text = st.text_area("Enter text (English)", height=120)
target = st.selectbox("Translate to", LANG_MAP.keys())

if st.button("Translate"):
    if not text.strip():
        st.warning("Enter text")
        st.stop()

    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[LANG_MAP[target]]
        )

    translated = tokenizer.decode(output[0], skip_special_tokens=True)

    st.success("Translation done")
    st.text_area("Translated Text", translated, height=120)
