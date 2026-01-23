import streamlit as st
from pipeline import process_audio

st.set_page_config(page_title="Voice Clone Translator", layout="centered")

st.title("Voice Clone + Translate")

uploaded_audio = st.file_uploader(
    "Upload your voice (wav/mp3)",
    type=["wav", "mp3"]
)

source_lang = st.selectbox(
    "Source Language",
    ["auto", "en", "hi", "fr", "de"]
)

target_lang = st.selectbox(
    "Target Language",
    ["en", "hi", "fr", "de"]
)

if uploaded_audio and st.button("Process"):
    with st.spinner("Processing..."):
        output_audio = process_audio(
            uploaded_audio,
            source_lang,
            target_lang
        )

    st.audio(output_audio, format="audio/wav")
    st.success("Done")
