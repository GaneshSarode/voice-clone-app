import streamlit as st
import torch
import librosa
import soundfile as sf
import os

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer,
    pipeline
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Voice Translator",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ English → Hindi Voice Translator")
st.caption("CPU-based | Hugging Face Models | Offline Cache")

# ---------------- MODEL LOADING ----------------
@st.cache_resource(show_spinner=False)
def load_models():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    tr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    tr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

    tts = pipeline(
        "text-to-speech",
        model="facebook/mms-tts-hin",
        device=-1
    )
    return processor, asr, tr_tokenizer, tr_model, tts


with st.spinner("Loading models (first time only)..."):
    processor, asr_model, tr_tokenizer, tr_model, tts = load_models()

st.success("Models loaded from cache ✔️")

# ---------------- FILE UPLOAD ----------------
audio_file = st.file_uploader(
    "Upload English audio (.wav)",
    type=["wav"]
)

if audio_file:
    st.audio(audio_file)

    if st.button("Translate", use_container_width=True):
        with st.spinner("Processing... please wait"):

            # Load audio
            audio, _ = librosa.load(audio_file, sr=16000)

            # ASR
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                ids = asr_model.generate(inputs.input_features)

            english_text = processor.decode(ids[0], skip_special_tokens=True)

            st.subheader("Recognized English")
            st.write(english_text)

            # Translation
            tokens = tr_tokenizer(english_text, return_tensors="pt", padding=True)
            translated = tr_model.generate(**tokens)
            hindi_text = tr_tokenizer.decode(translated[0], skip_special_tokens=True)

            st.subheader("Translated Hindi")
            st.write(hindi_text)

            # TTS
            speech = tts(hindi_text)
            output_path = "output.wav"
            sf.write(output_path, speech["audio"], speech["sampling_rate"])

            st.subheader("Hindi Voice Output")
            st.audio(output_path)
            st.download_button(
                "Download Audio",
                data=open(output_path, "rb"),
                file_name="translated_hindi.wav"
            )
