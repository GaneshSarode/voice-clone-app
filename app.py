import streamlit as st
import librosa
import tempfile
from transformers import pipeline
from TTS.api import TTS

from ui import render_header, render_sidebar ,render_status

st.set_page_config(page_title="Voice Clone Translator", layout="wide")
render_header()
render_sidebar()
render_status()
st.title("🎙️ Voice Cloning Translator (English → Hindi / French / Japanese)")

# -------- Load models --------
@st.cache_resource
def load_asr():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=-1
    )

@st.cache_resource
def load_translator(model_name, target_lang):
    if model_name.startswith("facebook/m2m100"):
        return pipeline(
            "translation",
            model=model_name,
            src_lang="en",
            tgt_lang=target_lang,
            device=-1
        )
    else:
        return pipeline(
            "translation",
            model=model_name,
            device=-1
        )


@st.cache_resource
def load_xtts():
    return TTS(
        "tts_models/multilingual/multi-dataset/xtts_v2",
        gpu=False
    )

asr = load_asr()
xtts = load_xtts()

# -------- Language config --------
LANGS = {
    "Hindi": {
        "translator": "Helsinki-NLP/opus-mt-en-hi",
        "code": "hi",
        "file": "hindi_my_voice.wav"
    },
    "French": {
        "translator": "Helsinki-NLP/opus-mt-en-fr",
        "code": "fr",
        "file": "french_my_voice.wav"
    },
    "Japanese": {
        "translator": "facebook/m2m100_418M",
        "code": "ja",
        "file": "japanese_my_voice.wav"
    }
}

# -------- UI --------
target_lang = st.selectbox("Select Target Language", list(LANGS.keys()))
uploaded = st.file_uploader("Upload English voice (WAV)", type=["wav"])
text_input = st.text_area("Or type English text")
convert = st.button("Convert to Voice")
tab1, tab2, tab3 = st.tabs(["📝 Text", "🌍 Translation", "🔊 Voice"])
# -------- Processing --------
if convert:
    if not uploaded and not text_input.strip():
        st.warning("Upload audio or type text.")
    else:
        with st.spinner("Processing (CPU – slow but working)..."):

            # -------- Handle uploaded audio --------
            # -------- Get English text --------
            if uploaded:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded.read())
                    speaker_path = tmp.name

                audio, sr = librosa.load(speaker_path, sr=16000)
                english_text = asr(audio)["text"]

            elif text_input.strip():
                st.warning("⚠️ Upload a voice sample to clone your voice.")
                st.stop()

            else:
                st.warning("Provide text or upload audio.")
                st.stop()


            with tab1:
                st.subheader("Recognized English")
                st.write(english_text)

            # -------- Translation --------
            translator = load_translator(
    LANGS[target_lang]["translator"],
    LANGS[target_lang]["code"]
)

            translated_text = translator(english_text)[0]["translation_text"]

            with tab2:
                st.subheader(f"{target_lang} Text")
                st.write(translated_text)

            # -------- XTTS (Real Voice Cloning) --------
            out_path = "out.wav"
            xtts.tts_to_file(
                text=translated_text,
                speaker_wav=speaker_path,
                language=LANGS[target_lang]["code"],
                file_path=out_path,
                split_sentences=False
            )

            with tab3:
                st.subheader(f"{target_lang} Voice (Your Voice)")
                st.audio(out_path)
                st.download_button(
                    "⬇ Download Audio",
                    open(out_path, "rb"),
                    file_name=LANGS[target_lang]["file"]
                )

