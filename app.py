import streamlit as st

from ui import render_header, render_sidebar, render_status
from logic import transcribe_audio, translate_text

# -------- Page config --------
st.set_page_config(page_title="Voice Clone Translator", layout="wide")
render_header()
render_sidebar()
render_status()

st.title("🎙️ Voice Cloning Translator (English → Hindi / French / Japanese)")


# -------- Load models (cached so they load only once) --------
@st.cache_resource
def load_asr():
    from transformers import pipeline
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=-1,
    )


@st.cache_resource
def load_translator(model_name, target_lang):
    from transformers import pipeline
    if model_name.startswith("facebook/m2m100"):
        return pipeline(
            "translation",
            model=model_name,
            src_lang="en",
            tgt_lang=target_lang,
            device=-1,
        )
    return pipeline(
        "translation",
        model=model_name,
        device=-1,
    )


@st.cache_resource
def load_xtts():
    from TTS.api import TTS
    return TTS(
        "tts_models/multilingual/multi-dataset/xtts_v2",
        gpu=False,
    )


# -------- Language config --------
LANGS = {
    "Hindi": {
        "translator": "Helsinki-NLP/opus-mt-en-hi",
        "code": "hi",
        "file": "hindi_my_voice.wav",
    },
    "French": {
        "translator": "Helsinki-NLP/opus-mt-en-fr",
        "code": "fr",
        "file": "french_my_voice.wav",
    },
    "Japanese": {
        "translator": "facebook/m2m100_418M",
        "code": "ja",
        "file": "japanese_my_voice.wav",
    },
}

# -------- UI controls --------
target_lang = st.selectbox("Select Target Language", list(LANGS.keys()))
uploaded = st.file_uploader(
    "Upload English voice (WAV) — used for transcription AND voice cloning",
    type=["wav"],
)
text_input = st.text_area(
    "Or type / paste English text (upload audio above to clone your voice)"
)
convert = st.button("🚀 Convert to Voice")

tab1, tab2, tab3 = st.tabs(["📝 Text", "🌍 Translation", "🔊 Voice"])

# -------- Processing --------
if convert:
    # --- Validate input ---
    has_audio = uploaded is not None
    has_text = bool(text_input.strip())

    if not has_audio and not has_text:
        st.warning("⚠️ Please upload an audio file or type some English text.")
        st.stop()

    with st.spinner("Processing (CPU – this may take a minute)..."):

        # --- Step 1: Get English text ---
        speaker_path = None

        if has_audio:
            asr = load_asr()
            english_text, speaker_path = transcribe_audio(asr, uploaded)
        else:
            english_text = text_input.strip()

        # If user provided BOTH, prefer typed text but still use audio for cloning
        if has_audio and has_text:
            english_text = text_input.strip()

        with tab1:
            st.subheader("📝 English Text")
            st.write(english_text)

        # --- Step 2: Translate ---
        lang_cfg = LANGS[target_lang]
        translator = load_translator(lang_cfg["translator"], lang_cfg["code"])
        translated_text = translate_text(translator, english_text)

        with tab2:
            st.subheader(f"🌍 {target_lang} Translation")
            st.write(translated_text)

        # --- Step 3: Voice cloning (only if we have a voice sample) ---
        with tab3:
            if speaker_path is None:
                st.info(
                    "ℹ️ No audio uploaded — showing translation only. "
                    "Upload a WAV file to generate cloned voice output."
                )
            else:
                xtts = load_xtts()
                out_path = "out.wav"
                xtts.tts_to_file(
                    text=translated_text,
                    speaker_wav=speaker_path,
                    language=lang_cfg["code"],
                    file_path=out_path,
                    split_sentences=False,
                )

                st.subheader(f"🔊 {target_lang} Voice (Your Cloned Voice)")
                st.audio(out_path)
                with open(out_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Audio",
                        f,
                        file_name=lang_cfg["file"],
                    )
