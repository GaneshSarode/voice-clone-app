import librosa
import tempfile


def transcribe_audio(asr, uploaded_file):
    """Save uploaded audio to a temp file, load at 16 kHz, and transcribe."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        speaker_path = tmp.name

    audio, _ = librosa.load(speaker_path, sr=16000)
    text = asr(audio)["text"]
    return text, speaker_path


def translate_text(translator, text):
    """Translate text using the given HuggingFace translator pipeline."""
    result = translator(text)
    return result[0]["translation_text"]
