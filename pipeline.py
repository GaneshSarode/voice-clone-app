import tempfile
import soundfile as sf
from faster_whisper import WhisperModel

# load once (important)
asr_model = WhisperModel("base", device="cpu")

def speech_to_text(audio_path, source_lang):
    segments, _ = asr_model.transcribe(
        audio_path,
        language=None if source_lang == "auto" else source_lang
    )
    return " ".join(seg.text for seg in segments)


def translate_text(text, target_lang):
    # placeholder (replace later)
    return f"[{target_lang}] {text}"


def voice_clone_tts(text):
    # placeholder (replace with XTTS later)
    # generate silent dummy audio for now
    return [0.0] * 16000


def process_audio(uploaded_file, source_lang, target_lang):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name

    text = speech_to_text(input_path, source_lang)
    translated = translate_text(text, target_lang)
    audio = voice_clone_tts(translated)

    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(out_path, audio, 16000)

    return out_path
