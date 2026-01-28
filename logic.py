import librosa
import tempfile

def get_english_text(asr, uploaded, typed_text):
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read())
            path = tmp.name
        audio, _ = librosa.load(path, sr=16000)
        text = asr(audio)["text"]
        return text, path
    else:
        return typed_text, "/mnt/c/Users/Ganesh_Ashok_Sarode/Downloads/test.wav"
