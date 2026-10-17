# logic.py - VoxStudio platform engine: synthesis, voice profiles, ASR,
# translation, audio post-processing and evaluation metrics.
import os
import json
import uuid
import asyncio
import logging
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import librosa
import soundfile
import edge_tts

logger = logging.getLogger("voxstudio")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ---------------------------------------------------------------------------
# Directory layout (all generated data stays out of the source tree)
# ---------------------------------------------------------------------------
CLONED_VOICES_DIR = Path("cloned_voices")
OUTPUT_DIR = Path("outputs")
CLONED_VOICES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_USER = "default"


def _safe_component(value: str, fallback: str = DEFAULT_USER) -> str:
    """Strip a string down to a filesystem-safe path component."""
    cleaned = "".join(c for c in str(value) if c.isalnum() or c in "-_")
    return cleaned or fallback


def safe_filename(name: str, fallback: str = "audio") -> str:
    """Convert a display name into a safe download file name."""
    cleaned = "".join(
        c if (c.isalnum() or c in "-_ ") else "_" for c in str(name)
    ).strip()
    return cleaned.replace(" ", "_") or fallback


def _user_dir(user_id: str = DEFAULT_USER) -> Path:
    """Per-session storage directory so users never see each other's voices."""
    directory = CLONED_VOICES_DIR / _safe_component(user_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _metadata_file(user_id: str = DEFAULT_USER) -> Path:
    return _user_dir(user_id) / "voices_metadata.json"


def new_output_path(prefix: str, extension: str) -> Path:
    """Return a unique path inside the outputs directory."""
    return OUTPUT_DIR / f"{_safe_component(prefix, 'audio')}_{uuid.uuid4().hex[:10]}.{extension.lstrip('.')}"


def cleanup_outputs(max_age_minutes: int = 120) -> int:
    """Delete generated audio older than ``max_age_minutes`` (bounded disk use)."""
    cutoff = datetime.now() - timedelta(minutes=max_age_minutes)
    removed = 0
    for item in OUTPUT_DIR.glob("*"):
        try:
            if item.is_file() and datetime.fromtimestamp(item.stat().st_mtime) < cutoff:
                item.unlink()
                removed += 1
        except OSError as exc:  # pragma: no cover - best effort housekeeping
            logger.debug("Could not remove %s: %s", item, exc)
    if removed:
        logger.info("Cleaned up %d stale output file(s).", removed)
    return removed


# ---------------------------------------------------------------------------
# Built-in voice library
# ---------------------------------------------------------------------------
PRESET_VOICES = {
    "Rachel (Calm Narrator - US)": {
        "voice_id": "rachel",
        "name": "Rachel",
        "gender": "Female",
        "accent": "American",
        "role": "Narration & Audiobooks",
        "desc": "Calm, warm and natural voice perfect for storytelling, long-form content and tutorials.",
        "edge_voice": "en-US-JennyNeural",
        "lang_code": "en",
        "avatar": "👩‍💼",
    },
    "Adam (Deep Narrator - US)": {
        "voice_id": "adam",
        "name": "Adam",
        "gender": "Male",
        "accent": "American",
        "role": "Commercial & Documentary",
        "desc": "Deep, authoritative, and cinematic male voice suitable for voiceovers and trailers.",
        "edge_voice": "en-US-ChristopherNeural",
        "lang_code": "en",
        "avatar": "🎙️",
    },
    "Antoni (Conversational - UK)": {
        "voice_id": "antoni",
        "name": "Antoni",
        "gender": "Male",
        "accent": "British",
        "role": "Podcast & Conversational",
        "desc": "Friendly, approachable British accent with natural inflection and warmth.",
        "edge_voice": "en-GB-RyanNeural",
        "lang_code": "en",
        "avatar": "👨‍💼",
    },
    "Bella (Expressive - UK)": {
        "voice_id": "bella",
        "name": "Bella",
        "gender": "Female",
        "accent": "British",
        "role": "Social Media & Advertising",
        "desc": "Bright, engaging female voice with crisp British diction and high clarity.",
        "edge_voice": "en-GB-SoniaNeural",
        "lang_code": "en",
        "avatar": "✨",
    },
    "Arnav (Executive - Indian English)": {
        "voice_id": "arnav",
        "name": "Arnav",
        "gender": "Male",
        "accent": "Indian English",
        "role": "Corporate & Tech Explainer",
        "desc": "Clear, professional Indian English male voice ideal for corporate presentations.",
        "edge_voice": "en-IN-PrabhatNeural",
        "lang_code": "en",
        "avatar": "👨‍💻",
    },
    "Neerja (Pleasant - Indian English)": {
        "voice_id": "neerja",
        "name": "Neerja",
        "gender": "Female",
        "accent": "Indian English",
        "role": "Customer Experience & Audiobooks",
        "desc": "Pleasant, welcoming Indian English female voice with warm tone.",
        "edge_voice": "en-IN-NeerjaNeural",
        "lang_code": "en",
        "avatar": "👩‍🎓",
    },
    "Kavya (Expressive - Hindi)": {
        "voice_id": "kavya",
        "name": "Kavya",
        "gender": "Female",
        "accent": "Hindi",
        "role": "Vernacular Storytelling",
        "desc": "Expressive native Hindi female speaker with authentic inflection.",
        "edge_voice": "hi-IN-SwaraNeural",
        "lang_code": "hi",
        "avatar": "🌸",
    },
    "Madhur (Authoritative - Hindi)": {
        "voice_id": "madhur",
        "name": "Madhur",
        "gender": "Male",
        "accent": "Hindi",
        "role": "News & Narration",
        "desc": "Deep, confident native Hindi male voice for news, documentaries and ads.",
        "edge_voice": "hi-IN-MadhurNeural",
        "lang_code": "hi",
        "avatar": "📻",
    },
    "Denise (Chic - French)": {
        "voice_id": "denise",
        "name": "Denise",
        "gender": "Female",
        "accent": "French",
        "role": "Luxury & Storytelling",
        "desc": "Sophisticated native Parisian French voice with subtle articulation.",
        "edge_voice": "fr-FR-DeniseNeural",
        "lang_code": "fr",
        "avatar": "🥐",
    },
    "Henri (Cultured - French)": {
        "voice_id": "henri",
        "name": "Henri",
        "gender": "Male",
        "accent": "French",
        "role": "Documentary & Dialogue",
        "desc": "Cultured and balanced French male voice for podcasts and voiceovers.",
        "edge_voice": "fr-FR-HenriNeural",
        "lang_code": "fr",
        "avatar": "🎨",
    },
    "Nanami (Gentle - Japanese)": {
        "voice_id": "nanami",
        "name": "Nanami",
        "gender": "Female",
        "accent": "Japanese",
        "role": "Anime & Virtual Assistant",
        "desc": "Gentle, polished Japanese voice suitable for virtual agents and audiobooks.",
        "edge_voice": "ja-JP-NanamiNeural",
        "lang_code": "ja",
        "avatar": "🎐",
    },
    "Keita (Natural - Japanese)": {
        "voice_id": "keita",
        "name": "Keita",
        "gender": "Male",
        "accent": "Japanese",
        "role": "Gaming & Dialogue",
        "desc": "Natural, clear modern Japanese male voice for games and narration.",
        "edge_voice": "ja-JP-KeitaNeural",
        "lang_code": "ja",
        "avatar": "🎮",
    },
}


# ---------------------------------------------------------------------------
# Voice profile storage (per session)
# ---------------------------------------------------------------------------
def load_cloned_voices_metadata(user_id: str = DEFAULT_USER) -> dict:
    """Load the voice profiles registered by one session."""
    metadata_file = _metadata_file(user_id)
    if not metadata_file.exists():
        return {}
    try:
        with open(metadata_file, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read voice metadata %s: %s", metadata_file, exc)
        return {}


def _write_metadata(metadata: dict, user_id: str = DEFAULT_USER) -> None:
    with open(_metadata_file(user_id), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def save_cloned_voice_file(
    name,
    audio_bytes,
    description="",
    role="Custom Voice",
    gender="Unknown",
    accent="Custom",
    user_id: str = DEFAULT_USER,
):
    """Store a reference recording as a reusable voice profile.

    Note: this registers a *profile* (the reference sample used by the XTTS v2
    engine for true zero-shot cloning when that engine is available). It does not
    itself train or extract a speaker embedding.
    """
    voice_id = f"voice_{uuid.uuid4().hex[:12]}"
    file_path = _user_dir(user_id) / f"{voice_id}.wav"

    with open(file_path, "wb") as handle:
        handle.write(audio_bytes)

    metadata = load_cloned_voices_metadata(user_id)
    metadata[voice_id] = {
        "voice_id": voice_id,
        "name": name,
        "gender": gender,
        "accent": accent,
        "role": role,
        "desc": description or "Voice profile created from an uploaded reference sample.",
        "audio_path": str(file_path),
        "is_cloned": True,
        "avatar": "🧬",
    }
    _write_metadata(metadata, user_id)
    logger.info("Saved voice profile '%s' (%s) for session %s.", name, voice_id, user_id)
    return metadata[voice_id]


def delete_cloned_voice(voice_id: str, user_id: str = DEFAULT_USER) -> bool:
    """Delete a voice profile and its reference audio (session-scoped)."""
    metadata = load_cloned_voices_metadata(user_id)
    record = metadata.get(voice_id)
    if not record:
        return False

    audio_path = Path(record.get("audio_path", ""))
    # Only ever delete files we created inside this session's directory.
    if audio_path.exists() and _user_dir(user_id) in audio_path.resolve().parents:
        try:
            audio_path.unlink()
        except OSError as exc:
            logger.warning("Could not delete %s: %s", audio_path, exc)

    del metadata[voice_id]
    _write_metadata(metadata, user_id)
    return True


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
def rate_string_from_speed(speed_factor: float) -> str:
    """Convert a speed multiplier into an edge-tts rate string."""
    rate_percent = int(round((float(speed_factor) - 1.0) * 100))
    rate_percent = max(-90, min(200, rate_percent))
    return f"{'+' if rate_percent >= 0 else ''}{rate_percent}%"


async def _generate_edge_tts(text, voice_name, rate_str, output_path):
    communicate = edge_tts.Communicate(text, voice_name, rate=rate_str)
    await communicate.save(output_path)


def synthesize_neural_speech(text, edge_voice, speed_factor=1.0, output_path=None):
    """Generate speech with the neural TTS engine (edge-tts)."""
    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    if output_path is None:
        output_path = str(new_output_path("tts", "mp3"))
    output_path = str(output_path)

    rate_str = rate_string_from_speed(speed_factor)
    try:
        asyncio.run(_generate_edge_tts(text, edge_voice, rate_str, output_path))
    except RuntimeError as exc:
        # Defensive: fall back to a private loop if one is already running.
        logger.debug("asyncio.run failed (%s); using a dedicated event loop.", exc)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_generate_edge_tts(text, edge_voice, rate_str, output_path))
        finally:
            loop.close()

    logger.info("Synthesized %s (%s)", output_path, edge_voice)
    return output_path


def synthesize_xtts_clone(xtts_model, text, speaker_wav_path, language="en", output_path=None):
    """Generate cloned speech with Coqui XTTS v2 from a reference sample."""
    if output_path is None:
        output_path = str(new_output_path("clone", "wav"))
    xtts_model.tts_to_file(
        text=text,
        speaker_wav=speaker_wav_path,
        language=language,
        file_path=str(output_path),
        split_sentences=True,
    )
    return str(output_path)


def postprocess_audio(input_path, pitch_semitones=0.0, gain_db=0.0, output_path=None):
    """Apply a pitch shift and/or volume gain to an audio file.

    Returns a WAV path, or the original path when there is nothing to change.
    Falls back to the original file if processing is unavailable.
    """
    pitch_semitones = float(pitch_semitones or 0.0)
    gain_db = float(gain_db or 0.0)

    if abs(pitch_semitones) < 1e-6 and abs(gain_db) < 1e-6:
        return str(input_path)

    try:
        samples, sample_rate = librosa.load(str(input_path), sr=None, mono=True)
        if abs(pitch_semitones) >= 1e-6:
            samples = librosa.effects.pitch_shift(samples, sr=sample_rate, n_steps=pitch_semitones)
        if abs(gain_db) >= 1e-6:
            samples = samples * (10.0 ** (gain_db / 20.0))

        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 1.0:  # avoid clipping
            samples = samples / peak

        if output_path is None:
            output_path = str(new_output_path("processed", "wav"))
        soundfile.write(str(output_path), samples, int(sample_rate))
        return str(output_path)
    except Exception as exc:  # librosa/soundfile can fail on odd inputs
        logger.warning("Post-processing failed (%s); returning original audio.", exc)
        return str(input_path)


# ---------------------------------------------------------------------------
# Speech recognition
# ---------------------------------------------------------------------------
def transcribe_audio_file(asr_pipeline, audio_file):
    """Transcribe an uploaded audio file via a local ASR model or Google Web Speech."""
    audio_bytes = audio_file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name

    # 1. Local ASR pipeline (faster-whisper WhisperModel or a HF pipeline).
    if asr_pipeline is not None:
        try:
            if hasattr(asr_pipeline, "transcribe"):
                segments, _info = asr_pipeline.transcribe(temp_path, beam_size=1)
                text = " ".join(segment.text for segment in segments).strip()
                if text:
                    return text, temp_path, None
            else:
                audio, _ = librosa.load(temp_path, sr=16000)
                result = asr_pipeline(audio)
                text = result.get("text", "").strip()
                if text:
                    return text, temp_path, None
        except Exception as exc:
            logger.warning("Local ASR failed, falling back to cloud STT: %s", exc)

    # 2. Cloud STT (Google Web Speech via SpeechRecognition).
    #    NOTE: audio is uploaded to Google's service for transcription.
    try:
        import speech_recognition as sr

        data, _ = librosa.load(temp_path, sr=16000, mono=True)
        converted_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        soundfile.write(converted_wav, data, 16000)

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 250
        recognizer.dynamic_energy_threshold = True

        with sr.AudioFile(converted_wav) as source:
            audio_data = recognizer.record(source)

        try:
            os.remove(converted_wav)
        except OSError:
            pass

        for lang_code in ("en-IN", "en-US", "hi-IN"):
            try:
                text = recognizer.recognize_google(audio_data, language=lang_code)
                if text and text.strip():
                    return text.strip(), temp_path, None
            except sr.UnknownValueError:
                continue
            except sr.RequestError as exc:
                return None, temp_path, f"Speech service error: {exc}"
    except ImportError:
        return None, temp_path, "SpeechRecognition is not installed on this instance."
    except Exception as exc:
        return None, temp_path, f"Speech recognition error: {exc}"

    return (
        None,
        temp_path,
        "Could not detect clear words. The recording may be too quiet, noisy, or unsupported.",
    )


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def translate_text(translator_pipeline, text, target_lang_code="hi"):
    """Translate text with a HuggingFace pipeline, falling back to MyMemory.

    NOTE: the MyMemory fallback sends the text to a public web service.
    """
    if translator_pipeline is not None:
        try:
            result = translator_pipeline(text)
            return result[0]["translation_text"]
        except Exception as exc:
            logger.warning("HF translation failed, using fallback: %s", exc)

    try:
        url = (
            "https://api.mymemory.translated.net/get?q="
            f"{urllib.parse.quote(text)}&langpair=en|{target_lang_code}"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode())
            translated = payload.get("responseData", {}).get("translatedText")
            if translated:
                return translated
    except Exception as exc:
        logger.warning("Fallback translation failed: %s", exc)

    return text


# ---------------------------------------------------------------------------
# Evaluation metrics (real, measurable numbers)
# ---------------------------------------------------------------------------
def compute_audio_metrics(audio_path):
    """Return basic acoustic statistics for an audio file."""
    try:
        samples, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
        return {
            "duration_s": round(float(librosa.get_duration(y=samples, sr=sample_rate)), 2),
            "sample_rate": int(sample_rate),
            "rms": round(float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0, 4),
            "peak": round(float(np.max(np.abs(samples))) if samples.size else 0.0, 4),
        }
    except Exception as exc:
        logger.warning("Could not compute metrics for %s: %s", audio_path, exc)
        return {"duration_s": None, "sample_rate": None, "rms": None, "peak": None}


def generation_metrics(start_ts, end_ts, audio_path=None):
    """Latency, audio duration and real-time factor for one generation."""
    latency_ms = int((end_ts - start_ts) * 1000)
    stats = compute_audio_metrics(audio_path) if audio_path else {}
    duration = stats.get("duration_s")
    rtf = round((latency_ms / 1000.0) / duration, 3) if duration else None
    return {
        "latency_ms": latency_ms,
        "duration_s": duration,
        "rtf": rtf,
        "sample_rate": stats.get("sample_rate"),
    }


def _logmel_profile(samples, sample_rate=16000, n_mels=40):
    """Mean log-mel spectral envelope, z-scored (content-robust speaker profile)."""
    mel = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_mels=n_mels)
    logmel = librosa.power_to_db(mel, ref=np.max).mean(axis=1)
    return (logmel - logmel.mean()) / (logmel.std() + 1e-9)


def speaker_similarity(reference_path, candidate_path):
    """Spectral similarity in [0, 1] between two clips.

    Correlation between the mean log-mel spectral envelopes of the reference and
    the generated audio. This is a lightweight, content-robust speaker proxy
    (identical voice/content scores ~1.0; different voices drop lower) rather than
    a trained speaker-verification model.
    """
    try:
        reference, _ = librosa.load(str(reference_path), sr=16000, mono=True)
        candidate, _ = librosa.load(str(candidate_path), sr=16000, mono=True)
        if reference.size < 1600 or candidate.size < 1600:
            return None

        profile_ref = _logmel_profile(reference)
        profile_cand = _logmel_profile(candidate)
        denom = float(np.linalg.norm(profile_ref) * np.linalg.norm(profile_cand))
        if denom == 0:
            return None
        cosine = float(np.dot(profile_ref, profile_cand) / denom)
        return round(max(0.0, min(1.0, cosine)), 3)
    except Exception as exc:
        logger.warning("Similarity computation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Rule-based voice assistant
# ---------------------------------------------------------------------------
AGENT_PERSONAS = {
    "Alex (Tech Support Specialist)": {
        "voice": "en-US-ChristopherNeural",
        "intro": "Hello! I am Alex, your Technical Support Engineer. How can I assist you with your systems today?",
        "prompt": "friendly, highly competent Senior Technical Support Engineer",
    },
    "Sarah (Executive Career Coach)": {
        "voice": "en-US-JennyNeural",
        "intro": "Welcome! I'm Sarah, your Executive Career Coach. Let's discuss your career goals, interview preparation, or leadership strategies.",
        "prompt": "empathetic, insightful Executive Career and Leadership Coach",
    },
    "Maya (Language & Culture Tutor)": {
        "voice": "en-GB-SoniaNeural",
        "intro": "Hello there! I'm Maya, your language and cultural tutor. Which language or dialect would you like to explore today?",
        "prompt": "warm, multilingual language and culture educator",
    },
    "Leo (Creative Storyteller & Roleplay)": {
        "voice": "en-GB-RyanNeural",
        "intro": "Greetings adventurer! I am Leo, your master storyteller. Give me a premise or world, and let's craft an epic tale together.",
        "prompt": "imaginative fantasy storyteller with rich, descriptive prose",
    },
}


def generate_agent_response(persona_name, user_input, conversation_history=None):
    """Return a scripted reply for the rule-based voice assistant.

    This is intentionally a transparent keyword-matching responder, not an LLM.
    """
    persona_name = persona_name if persona_name in AGENT_PERSONAS else next(iter(AGENT_PERSONAS))
    persona = AGENT_PERSONAS[persona_name]
    text = (user_input or "").strip().lower()
    history = conversation_history or []

    if any(greeting in text for greeting in ("hello", "hi ", "hi", "hey", "greetings")):
        reply = f"{persona['intro']} What is on your mind?"
    elif "who are you" in text or "what can you do" in text:
        reply = (
            f"I'm {persona_name.split(' ')[0]}, a rule-based voice assistant in VoxStudio. "
            "I match keywords to scripted answers and speak them aloud. I'm a demo of the "
            "speech pipeline, not a language model."
        )
    elif "interview" in text:
        reply = (
            "Use the STAR method: Situation, Task, Action and Result. Highlight your direct "
            "contribution and any number that shows impact."
        )
    elif "voice" in text or "clone" in text:
        reply = (
            "Voice cloning extracts a speaker embedding from a short reference clip, then "
            "conditions a synthesis model on it so new text keeps that timbre."
        )
    elif "python" in text or "code" in text or "bug" in text:
        reply = (
            "When debugging in Python, reproduce the failure in isolation, read the traceback "
            "top to bottom, and lock the fix in with a test."
        )
    elif "language" in text or "translate" in text:
        reply = (
            "Translation maps text between languages, and text-to-speech then renders the "
            "translated text with a target-language voice."
        )
    elif "thank" in text:
        reply = "You're very welcome! Anything else you'd like to explore?"
    elif len(history) > 0 and text in ("more", "continue", "go on", "and"):
        reply = "Sure — tell me which part you'd like me to expand on."
    else:
        reply = (
            f"Thanks for the question about '{user_input.strip()}'. I work on keyword matching, "
            "so try asking about interviews, voice cloning, Python, or translation."
        )

    return reply, persona["voice"]
