# logic.py - VoxStudio platform engine: synthesis, voice profiles, ASR,
# translation, audio post-processing and evaluation metrics.
import os
import sys
import io
import json
import uuid
import asyncio
import logging
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import librosa
import soundfile
import edge_tts

# Windows consoles default to a legacy code page (e.g. cp1252). XTTS logs the
# text it is about to synthesize, so a non-Latin script (Hindi, Japanese, ...)
# raises UnicodeEncodeError *inside* the engine and aborts synthesis. Force UTF-8
# with replacement so synthesis can never fail because of console encoding.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError, OSError):
        pass

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
    """Resolve a session's storage directory without touching the filesystem.

    Reads must not have write side effects: creating the directory here would
    leave one empty directory behind for every session that ever loaded a page.
    Writers call :func:`_ensure_user_dir` instead.
    """
    return CLONED_VOICES_DIR / _safe_component(user_id)


def _ensure_user_dir(user_id: str = DEFAULT_USER) -> Path:
    """Create the per-session storage directory, for write paths only."""
    directory = _user_dir(user_id)
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
# Engine capability detection
#
# The app must never claim a capability it does not have. These probes run at
# startup and are cheap: they check importability and (for XTTS) whether the
# weights are already on disk. They never load a multi-gigabyte model.
# ---------------------------------------------------------------------------
_CAPABILITY_CACHE = None


def _tts_cache_roots():
    """Known Coqui TTS cache locations, newest layout first."""
    home = Path.home()
    candidates = [
        os.environ.get("TTS_HOME"),
        str(home / ".local" / "share" / "tts"),
        str(home / "AppData" / "Local" / "tts"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "tts") if os.environ.get("LOCALAPPDATA") else None,
        os.path.join(os.environ.get("APPDATA", ""), "tts") if os.environ.get("APPDATA") else None,
    ]
    return [Path(c) for c in candidates if c]


def xtts_model_cached() -> bool:
    """True when the XTTS v2 checkpoint is already downloaded."""
    for root in _tts_cache_roots():
        try:
            if any(root.glob("tts_models--multilingual--multi-dataset--xtts_v2/*.pth")):
                return True
        except OSError:
            continue
    return False


@dataclass(frozen=True)
class Capability:
    """One engine's availability, with the honest reason either way."""

    key: str
    label: str
    available: bool
    detail: str


def _detect_capabilities_uncached() -> list:
    """Probe every optional engine and report availability + caveats.

    This is deliberately non-fatal: a missing engine degrades one feature, it
    does not stop the app.
    """
    capabilities = []

    # 1. Neural TTS (edge-tts) - the always-available baseline.
    try:
        import edge_tts as _edge_tts  # noqa: F401

        capabilities.append(
            Capability(
                "neural_tts",
                "Neural TTS (edge-tts)",
                True,
                "Microsoft neural voices. Requires outbound internet.",
            )
        )
    except Exception as exc:  # pragma: no cover - import guard
        capabilities.append(
            Capability("neural_tts", "Neural TTS (edge-tts)", False, f"Not importable: {exc}")
        )

    # 2. Voice cloning (Coqui XTTS v2).
    try:
        import TTS as _tts  # noqa: F401

        cached = xtts_model_cached()
        detail = (
            "Model cached locally."
            if cached
            else "Weights (~1.8 GB) download on first use."
        )
        capabilities.append(
            Capability(
                "cloning",
                "Voice cloning (XTTS v2)",
                True,
                f"{detail} Runs on CPU (~4x realtime) and needs ~4 GB RAM.",
            )
        )
    except Exception as exc:
        capabilities.append(
            Capability(
                "cloning",
                "Voice cloning (XTTS v2)",
                False,
                f"XTTS v2 needs ~4 GB RAM and runs locally via 'streamlit run app.py' (cloud sandbox limited to 1 GB; {exc})",
            )
        )

    # 3. Local speech recognition (faster-whisper).
    try:
        import faster_whisper as _fw  # noqa: F401

        capabilities.append(
            Capability(
                "asr_local",
                "Local ASR (faster-whisper)",
                True,
                "Runs on CPU (int8). Audio never leaves the machine.",
            )
        )
    except Exception as exc:
        capabilities.append(
            Capability("asr_local", "Local ASR (faster-whisper)", False, f"Not importable: {exc}")
        )

    # 4. Local translation (HuggingFace transformers).
    try:
        import transformers as _tf  # noqa: F401

        capabilities.append(
            Capability(
                "translation_local",
                "Local translation (transformers)",
                True,
                "Marian/M2M models. Large first-time download; fully local after.",
            )
        )
    except Exception as exc:
        capabilities.append(
            Capability(
                "translation_local",
                "Local translation (transformers)",
                False,
                f"Not importable: {exc}",
            )
        )

    return capabilities


def detect_capabilities() -> list:
    """Probe engines once per process.

    Capability availability cannot change while the process runs, and Streamlit
    re-executes the script on every interaction, so the (potentially slow)
    import probes are memoised rather than repeated on each rerun.
    """
    global _CAPABILITY_CACHE
    if _CAPABILITY_CACHE is None:
        _CAPABILITY_CACHE = _detect_capabilities_uncached()
    return _CAPABILITY_CACHE


def capability_summary() -> dict:
    """Convenience view: {key: available} for quick checks in the UI."""
    return {cap.key: cap.available for cap in detect_capabilities()}


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------
class ConsentRequired(ValueError):
    """Raised when a voice clone is requested without explicit consent."""


CONSENT_REQUIRED_HINT = (
    "I confirm that this is my own voice, or that I have the speaker's explicit "
    "permission to clone it, and that I will not use it to impersonate or mislead anyone."
)


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

# Neural voice used only when a voice profile cannot be cloned. Chosen from the
# declared gender so the fallback is at least plausible; the UI always labels it
# as a fallback so it is never mistaken for the profile's own voice.
_FALLBACK_EDGE_VOICES = {
    "Male": "en-US-ChristopherNeural",
    "Female": "en-US-JennyNeural",
    "Non-Binary": "en-US-AriaNeural",
}
DEFAULT_FALLBACK_EDGE_VOICE = "en-US-AriaNeural"


def fallback_edge_voice(profile: dict) -> str:
    """Neural voice to use when a voice profile cannot be cloned."""
    return _FALLBACK_EDGE_VOICES.get((profile or {}).get("gender", ""), DEFAULT_FALLBACK_EDGE_VOICE)


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
    with open(_ensure_user_dir(user_id) / "voices_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def save_cloned_voice_file(
    name,
    audio_bytes,
    description="",
    role="Custom Voice",
    gender="Unknown",
    accent="Custom",
    user_id: str = DEFAULT_USER,
    consent_confirmed: bool = False,
    consent_note: str = "",
    consent_owner: str = "",
):
    """Store a reference recording as a reusable voice profile.

    The reference sample is what the XTTS v2 engine conditions on for true
    zero-shot cloning. This does not train or extract a speaker embedding.

    Cloning requires explicit consent: ``consent_confirmed`` must be True, and
    the confirmation is recorded with the profile so the basis for use is
    auditable. The uploaded audio is decoded and normalised to 16 kHz mono WAV
    (uploads may be MP3, which would otherwise not be usable as a ``.wav``).
    """
    if not consent_confirmed:
        raise ConsentRequired(
            "Voice cloning requires explicit confirmation that you own this voice "
            "or have the speaker's permission to clone it."
        )

    if not audio_bytes:
        raise ValueError("The uploaded audio sample is empty.")

    try:
        samples, _sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    except Exception as exc:
        raise ValueError(f"Could not decode the uploaded audio: {exc}") from exc

    if samples.size < 16000:  # shorter than ~1 second
        raise ValueError(
            "The uploaded sample is too short. Please provide at least a few "
            "seconds of clear speech."
        )

    voice_id = f"voice_{uuid.uuid4().hex[:12]}"
    file_path = _ensure_user_dir(user_id) / f"{voice_id}.wav"
    soundfile.write(str(file_path), samples, 16000)

    metadata = load_cloned_voices_metadata(user_id)
    metadata[voice_id] = {
        "voice_id": voice_id,
        "name": name,
        "gender": gender,
        "accent": accent,
        "role": role,
        "desc": description or "Voice profile created from an uploaded reference sample.",
        "audio_path": str(file_path),
        "sample_rate": 16000,
        "duration_s": round(float(samples.size) / 16000.0, 2),
        "is_cloned": True,
        "avatar": "🧬",
        "consent": {
            "confirmed": True,
            "confirmed_at": datetime.now().isoformat(timespec="seconds"),
            "declared_owner": consent_owner.strip() or "self",
            "note": consent_note.strip(),
        },
    }
    _write_metadata(metadata, user_id)
    logger.info("Saved voice profile '%s' (%s) for session %s.", name, voice_id, user_id)
    return metadata[voice_id]


def profile_has_consent(profile: dict) -> bool:
    """True when a stored voice profile carries a recorded consent confirmation."""
    consent = (profile or {}).get("consent") or {}
    return bool(consent.get("confirmed"))


def delete_cloned_voice(voice_id: str, user_id: str = DEFAULT_USER) -> bool:
    """Delete a voice profile and its reference audio (session-scoped)."""
    metadata = load_cloned_voices_metadata(user_id)
    record = metadata.get(voice_id)
    if not record:
        return False

    audio_path = Path(record.get("audio_path", ""))
    # Only ever delete files we created inside this session's directory. Resolve
    # both sides first: on Windows the same directory can be reported with an 8.3
    # short name in one place and the long name in another, which would otherwise
    # make an owned file look foreign and leak it.
    try:
        resolved_audio = audio_path.resolve()
        resolved_root = _user_dir(user_id).resolve()
        owned = resolved_audio.parent == resolved_root or resolved_root in resolved_audio.parents
    except OSError as exc:
        logger.warning("Could not resolve %s: %s", audio_path, exc)
        owned = False

    if owned and resolved_audio.exists():
        try:
            resolved_audio.unlink()
        except OSError as exc:
            logger.warning("Could not delete %s: %s", resolved_audio, exc)

    del metadata[voice_id]
    _write_metadata(metadata, user_id)
    return True


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SynthesisPlan:
    """How a piece of text will actually be spoken, and why.

    ``engine`` is "xtts" only when the reference sample will genuinely be used
    for zero-shot cloning. Every other case resolves to the neural engine and
    says so, so a "voice profile" can never silently speak with an unrelated
    built-in voice.
    """

    engine: str
    label: str
    reference_path: str
    edge_voice: str
    reason: str
    is_true_clone: bool


def plan_synthesis(voice_info, *, cloning_enabled, cloning_loaded, cloning_available: bool = True) -> SynthesisPlan:
    """Resolve the synthesis engine for one voice without silent substitution."""
    is_cloned = bool((voice_info or {}).get("is_cloned"))
    reference_path = (voice_info or {}).get("audio_path")
    edge_voice = (voice_info or {}).get("edge_voice") or "en-US-ChristopherNeural"
    has_reference = bool(reference_path) and os.path.exists(str(reference_path))

    if not is_cloned:
        return SynthesisPlan(
            engine="neural",
            label="Neural TTS (edge-tts)",
            reference_path="",
            edge_voice=edge_voice,
            reason="Built-in neural voice.",
            is_true_clone=False,
        )

    # A voice profile: cloning is the whole point, so be explicit when it cannot happen.
    if not has_reference:
        return SynthesisPlan(
            engine="neural",
            label="Neural fallback - no reference audio",
            reference_path="",
            edge_voice=edge_voice,
            reason="This voice profile has no usable reference recording, so it speaks "
            "with a built-in voice instead of the reference.",
            is_true_clone=False,
        )
    if not cloning_available:
        return SynthesisPlan(
            engine="neural",
            label="Neural fallback - cloning unavailable on cloud",
            reference_path="",
            edge_voice=edge_voice,
            reason="Streamlit Cloud has a 1 GB RAM limit, so local XTTS v2 is disabled here. Run locally on your PC via 'streamlit run app.py' to use true zero-shot voice cloning.",
            is_true_clone=False,
        )
    if not cloning_enabled:
        return SynthesisPlan(
            engine="neural",
            label="Neural fallback - cloning turned off",
            reference_path="",
            edge_voice=edge_voice,
            reason="Cloning is turned off in the sidebar, so this profile speaks with a built-in voice, "
            "not the reference sample.",
            is_true_clone=False,
        )
    if not cloning_loaded:
        return SynthesisPlan(
            engine="neural",
            label="Neural fallback - cloning engine unavailable",
            reference_path="",
            edge_voice=edge_voice,
            reason="The XTTS v2 cloning engine could not be loaded, so this profile speaks "
            "with a built-in voice instead of the reference sample.",
            is_true_clone=False,
        )
    return SynthesisPlan(
        engine="xtts",
        label="XTTS v2 zero-shot clone",
        reference_path=str(reference_path),
        edge_voice=edge_voice,
        reason="Reference sample is cloned with XTTS v2.",
        is_true_clone=True,
    )


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
    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text.")
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


def prepare_reference_wav(audio_path, output_path=None):
    """Decode any supported audio into a 16 kHz mono WAV for XTTS conditioning.

    Uploads may be MP3/M4A (or MP3 bytes in a file named ``.wav``), which XTTS
    cannot reliably condition on. Decoding to a real 16 kHz mono WAV removes the
    format ambiguity and the sample-rate mismatch in one place.
    """
    samples, _sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
    if not samples.size:
        raise ValueError("The reference recording could not be decoded.")
    if output_path is None:
        output_path = str(new_output_path("reference", "wav"))
    soundfile.write(str(output_path), samples, 16000)
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
@dataclass
class TranscriptionResult:
    """A transcription plus the backend that produced it."""

    text: str = None
    audio_path: str = ""
    error: str = None
    backend: str = "none"
    language: str = None
    language_probability: float = None

    def __iter__(self):
        """Keep legacy ``text, path, error = result`` unpacking working."""
        return iter((self.text, self.audio_path, self.error))


def transcribe_audio_file(asr_pipeline, audio_file, *, allow_cloud=True, language=None):
    """Transcribe an uploaded audio file and report which backend was used.

    Local engines (faster-whisper or a HuggingFace pipeline) run first. Google
    Web Speech is only used when ``allow_cloud`` is True, because it uploads the
    audio to Google. ``language`` is an optional hint; when omitted the model
    auto-detects and the detected language is returned.
    """
    audio_bytes = audio_file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name

    # 1. Local ASR pipeline (faster-whisper WhisperModel or a HF pipeline).
    if asr_pipeline is not None:
        try:
            if hasattr(asr_pipeline, "transcribe"):
                kwargs = {"beam_size": 1}
                if language:
                    kwargs["language"] = language
                segments, info = asr_pipeline.transcribe(temp_path, **kwargs)
                text = " ".join(segment.text for segment in segments).strip()
                if text:
                    probability = getattr(info, "language_probability", None)
                    return TranscriptionResult(
                        text=text,
                        audio_path=temp_path,
                        backend="faster-whisper (local)",
                        language=getattr(info, "language", language),
                        language_probability=(round(float(probability), 3) if probability else None),
                    )
            else:
                audio, _ = librosa.load(temp_path, sr=16000)
                result = asr_pipeline(audio)
                text = result.get("text", "").strip()
                if text:
                    return TranscriptionResult(
                        text=text,
                        audio_path=temp_path,
                        backend="transformers ASR (local)",
                        language=language,
                    )
        except Exception as exc:
            logger.warning("Local ASR failed, falling back to cloud STT: %s", exc)

    # 2. Cloud STT (Google Web Speech). Only with consent: audio leaves the machine.
    if not allow_cloud:
        return TranscriptionResult(
            audio_path=temp_path,
            backend="none",
            error="Local speech recognition could not read this audio, and cloud "
            "transcription is disabled. Try a clearer recording or raise the model size.",
        )

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

        language_candidates = tuple(dict.fromkeys(c for c in (language, "en-IN", "en-US", "hi-IN") if c))
        for lang_code in language_candidates:
            try:
                text = recognizer.recognize_google(audio_data, language=lang_code)
                if text and text.strip():
                    return TranscriptionResult(
                        text=text.strip(),
                        audio_path=temp_path,
                        backend="Google Web Speech (cloud)",
                        language=lang_code,
                    )
            except sr.UnknownValueError:
                continue
            except sr.RequestError as exc:
                return TranscriptionResult(
                    audio_path=temp_path,
                    backend="none",
                    error=f"Speech service error: {exc}",
                )
    except ImportError:
        return TranscriptionResult(
            audio_path=temp_path,
            backend="none",
            error="SpeechRecognition is not installed on this instance.",
        )
    except Exception as exc:
        return TranscriptionResult(
            audio_path=temp_path,
            backend="none",
            error=f"Speech recognition error: {exc}",
        )

    return TranscriptionResult(
        audio_path=temp_path,
        backend="none",
        error="Could not detect clear words. The recording may be too quiet, noisy, or unsupported.",
    )


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
class TranslationResult(str):
    """A translated string that also records which backend produced it."""

    backend: str

    def __new__(cls, text, backend):
        instance = super().__new__(cls, text)
        instance.backend = backend
        return instance

    def __getnewargs__(self):
        # Keep copy/pickle round-trips working.
        return (str(self), self.backend)


def translate_text(translator_pipeline, text, target_lang_code="hi", *, allow_cloud=True):
    """Translate text and record which backend produced the result.

    The local HuggingFace pipeline is preferred. The MyMemory fallback sends the
    text to a public web service, so it is only used when ``allow_cloud`` is True.
    """
    if translator_pipeline is not None:
        try:
            result = translator_pipeline(text)
            return TranslationResult(result[0]["translation_text"], "transformers (local)")
        except Exception as exc:
            logger.warning("HF translation failed, using fallback: %s", exc)

    if not allow_cloud:
        return TranslationResult(text, "none - cloud disabled (text unchanged)")

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
                return TranslationResult(translated, "MyMemory (cloud)")
    except Exception as exc:
        logger.warning("Fallback translation failed: %s", exc)

    return TranslationResult(text, "none - translation failed (text unchanged)")


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
# Voice conversion (speech-to-speech morph)
#
# Converts the *timbre* of source speech toward a target speaker while keeping
# the source's words and timing, because it transforms the actual waveform
# instead of re-synthesising text. It is honest DSP - pitch/register matching
# plus mean spectral-envelope transfer - not a trained neural VC model. A
# neural converter (e.g. OpenVoice / RVC) would sound more like the target but
# needs torch and a GPU, which the hosted CPU tier does not provide.
# ---------------------------------------------------------------------------
DEFAULT_CONVERSION_SR = 22050


@dataclass
class VoiceConversionResult:
    """Outcome of a voice morph, with the numbers needed to judge it."""

    output_path: str
    source_pitch_hz: float = None
    target_pitch_hz: float = None
    semitones: float = 0.0
    sample_rate: int = DEFAULT_CONVERSION_SR
    duration_s: float = None


def estimate_pitch_hz(samples, sample_rate, fmin=60.0, fmax=400.0):
    """Median voiced F0 in Hz, or None when it cannot be estimated.

    Uses YIN on the raw waveform and keeps only in-range frames, so silence and
    unvoiced noise do not drag the estimate toward the extremes.
    """
    try:
        if samples.size < sample_rate * 0.1:
            return None
        f0 = librosa.yin(samples, fmin=fmin, fmax=fmax, sr=sample_rate, frame_length=2048)
        f0 = f0[np.isfinite(f0)]
        f0 = f0[(f0 >= fmin) & (f0 <= fmax)]
        if f0.size == 0:
            return None
        return round(float(np.median(f0)), 1)
    except Exception as exc:
        logger.warning("Pitch estimation failed: %s", exc)
        return None


def _mean_logmel_db(samples, sample_rate, n_mels=40, n_fft=2048, hop_length=512):
    """Mean log-mel power envelope in dB with a fixed reference.

    ``ref=1.0`` (rather than ``ref=np.max``) keeps the values comparable across
    different clips, which the envelope transfer below relies on.
    """
    mel = librosa.feature.melspectrogram(
        y=samples, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    return librosa.power_to_db(mel, ref=1.0).mean(axis=1)


def _zscore(vec):
    return (vec - vec.mean()) / (vec.std() + 1e-9)


def _smooth_bands(vec, width=3):
    """Moving average across mel bands, to avoid narrow-band artefacts."""
    if width <= 1 or vec.size < width:
        return vec
    return np.convolve(vec, np.ones(width, dtype=float) / width, mode="same")


def _apply_spectral_envelope(samples, sample_rate, target_profile_db, max_gain_db=9.0,
                             n_fft=2048, hop_length=512, n_mels=40):
    """Tilt the magnitude spectrum so its mel envelope matches the target's.

    Phase is left untouched, so timing and intelligibility are preserved while
    the perceived timbre moves toward the target speaker.
    """
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    spectrum = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(spectrum), np.angle(spectrum)

    source_profile_db = librosa.power_to_db(mel_basis @ (magnitude ** 2), ref=1.0).mean(axis=1)
    gain_db = np.clip(target_profile_db - source_profile_db, -max_gain_db, max_gain_db)
    gain_db = _smooth_bands(gain_db, 3)

    # Project per-mel-band gains onto FFT bins via the band-centre frequencies.
    bin_freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    band_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sample_rate / 2.0)
    bin_gain = np.interp(bin_freqs, band_freqs, 10.0 ** (gain_db / 20.0))

    boosted = magnitude * bin_gain[:, None]
    return librosa.istft(boosted * np.exp(1j * phase), hop_length=hop_length, length=len(samples))


def morph_voice(source_path, target_path, strength=1.0, sample_rate=DEFAULT_CONVERSION_SR,
                max_pitch_ratio=2.0):
    """Convert source speech toward the target speaker's timbre.

    Steps: match the median pitch toward the target (bounded, so an odd estimate
    cannot produce a cartoon voice) and transfer the target's mean spectral
    envelope. Returns a :class:`VoiceConversionResult`.
    """
    source, _ = librosa.load(str(source_path), sr=sample_rate, mono=True)
    if source.size < sample_rate * 0.2:
        raise ValueError("Source audio is too short to convert (need at least ~0.2 s).")

    target, _ = librosa.load(str(target_path), sr=sample_rate, mono=True)
    if target.size < sample_rate * 0.2:
        raise ValueError("Target reference audio is too short to characterise.")

    source_pitch = estimate_pitch_hz(source, sample_rate)
    target_pitch = estimate_pitch_hz(target, sample_rate)

    converted = source
    semitones = 0.0
    if source_pitch and target_pitch:
        ratio = target_pitch / source_pitch
        if (1.0 / max_pitch_ratio) <= ratio <= max_pitch_ratio:
            semitones = float(np.clip(12.0 * np.log2(ratio) * float(strength), -10.0, 10.0))
            if abs(semitones) >= 0.1:
                converted = librosa.effects.pitch_shift(converted, sr=sample_rate, n_steps=semitones)

    try:
        converted = _apply_spectral_envelope(converted, sample_rate, _mean_logmel_db(target, sample_rate))
    except Exception as exc:
        logger.warning("Spectral envelope transfer failed (%s); keeping the pitch shift only.", exc)

    peak = float(np.max(np.abs(converted))) if converted.size else 0.0
    if peak > 1.0:
        converted = converted / peak

    output_path = str(new_output_path("converted", "wav"))
    soundfile.write(output_path, converted, sample_rate)
    logger.info("Morphed %s toward %s (%.2f semitones).", source_path, target_path, semitones)

    return VoiceConversionResult(
        output_path=output_path,
        source_pitch_hz=source_pitch,
        target_pitch_hz=target_pitch,
        semitones=round(semitones, 2),
        sample_rate=sample_rate,
        duration_s=round(float(converted.size) / sample_rate, 2),
    )


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
