# app.py - VoxStudio (Multilingual AI Voice & Speech Toolkit)
import os
import time
import uuid
import tempfile
import sys
import importlib
import streamlit as st
from pathlib import Path

# Hot-reload sibling modules if modified on disk
for _mod_name in ("ui", "logic"):
    if _mod_name in sys.modules:
        try:
            importlib.reload(sys.modules[_mod_name])
        except Exception:
            pass

from ui import (
    inject_custom_styles,
    render_header,
    render_sidebar,
    render_voice_card,
    render_footer,
)
import logic
from logic import (
    PRESET_VOICES,
    AGENT_PERSONAS,
    load_cloned_voices_metadata,
    save_cloned_voice_file,
    delete_cloned_voice,
    synthesize_neural_speech,
    synthesize_xtts_clone,
    transcribe_audio_file,
    translate_text,
    generate_agent_response,
    postprocess_audio,
    generation_metrics,
    speaker_similarity,
    morph_voice,
    cleanup_outputs,
    safe_filename,
    detect_capabilities,
    plan_synthesis,
    profile_has_consent,
    fallback_edge_voice,
    prepare_reference_wav,
    ConsentRequired,
    xtts_model_cached,
)

CONSENT_REQUIRED_HINT = getattr(
    logic,
    "CONSENT_REQUIRED_HINT",
    "I confirm that this is my own voice, or that I have the speaker's explicit permission to clone it, and that I will not use it to impersonate or mislead anyone."
)

# -------- Streamlit Page Configuration --------
st.set_page_config(
    page_title="VoxStudio | Multilingual AI Voice Toolkit",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_styles()

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

if "agent_chat" not in st.session_state:
    st.session_state.agent_chat = []

# Per-session id: keeps one visitor's voice profiles isolated from another's.
if "user_id" not in st.session_state:
    st.session_state.user_id = uuid.uuid4().hex[:12]

USER_ID = st.session_state.user_id

# Remove stale generated audio so the server disk stays bounded.
cleanup_outputs(max_age_minutes=120)

# Top Header Navigation
render_header()

# Engine capability detection. Cheap by design: it checks importability and
# whether the XTTS weights are on disk, and never loads a multi-gigabyte model.
CAPABILITIES = detect_capabilities()
CAPABILITY_MAP = {cap.key: cap for cap in CAPABILITIES}

# Sidebar Configuration
sidebar_settings = render_sidebar(capabilities=CAPABILITIES)

@st.cache_resource
def load_xtts_model():
    """Load Coqui XTTS v2 for true zero-shot cloning if the environment supports it."""
    try:
        from TTS.api import TTS

        return TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    except Exception as exc:
        print(f"XTTS load error: {exc}")
        return None


# The cloning engine is loaded only when the analyst explicitly turns it on: it
# costs ~4 GB of RAM and ~90 s on CPU, and is cached for the server's lifetime.
XTTS_MODEL = None
if sidebar_settings["enable_cloning"] and CAPABILITY_MAP["cloning"].available:
    with st.spinner("Loading the XTTS v2 cloning engine (one-time, ~90 s on CPU)..."):
        XTTS_MODEL = load_xtts_model()
    if XTTS_MODEL is None:
        st.error(
            "The XTTS v2 cloning engine could not be loaded on this machine. "
            "Voice profiles will speak with a built-in neural voice instead."
        )

CLONING_LOADED = XTTS_MODEL is not None
ALLOW_CLOUD = not sidebar_settings["local_only"]


def audio_mime(path):
    return "audio/mpeg" if str(path).lower().endswith(".mp3") else "audio/wav"


def resolve_plan(voice_info):
    """How this voice will be produced right now, with an honest reason."""
    cloning_cap = CAPABILITY_MAP.get("cloning")
    return plan_synthesis(
        voice_info,
        cloning_enabled=sidebar_settings["enable_cloning"],
        cloning_loaded=CLONING_LOADED,
        cloning_available=bool(cloning_cap and cloning_cap.available),
    )


def show_engine_notice(plan):
    """Render the resolved engine honestly; warn whenever a clone cannot clone."""
    if plan.is_true_clone:
        st.success(f"🧬 Engine: **{plan.label}** — {plan.reason}")
    elif plan.engine == "neural" and "fallback" in plan.label.lower():
        st.warning(f"⚠️ **{plan.label}** — {plan.reason}")
    else:
        st.caption(f"Engine: **{plan.label}** — {plan.reason}")


def render_metrics(metrics, similarity=None):
    """Show real, measurable numbers for one generation."""
    cols = st.columns(4)
    cols[0].metric("Generation latency", f"{metrics['latency_ms']:,} ms")
    cols[1].metric("Audio length", f"{metrics['duration_s']} s" if metrics.get("duration_s") else "—")
    cols[2].metric("Real-time factor", f"{metrics['rtf']}" if metrics.get("rtf") else "—")
    if similarity is not None:
        cols[3].metric("Spectral similarity*", f"{similarity:.3f}")
    else:
        cols[3].metric("Sample rate", f"{metrics['sample_rate']} Hz" if metrics.get("sample_rate") else "—")
    if similarity is not None:
        st.caption("*Correlation of the log-mel spectral envelope vs. the reference sample — a lightweight proxy, not trained speaker verification.")


def materialize_upload(uploaded, fallback_suffix=".wav"):
    """Persist a Streamlit upload or mic recording to a temporary file path."""
    try:
        uploaded.seek(0)
    except Exception:
        pass
    data = uploaded.read()
    if not data:
        raise ValueError("The provided audio is empty.")
    suffix = os.path.splitext(getattr(uploaded, "name", "") or "")[1] or fallback_suffix
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        handle.write(data)
    finally:
        handle.close()
    return handle.name


# -------- Model Loading Cached Functions --------
@st.cache_resource
def load_asr_model(size="tiny"):
    """Load a local faster-whisper model for CPU transcription (no network).

    Cached per ``size`` so switching model size is the only thing that reloads.
    """
    try:
        from faster_whisper import WhisperModel

        return WhisperModel(size, device="cpu", compute_type="int8")
    except Exception:
        try:
            from transformers import pipeline

            return pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny",
                device=-1,
            )
        except Exception:
            return None


@st.cache_resource
def load_translator_model(model_name, target_lang):
    """Load a HuggingFace translation pipeline if available."""
    try:
        from transformers import pipeline

        if model_name.startswith("facebook/m2m100"):
            return pipeline(
                "translation",
                model=model_name,
                src_lang="en",
                tgt_lang=target_lang,
                device=-1,
            )
        return pipeline("translation", model=model_name, device=-1)
    except Exception:
        return None


# Multilingual Translation Config (6 languages)
TRANSLATION_LANGS = {
    "Hindi": {
        "model": "Helsinki-NLP/opus-mt-en-hi",
        "code": "hi",
        "voices": {
            "Madhur (Deep Male)": "hi-IN-MadhurNeural",
            "Swara / Kavya (Expressive Female)": "hi-IN-SwaraNeural",
        },
        "default_voice": "hi-IN-MadhurNeural",
    },
    "French": {
        "model": "Helsinki-NLP/opus-mt-en-fr",
        "code": "fr",
        "voices": {
            "Henri (Sophisticated Male)": "fr-FR-HenriNeural",
            "Denise (Chic Female)": "fr-FR-DeniseNeural",
        },
        "default_voice": "fr-FR-HenriNeural",
    },
    "Japanese": {
        "model": "facebook/m2m100_418M",
        "code": "ja",
        "voices": {
            "Keita (Natural Male)": "ja-JP-KeitaNeural",
            "Nanami (Gentle Female)": "ja-JP-NanamiNeural",
        },
        "default_voice": "ja-JP-KeitaNeural",
    },
    "Spanish": {
        "model": "Helsinki-NLP/opus-mt-en-es",
        "code": "es",
        "voices": {
            "Alvaro (Confident Male)": "es-ES-AlvaroNeural",
            "Elvira (Warm Female)": "es-ES-ElviraNeural",
        },
        "default_voice": "es-ES-AlvaroNeural",
    },
    "German": {
        "model": "Helsinki-NLP/opus-mt-en-de",
        "code": "de",
        "voices": {
            "Killian (Professional Male)": "de-DE-KillianNeural",
            "Katja (Clear Female)": "de-DE-KatjaNeural",
        },
        "default_voice": "de-DE-KillianNeural",
    },
}

# Language hints for speech recognition. None lets the model auto-detect, in
# which case the detected language and its confidence are reported back.
STT_LANGUAGE_CODES = {
    "Auto-detect": None,
    "English": "en",
    "Hindi": "hi",
    "French": "fr",
    "Japanese": "ja",
    "Spanish": "es",
    "German": "de",
}


def get_all_available_voices(user_id):
    """Combine preset voices with this session's registered voice profiles."""
    all_voices = {}

    for key, voice in PRESET_VOICES.items():
        all_voices[key] = {**voice, "is_cloned": False}

    for voice_id, profile in load_cloned_voices_metadata(user_id).items():
        label = f"🧬 {profile['name']} (Voice Profile)"
        all_voices[label] = {
            "voice_id": voice_id,
            "name": profile["name"],
            "gender": profile.get("gender", "Custom"),
            "accent": profile.get("accent", "Reference"),
            "role": profile.get("role", "Voice Profile"),
            "desc": profile.get("desc", "Reference-based voice profile"),
            # Only used when the reference cannot be cloned; plan_synthesis always
            # labels that case as a fallback so it is never mistaken for the clone.
            "edge_voice": fallback_edge_voice(profile),
            "audio_path": profile.get("audio_path"),
            "is_cloned": True,
            "has_consent": profile_has_consent(profile),
            "avatar": "🧬",
        }
    return all_voices


available_voices = get_all_available_voices(USER_ID)

# -------- Main Navigation Tabs --------
tabs = st.tabs([
    "🎙️ Text to Speech",
    "📝 Transcribe (STT)",
    "🧬 Voice Profiles",
    "🎛️ Voice Changer",
    "🌍 Multilingual Translator",
    "🤖 Voice Assistant",
    "📚 Voice Library",
    "📜 Generation History",
])


# ==============================================================================
# TAB 1: TEXT TO SPEECH (TTS)
# ==============================================================================
with tabs[0]:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #fff;">Text to Speech Synthesis</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Turn written text into speech with one of the built-in neural voices. Adjust rate,
            pitch and volume from the sidebar, then generate.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_voice, col_engine = st.columns([2, 1])
    with col_voice:
        selected_voice_key = st.selectbox(
            "Select Voice Persona",
            list(available_voices.keys()),
            index=0,
        )
    voice_info = available_voices[selected_voice_key]

    # Resolve the engine *before* synthesising so the choice is visible up front
    # and never silently substituted (see logic.plan_synthesis).
    tts_plan = resolve_plan(voice_info)
    with col_engine:
        st.markdown("**Generation Engine**")
        show_engine_notice(tts_plan)

    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 0.85rem 1.2rem; margin-bottom: 1.2rem; display: flex; align-items: center; justify-content: space-between;">
        <div>
            <span style="font-size: 1.1rem; margin-right: 0.4rem;">{voice_info.get('avatar', '🎙️')}</span>
            <b style="color: #fff;">{voice_info['name']}</b>
            <span style="color: #64748b; font-size: 0.82rem; margin-left: 0.5rem;">({voice_info['gender']} • {voice_info['accent']})</span>
            <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.2rem;">{voice_info['desc']}</div>
        </div>
        <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.75rem; border: 1px solid rgba(56, 189, 248, 0.3);">
            #{voice_info['role']}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Quick Inspiration Templates:")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    preset_text = ""
    if p_col1.button("🎬 Documentary", use_container_width=True):
        preset_text = "In the heart of the ancient forest, towering redwood trees whisper secrets that have survived for thousands of years."
    if p_col2.button("🚀 Tech Announcement", use_container_width=True):
        preset_text = "Today we are introducing VoxStudio: a toolkit for text to speech, voice profiling, translation and a voice assistant."
    if p_col3.button("☕ Relaxed Greeting", use_container_width=True):
        preset_text = "Good morning! I hope you are having a wonderful start to your day. Let's make today productive and inspiring."
    if p_col4.button("⚡ Gaming Dialogue", use_container_width=True):
        preset_text = "System alert: core reactor temperature stabilizing at ninety-eight percent. Prepare for hyperspace jump in three, two, one."

    default_input = preset_text if preset_text else "Welcome to VoxStudio. This is a neural text to speech demo with live audio metrics."
    input_text = st.text_area(
        "Enter text to synthesize",
        value=default_input,
        height=140,
        placeholder="Type or paste text here...",
    )

    char_count = len(input_text)
    st.markdown(f'<div class="char-counter">{char_count} / 2,500 characters</div>', unsafe_allow_html=True)

    if st.button("✨ Generate Speech", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("⚠️ Please provide text to synthesize.")
        else:
            with st.spinner("Synthesizing audio..."):
                start_ts = time.time()
                try:
                    reference_path = tts_plan.reference_path or None
                    used_xtts = tts_plan.engine == "xtts"

                    if used_xtts:
                        source_path = synthesize_xtts_clone(
                            xtts_model=XTTS_MODEL,
                            text=input_text,
                            speaker_wav_path=tts_plan.reference_path,
                        )
                    else:
                        source_path = synthesize_neural_speech(
                            text=input_text,
                            edge_voice=tts_plan.edge_voice,
                            speed_factor=sidebar_settings["speed_factor"],
                        )

                    # Pitch/volume post-processing backed by the sidebar sliders.
                    output_file = postprocess_audio(
                        source_path,
                        pitch_semitones=sidebar_settings["pitch_semitones"],
                        gain_db=sidebar_settings["gain_db"],
                    )

                    metrics = generation_metrics(start_ts, time.time(), output_file)
                    similarity = (
                        speaker_similarity(reference_path, output_file)
                        if used_xtts and reference_path
                        else None
                    )

                    engine_label = tts_plan.label
                    st.session_state.history.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "voice": voice_info["name"],
                        "text": input_text,
                        "file_path": output_file,
                        "engine": engine_label,
                        "metrics": metrics,
                        "similarity": similarity,
                    })

                    st.success(f"🎉 Speech synthesized — {engine_label}.")
                    st.audio(output_file)
                    render_metrics(metrics, similarity)

                    st.download_button(
                        label="⬇️ Download Audio File",
                        data=Path(output_file).read_bytes(),
                        file_name=f"voxstudio_{safe_filename(voice_info['name'])}.{str(output_file).rsplit('.', 1)[-1]}",
                        mime=audio_mime(output_file),
                        use_container_width=True,
                    )

                except Exception as exc:
                    st.error(f"Synthesis error: {exc}")


# ==============================================================================
# TAB 2: TRANSCRIBE (SPEECH TO TEXT)
# ==============================================================================
with tabs[1]:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #fff;">Speech to Text</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Transcribe a recording with a local Whisper model. The audio is processed on this
            machine; Google Web Speech is used only as a fallback, and only when it is allowed.
        </p>
    </div>
    """, unsafe_allow_html=True)

    stt_audio = st.file_uploader(
        "Upload audio (WAV, MP3, M4A, FLAC, OGG)",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        key="stt_upload",
    )
    if stt_audio:
        st.audio(stt_audio)

    stt_col1, stt_col2, stt_col3 = st.columns(3)
    with stt_col1:
        stt_model_size = st.selectbox(
            "Whisper model size",
            ["tiny", "base", "small"],
            index=0,
            help="Larger models are more accurate and noticeably slower on CPU.",
        )
    with stt_col2:
        stt_language_label = st.selectbox(
            "Language",
            list(STT_LANGUAGE_CODES.keys()),
            index=0,
            help="A hint improves speed and accuracy. Auto-detect reports the detected language.",
        )
    with stt_col3:
        stt_no_cloud = st.checkbox(
            "Never use cloud STT",
            value=not ALLOW_CLOUD,
            help="When on, audio is never sent to Google Web Speech.",
        )

    st.caption(
        "Local model: faster-whisper on CPU (int8), no network. "
        + ("Cloud fallback: allowed." if not stt_no_cloud else "Cloud fallback: disabled.")
    )

    if st.button("📝 Transcribe Audio", type="primary", use_container_width=True):
        if not stt_audio:
            st.warning("⚠️ Please upload an audio file.")
        else:
            with st.spinner("Transcribing..."):
                start_ts = time.time()
                try:
                    asr = load_asr_model(stt_model_size)
                    result = transcribe_audio_file(
                        asr,
                        stt_audio,
                        allow_cloud=not stt_no_cloud,
                        language=STT_LANGUAGE_CODES[stt_language_label],
                    )
                    if not result.text:
                        st.error(f"⚠️ {result.error or 'No speech detected in this audio.'}")
                    else:
                        elapsed = time.time() - start_ts
                        st.success("🎉 Transcription complete.")
                        st.text_area("Transcript", value=result.text, height=180)

                        meta_bits = [f"Backend: {result.backend}"]
                        if result.language:
                            meta_bits.append(f"Language: {result.language}")
                        if result.language_probability:
                            meta_bits.append(f"Confidence: {result.language_probability}")
                        meta_bits.append(f"Time: {elapsed:.2f} s")
                        st.caption(" • ".join(meta_bits))

                        if not result.backend.startswith(("faster-whisper", "transformers")):
                            st.warning(
                                "⚠️ This transcript came from a cloud service, not the local "
                                "model. Enable the local model or check the audio quality."
                            )

                        st.download_button(
                            label="⬇️ Download Transcript (.txt)",
                            data=result.text.encode("utf-8"),
                            file_name="voxstudio_transcript.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )
                except Exception as exc:
                    st.error(f"Transcription error: {exc}")


# ==============================================================================
# TAB 3: VOICE PROFILES (reference-based)
# ==============================================================================
with tabs[2]:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #fff;">Voice Profiles</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Upload a clean reference recording (10–60 s) to register a reusable voice profile.
            Profiles are stored per session. When the XTTS v2 cloning engine is enabled, a
            profile is cloned from your reference; when it is off or unavailable, the profile
            speaks with a built-in neural voice and is clearly labelled as a fallback.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cloning_cap = CAPABILITY_MAP.get("cloning")
    cloning_available = bool(cloning_cap and cloning_cap.available)

    if not cloning_available:
        st.info(
            "💡 **Running on Streamlit Cloud:** Streamlit Cloud's free sandbox has a 1 GB RAM limit, whereas deep neural voice cloning (XTTS v2) requires ~4 GB RAM. "
            "To clone your voice with true XTTS v2, run the app on your computer (`streamlit run app.py`). "
            "On this cloud server, profiles speak using natural studio voices."
        )
    elif not sidebar_settings["enable_cloning"]:
        st.warning(
            "⚠️ The XTTS v2 cloning engine is off, so voice profiles will speak with a built-in "
            "neural voice instead of your reference. Turn it on in the sidebar to clone."
        )
    elif not CLONING_LOADED:
        st.error("The cloning engine is enabled but failed to load; profiles will use a fallback voice.")
    elif not xtts_model_cached():
        st.info("ℹ️ The first clone will download roughly 1.8 GB of XTTS v2 weights.")

    clone_col1, clone_col2 = st.columns([1.2, 1])

    with clone_col1:
        st.subheader("1. Voice Profile Details")
        clone_name = st.text_input("Profile Name", placeholder="e.g., My Narration Voice")

        detail_c1, detail_c2 = st.columns(2)
        with detail_c1:
            clone_gender = st.selectbox("Gender", ["Male", "Female", "Non-Binary", "Other"])
        with detail_c2:
            clone_accent = st.selectbox(
                "Accent / Language",
                ["Indian English", "American English", "British English", "Hindi", "French", "Japanese", "Other"],
            )

        clone_role = st.selectbox(
            "Primary Use Case / Category",
            ["Narration", "Conversational", "YouTube & Social", "Education & Training", "Corporate"],
        )
        clone_desc = st.text_area("Description (Optional)", placeholder="Warm tone, natural pacing, clear articulation...")

        uploaded_sample = st.file_uploader(
            "Upload Reference Audio (WAV or MP3)",
            type=["wav", "mp3"],
            help="For best results, upload a clear recording with minimal background noise.",
        )

        st.markdown("**Consent**")
        clone_consent = st.checkbox(CONSENT_REQUIRED_HINT, value=False, key="clone_consent")
        consent_owner = st.text_input(
            "Whose voice is this?",
            value="self",
            help="Leave as 'self' for your own voice, otherwise name the person who gave permission.",
        )

        if st.button("🧬 Save Voice Profile", type="primary", use_container_width=True):
            if not clone_name.strip():
                st.warning("⚠️ Please provide a name for your voice profile.")
            elif not uploaded_sample:
                st.warning("⚠️ Please upload a reference audio sample.")
            elif not clone_consent:
                st.warning("⚠️ Please confirm you have permission to clone this voice.")
            else:
                with st.spinner("Saving voice profile..."):
                    try:
                        save_cloned_voice_file(
                            name=clone_name.strip(),
                            audio_bytes=uploaded_sample.read(),
                            description=clone_desc.strip(),
                            role=clone_role,
                            gender=clone_gender,
                            accent=clone_accent,
                            user_id=USER_ID,
                            consent_confirmed=True,
                            consent_owner=consent_owner,
                        )
                        st.success(f"✅ Voice profile '{clone_name}' added to your library!")
                        st.rerun()
                    except ConsentRequired as err:
                        st.error(f"Consent required: {err}")
                    except Exception as err:
                        st.error(f"Error saving voice profile: {err}")

    with clone_col2:
        st.subheader("2. Your Voice Profiles")
        my_profiles = load_cloned_voices_metadata(USER_ID)

        if not my_profiles:
            st.info("ℹ️ No voice profiles yet. Upload a reference sample on the left to create one.")
        else:
            for voice_id, profile in my_profiles.items():
                consented = profile_has_consent(profile)
                if consented:
                    consent = profile.get("consent", {})
                    badge = (
                        '<span style="font-size: 0.72rem; color: #10b981; background: rgba(16, 185, 129, 0.15); padding: 0.15rem 0.5rem; border-radius: 4px;">Consent recorded</span>'
                    )
                    consent_line = (
                        f"consent: {consent.get('declared_owner', 'self')} • "
                        f"{consent.get('confirmed_at', 'n/a')}"
                    )
                else:
                    badge = (
                        '<span style="font-size: 0.72rem; color: #f59e0b; background: rgba(245, 158, 11, 0.15); padding: 0.15rem 0.5rem; border-radius: 4px;">No consent on file</span>'
                    )
                    consent_line = "consent: not recorded"

                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 0.9rem; margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <b style="color: #fff; font-size: 1rem;">🧬 {profile['name']}</b>
                        {badge}
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin: 0.3rem 0;">{profile.get('desc')}</div>
                    <div style="font-size: 0.72rem; color: #64748b;">#{profile.get('role')} • {profile.get('accent')} ({profile.get('gender')})</div>
                    <div style="font-size: 0.7rem; color: #475569; margin-top: 0.25rem;">{consent_line}</div>
                </div>
                """, unsafe_allow_html=True)

                if not consented:
                    st.warning(
                        f"⚠️ '{profile['name']}' was saved before consent was required. "
                        "Delete and re-add it to record consent."
                    )

                if os.path.exists(profile.get("audio_path", "")):
                    st.audio(profile["audio_path"])

                if st.button(f"🗑️ Delete {profile['name']}", key=f"del_{voice_id}"):
                    delete_cloned_voice(voice_id, user_id=USER_ID)
                    st.success(f"Deleted {profile['name']}")
                    st.rerun()


# ==============================================================================
# TAB 4: VOICE CHANGER (SPEECH-TO-SPEECH)
# ==============================================================================
with tabs[3]:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #fff;">Voice Changer (Speech-to-Speech)</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Keep what you say and how you say it, and change <i>who</i> it sounds like, using a
            target voice you provide. Choose a waveform <b>voice morph</b> (preserves your words
            and timing) or <b>dubbing</b> (transcribes, then re-speaks the text).
        </p>
    </div>
    """, unsafe_allow_html=True)

    my_profiles = load_cloned_voices_metadata(USER_ID)
    profile_choices = {f"🧬 {p['name']}": vid for vid, p in my_profiles.items()}

    vc_col1, vc_col2 = st.columns(2)

    with vc_col1:
        st.subheader("1. Source speech")
        source_mode = st.radio(
            "Provide the speech",
            ["🎙️ Record now", "📁 Upload a file"],
            key="vc_source_mode",
            horizontal=True,
        )
        source_audio = None
        if source_mode.startswith("🎙️"):
            try:
                source_audio = st.audio_input("Record the speech to convert", key="vc_record")
            except Exception:
                st.info("Recording is unavailable in this browser; upload a file instead.")
        else:
            source_audio = st.file_uploader(
                "Upload speech (WAV, MP3, M4A, FLAC, OGG)",
                type=["wav", "mp3", "m4a", "flac", "ogg"],
                key="vc_source_audio",
            )
        if source_audio is not None:
            st.audio(source_audio)

    with vc_col2:
        st.subheader("2. Target voice")
        target_mode = st.radio(
            "Target voice source",
            ["🧬 Saved voice profile", "📁 Upload a target sample"],
            key="vc_target_mode",
            horizontal=True,
        )
        target_profile_label = None
        target_upload = None
        if target_mode.startswith("🧬"):
            if profile_choices:
                target_profile_label = st.selectbox(
                    "Select target profile",
                    list(profile_choices.keys()),
                    key="vc_target_profile",
                )
            else:
                st.info(
                    "No saved voice profiles yet. Create one in the **Voice Profiles** tab, or "
                    "upload a target sample here."
                )
        else:
            target_upload = st.file_uploader(
                "Upload a clean sample of the target voice (10–60 s)",
                type=["wav", "mp3", "m4a", "flac", "ogg"],
                key="vc_target_upload",
            )
            if target_upload is not None:
                st.audio(target_upload)

    st.markdown("---")
    conv_mode = st.radio(
        "Conversion mode",
        [
            "🎛️ Voice morph — keep my words & timing",
            "🗣️ Dub — transcribe & re-speak the text",
        ],
        key="vc_conv_mode",
        horizontal=True,
    )
    st.caption(
        "**Voice morph** transforms the actual recording, so words, timing and pauses are kept, by "
        "matching the target's pitch and spectral envelope — honest DSP, not a trained neural "
        "identity model, so it changes register and colour rather than perfectly impersonating. "
        "**Dub** generates a new performance (timing and emotion are lost) but can use a true XTTS "
        "clone when the cloning engine is enabled."
    )

    if st.button("🎛️ Convert Voice", type="primary", use_container_width=True):
        if source_audio is None:
            st.warning("⚠️ Please record or upload the speech to convert.")
        else:
            is_morph = conv_mode.startswith("🎛️")

            # Resolve the target reference audio up front (never inside the try,
            # so validation warnings are not swallowed by the error handler).
            target_reference = None
            target_name = "target voice"
            target_info = None
            resolve_error = None

            if target_mode.startswith("🧬"):
                if not target_profile_label:
                    resolve_error = "No target voice profile is available."
                else:
                    profile = my_profiles[profile_choices[target_profile_label]]
                    target_reference = profile.get("audio_path")
                    target_name = profile["name"]
                    target_info = available_voices.get(f"🧬 {target_name} (Voice Profile)")
                    if not (target_reference and os.path.exists(target_reference)):
                        resolve_error = "The selected profile has no usable reference audio."
            else:
                if target_upload is None:
                    resolve_error = "Please upload a target voice sample."
                else:
                    target_reference = materialize_upload(target_upload)
                    target_name = getattr(target_upload, "name", "") or "uploaded target sample"

            if resolve_error:
                st.warning(f"⚠️ {resolve_error}")
            elif (not is_morph) and target_info is None:
                st.warning(
                    "Dubbing needs a **saved voice profile** to speak with. Save the uploaded "
                    "sample in the Voice Profiles tab, or switch to voice morph mode."
                )
            elif is_morph:
                with st.spinner("Morphing the voice toward the target..."):
                    try:
                        start_ts = time.time()
                        source_path = materialize_upload(source_audio)
                        similarity_before = speaker_similarity(target_reference, source_path)
                        converted = morph_voice(source_path, target_reference)
                        similarity_after = speaker_similarity(target_reference, converted.output_path)
                        metrics = generation_metrics(start_ts, time.time(), converted.output_path)

                        st.session_state.history.append({
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "voice": target_name,
                            "text": f"[voice morph] source \u2192 {target_name}",
                            "file_path": converted.output_path,
                            "engine": "Voice morph (pitch + envelope)",
                            "metrics": metrics,
                            "similarity": similarity_after,
                        })

                        st.success(f"🎉 Voice morphed toward '{target_name}'.")
                        st.audio(converted.output_path)

                        pitch_cols = st.columns(3)
                        pitch_cols[0].metric(
                            "Source pitch",
                            f"{converted.source_pitch_hz} Hz" if converted.source_pitch_hz else "—",
                        )
                        pitch_cols[1].metric(
                            "Target pitch",
                            f"{converted.target_pitch_hz} Hz" if converted.target_pitch_hz else "—",
                        )
                        pitch_cols[2].metric("Pitch shift", f"{converted.semitones:+.2f} st")
                        render_metrics(metrics)

                        sim_cols = st.columns(2)
                        sim_cols[0].metric(
                            "Spectral similarity — before",
                            f"{similarity_before:.3f}" if similarity_before is not None else "—",
                        )
                        sim_cols[1].metric(
                            "Spectral similarity — after",
                            f"{similarity_after:.3f}" if similarity_after is not None else "—",
                        )
                        st.caption(
                            "Spectral similarity is the log-mel envelope correlation against the "
                            "target reference. 'After' higher than 'before' means it moved toward "
                            "the target voice."
                        )

                        st.download_button(
                            label="⬇️ Download Converted Voice",
                            data=Path(converted.output_path).read_bytes(),
                            file_name=f"converted_{safe_filename(target_name)}.wav",
                            mime="audio/wav",
                            use_container_width=True,
                        )
                    except Exception as exc:
                        st.error(f"Voice conversion error: {exc}")
            else:
                with st.spinner("Transcribing and re-synthesizing..."):
                    try:
                        start_ts = time.time()
                        result = transcribe_audio_file(
                            load_asr_model(), source_audio, allow_cloud=ALLOW_CLOUD
                        )
                        if not result.text:
                            st.warning(
                                f"⚠️ {result.error or 'Could not detect clear words in this recording.'}"
                            )
                        else:
                            st.info(f"📝 Transcribed input ({result.backend}): *\"{result.text}\"*")

                            dub_plan = resolve_plan(target_info)
                            show_engine_notice(dub_plan)

                            if dub_plan.engine == "xtts":
                                source_path = synthesize_xtts_clone(
                                    xtts_model=XTTS_MODEL,
                                    text=result.text,
                                    speaker_wav_path=dub_plan.reference_path,
                                )
                            else:
                                source_path = synthesize_neural_speech(
                                    text=result.text,
                                    edge_voice=dub_plan.edge_voice,
                                    speed_factor=sidebar_settings["speed_factor"],
                                )

                            output_file = postprocess_audio(
                                source_path,
                                pitch_semitones=sidebar_settings["pitch_semitones"],
                                gain_db=sidebar_settings["gain_db"],
                            )
                            metrics = generation_metrics(start_ts, time.time(), output_file)
                            similarity = (
                                speaker_similarity(dub_plan.reference_path, output_file)
                                if dub_plan.engine == "xtts"
                                else None
                            )

                            st.session_state.history.append({
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "voice": target_info["name"],
                                "text": result.text,
                                "file_path": output_file,
                                "engine": dub_plan.label,
                                "metrics": metrics,
                                "similarity": similarity,
                            })

                            st.success(f"🎉 Dubbed audio generated — {dub_plan.label}.")
                            st.audio(output_file)
                            render_metrics(metrics, similarity)

                            st.download_button(
                                label="⬇️ Download Dubbed Audio",
                                data=Path(output_file).read_bytes(),
                                file_name=f"dubbed_{safe_filename(target_info['name'])}.{str(output_file).rsplit('.', 1)[-1]}",
                                mime=audio_mime(output_file),
                                use_container_width=True,
                            )
                    except Exception as exc:
                        st.error(f"Dubbing error: {exc}")


# ==============================================================================
# TAB 5: MULTILINGUAL TRANSLATOR
# ==============================================================================
with tabs[4]:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #fff;">Multilingual Voice Translator</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Translate English text (typed or transcribed from audio) into Hindi, French, Japanese,
            Spanish or German, then speak it with a native neural voice or with your own cloned
            reference. Cloning into Japanese additionally requires the optional <code>cutlet</code>
            package.
        </p>
    </div>
    """, unsafe_allow_html=True)

    ml_col1, ml_col2 = st.columns([1.5, 1])

    with ml_col1:
        target_lang = st.selectbox("Select Target Language", list(TRANSLATION_LANGS.keys()))

        ml_audio = st.file_uploader(
            "Upload English voice (optional) — used for transcription and cloning",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            key="ml_voice_upload",
        )

        ml_text = st.text_area(
            "Or type / paste text to translate",
            placeholder="Type your sentence here...",
            key="ml_text_input",
        )

    with ml_col2:
        st.subheader("Voice Source")
        delivery = st.radio(
            "How should the translation be spoken?",
            ["Native neural voice", "Clone from my uploaded reference"],
            help="Cloning needs the XTTS v2 engine enabled in the sidebar and an uploaded reference.",
        )

        lang_voices = TRANSLATION_LANGS[target_lang]["voices"]
        chosen_voice_label = st.selectbox(f"Select {target_lang} Voice Persona", list(lang_voices.keys()))
        chosen_neural_voice = lang_voices[chosen_voice_label]

        st.caption("• **Native neural voice:** a native-language voice speaks the translation.")
        st.caption("• **Clone:** XTTS v2 speaks the translation in your uploaded reference's voice.")

    if st.button("🌐 Translate & Speak", type="primary", use_container_width=True):
        has_audio = ml_audio is not None
        has_text = bool(ml_text.strip())

        if not has_audio and not has_text:
            st.warning("⚠️ Please upload an audio file or type English text.")
        else:
            with st.spinner("Translating and generating speech..."):
                start_ts = time.time()
                try:
                    speaker_path = None
                    transcribed_text = None
                    if has_audio:
                        asr = load_asr_model()
                        ml_result = transcribe_audio_file(asr, ml_audio, allow_cloud=ALLOW_CLOUD)
                        transcribed_text = ml_result.text
                        speaker_path = ml_result.audio_path

                    if has_text:
                        english_text = ml_text.strip()
                    elif transcribed_text:
                        english_text = transcribed_text
                    else:
                        english_text = None

                    if not english_text:
                        st.warning("⚠️ Could not read the audio and no text was provided.")
                        st.info("💡 You can type your sentence directly into the text box above instead.")
                    else:
                        lang_info = TRANSLATION_LANGS[target_lang]
                        translator = load_translator_model(lang_info["model"], lang_info["code"])
                        translated_text = translate_text(
                            translator, english_text, lang_info["code"], allow_cloud=ALLOW_CLOUD
                        )

                        res_c1, res_c2 = st.columns(2)
                        with res_c1:
                            st.markdown(f"**Recognized English:**\n\n> {english_text}")
                        with res_c2:
                            st.markdown(f"**{target_lang} Translation:**\n\n> {translated_text}")
                        st.caption(f"Translation backend: {translated_text.backend}")

                        wants_clone = delivery == "Clone from my uploaded reference"
                        if wants_clone and speaker_path:
                            ml_plan = resolve_plan(
                                {"is_cloned": True, "audio_path": speaker_path}
                            )
                        else:
                            if wants_clone and not speaker_path:
                                st.warning(
                                    "⚠️ Cloning needs an uploaded reference recording; "
                                    "using a native neural voice instead."
                                )
                            ml_plan = resolve_plan(
                                {"is_cloned": False, "edge_voice": chosen_neural_voice}
                            )
                        show_engine_notice(ml_plan)

                        if ml_plan.engine == "xtts":
                            reference_wav = prepare_reference_wav(ml_plan.reference_path)
                            source_path = synthesize_xtts_clone(
                                xtts_model=XTTS_MODEL,
                                text=translated_text,
                                speaker_wav_path=reference_wav,
                                language=lang_info["code"],
                            )
                            similarity_reference = reference_wav
                        else:
                            source_path = synthesize_neural_speech(
                                text=translated_text,
                                edge_voice=ml_plan.edge_voice,
                                speed_factor=sidebar_settings["speed_factor"],
                            )
                            similarity_reference = None

                        output_file = postprocess_audio(
                            source_path,
                            pitch_semitones=sidebar_settings["pitch_semitones"],
                            gain_db=sidebar_settings["gain_db"],
                        )
                        metrics = generation_metrics(start_ts, time.time(), output_file)
                        similarity = (
                            speaker_similarity(similarity_reference, output_file)
                            if similarity_reference
                            else None
                        )

                        st.session_state.history.append({
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "voice": f"{target_lang} • {ml_plan.label}",
                            "text": translated_text,
                            "file_path": output_file,
                            "engine": ml_plan.label,
                            "metrics": metrics,
                            "similarity": similarity,
                        })

                        st.success(f"🎉 Generated {target_lang} speech — {ml_plan.label}.")
                        st.audio(output_file)
                        render_metrics(metrics, similarity)

                        st.download_button(
                            label=f"⬇️ Download {target_lang} Audio",
                            data=Path(output_file).read_bytes(),
                            file_name=f"translated_{target_lang.lower()}.{str(output_file).rsplit('.', 1)[-1]}",
                            mime=audio_mime(output_file),
                            use_container_width=True,
                        )

                except Exception as exc:
                    st.error(f"Translation & synthesis error: {exc}")


# ==============================================================================
# TAB 6: RULE-BASED VOICE ASSISTANT
# ==============================================================================
with tabs[5]:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #fff;">Voice Assistant (rule-based)</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            A small demo of a spoken dialogue loop: type or record a question, and the assistant
            replies with scripted, keyword-matched answers spoken aloud. It is not an LLM.
        </p>
    </div>
    """, unsafe_allow_html=True)

    agent_persona_choice = st.selectbox("Choose Assistant Persona", list(AGENT_PERSONAS.keys()))
    agent_info = AGENT_PERSONAS[agent_persona_choice]
    st.info(f"💡 **Greeting:** \"{agent_info['intro']}\"")

    agent_input = st.text_input("Talk to the assistant:", placeholder="Ask about interviews, voice cloning, Python...")

    voice_note = None
    try:
        voice_note = st.audio_input("…or record your question")
    except Exception:
        voice_note = None

    col_ag1, col_ag2 = st.columns([1, 4])
    with col_ag1:
        agent_send = st.button("💬 Send & Speak", type="primary", use_container_width=True)
    with col_ag2:
        if st.button("🧹 Clear Conversation"):
            st.session_state.agent_chat = []
            st.rerun()

    user_message = agent_input.strip()
    if not user_message and voice_note is not None:
        with st.spinner("Transcribing your question..."):
            asr = load_asr_model()
            agent_result = transcribe_audio_file(asr, voice_note, allow_cloud=ALLOW_CLOUD)
            if agent_result.text:
                user_message = agent_result.text
            else:
                st.warning(f"⚠️ {agent_result.error or 'Could not transcribe the recording.'}")

    if agent_send and user_message:
        with st.spinner("Assistant is speaking..."):
            reply_text, voice_key = generate_agent_response(
                agent_persona_choice, user_message, st.session_state.agent_chat
            )
            agent_audio = synthesize_neural_speech(text=reply_text, edge_voice=voice_key)
            agent_audio = postprocess_audio(
                agent_audio,
                pitch_semitones=sidebar_settings["pitch_semitones"],
                gain_db=sidebar_settings["gain_db"],
            )
            st.session_state.agent_chat.append({
                "user": user_message,
                "agent": reply_text,
                "audio": agent_audio,
            })

    for chat in reversed(st.session_state.agent_chat):
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["agent"])
            if os.path.exists(chat["audio"]):
                st.audio(chat["audio"])


# ==============================================================================
# TAB 7: VOICE LIBRARY
# ==============================================================================
with tabs[6]:
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h3 style="margin: 0; color: #fff;">Voice Library</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Browse the built-in neural voices and your saved voice profiles. Every card can be
            previewed with a generated sample.
        </p>
    </div>
    """, unsafe_allow_html=True)

    v_cols = st.columns(3)
    idx = 0
    for v_key, v in available_voices.items():
        with v_cols[idx % 3]:
            card_html = render_voice_card(
                voice_name=v["name"],
                role_tag=v["role"],
                accent=v["accent"],
                gender=v["gender"],
                desc=v["desc"],
                is_cloned=v.get("is_cloned", False),
            )
            st.markdown(card_html, unsafe_allow_html=True)

            if st.button(f"▶️ Sample: {v['name']}", key=f"preview_{v['voice_id']}"):
                with st.spinner(f"Generating preview for {v['name']}..."):
                    preview_text = (
                        f"Hello! My name is {v['name']}. This is a sample generated by VoxStudio."
                    )
                    preview_plan = resolve_plan(v)
                    try:
                        if preview_plan.engine == "xtts":
                            preview_path = synthesize_xtts_clone(
                                xtts_model=XTTS_MODEL,
                                text=preview_text,
                                speaker_wav_path=preview_plan.reference_path,
                            )
                        else:
                            preview_path = synthesize_neural_speech(
                                text=preview_text,
                                edge_voice=preview_plan.edge_voice,
                                speed_factor=sidebar_settings["speed_factor"],
                            )
                        preview_path = postprocess_audio(
                            preview_path,
                            pitch_semitones=sidebar_settings["pitch_semitones"],
                            gain_db=sidebar_settings["gain_db"],
                        )
                        st.caption(preview_plan.label)
                        st.audio(preview_path)
                    except Exception as exc:
                        st.error(f"Preview error: {exc}")
        idx += 1


# ==============================================================================
# TAB 8: GENERATION HISTORY
# ==============================================================================
with tabs[7]:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #fff;">Generation History</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Every clip synthesized during this session, with its measured metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No audio generated yet in this session.")
    else:
        for item in reversed(st.session_state.history):
            metrics = item.get("metrics", {})
            similarity = item.get("similarity")
            sim_text = f" • similarity {similarity:.3f}" if similarity is not None else ""
            rtf_text = f" • RTF {metrics.get('rtf')}" if metrics.get("rtf") else ""
            st.markdown(f"""
            <div style="background: rgba(17, 24, 39, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 1rem; margin-bottom: 0.8rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                    <b style="color: #fff;">🎙️ {item['voice']} <span style="color:#64748b; font-weight:400; font-size:0.8rem;">({item.get('engine', 'n/a')})</span></b>
                    <span style="font-size: 0.75rem; color: #64748b;">{item['timestamp']}</span>
                </div>
                <div style="color: #cbd5e1; font-size: 0.85rem; margin-bottom: 0.5rem;">\"{item['text'][:220]}\"</div>
                <div style="font-size: 0.72rem; color: #64748b;">{metrics.get('latency_ms', '—')} ms • {metrics.get('duration_s', '—')} s{rtf_text}{sim_text}</div>
            </div>
            """, unsafe_allow_html=True)
            if os.path.exists(item["file_path"]):
                st.audio(item["file_path"])


# Footer
render_footer()
