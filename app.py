# app.py - VoxStudio (Multilingual AI Voice & Speech Toolkit)
import os
import time
import uuid
import streamlit as st
from pathlib import Path

from ui import (
    inject_custom_styles,
    render_header,
    render_sidebar,
    render_voice_card,
    render_footer,
)
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
    cleanup_outputs,
    safe_filename,
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

# Sidebar Configuration
sidebar_settings = render_sidebar()


def audio_mime(path):
    return "audio/mpeg" if str(path).lower().endswith(".mp3") else "audio/wav"


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


# -------- Model Loading Cached Functions --------
@st.cache_resource
def load_asr_model():
    """Load a lightweight local Whisper model for CPU transcription."""
    try:
        from faster_whisper import WhisperModel

        return WhisperModel("tiny", device="cpu", compute_type="int8")
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


@st.cache_resource
def load_xtts_model():
    """Load Coqui XTTS v2 for true zero-shot cloning if the environment supports it."""
    try:
        from TTS.api import TTS

        return TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
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
            "edge_voice": "en-US-ChristopherNeural",
            "audio_path": profile.get("audio_path"),
            "is_cloned": True,
            "avatar": "🧬",
        }
    return all_voices


available_voices = get_all_available_voices(USER_ID)

# -------- Main Navigation Tabs --------
tabs = st.tabs([
    "🎙️ Text to Speech",
    "🧬 Voice Profiles",
    "🔄 Speech-to-Speech Dubbing",
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

    col_voice, col_model = st.columns([2, 1])
    with col_voice:
        selected_voice_key = st.selectbox(
            "Select Voice Persona",
            list(available_voices.keys()),
            index=0,
        )
    with col_model:
        tts_engine = st.selectbox(
            "Engine",
            ["Neural engine (fast)", "XTTS v2 cloning (needs GPU)"],
            index=0,
        )

    voice_info = available_voices[selected_voice_key]

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
                    used_xtts = False
                    source_path = None
                    reference_path = voice_info.get("audio_path")

                    if "XTTS" in tts_engine and voice_info.get("is_cloned") and reference_path:
                        xtts = load_xtts_model()
                        if xtts is not None:
                            source_path = synthesize_xtts_clone(
                                xtts_model=xtts,
                                text=input_text,
                                speaker_wav_path=reference_path,
                            )
                            used_xtts = True
                        else:
                            st.info(
                                "💡 The XTTS v2 engine needs a GPU/high-RAM machine and is not "
                                "available here, so the fast neural engine is used instead."
                            )

                    if not used_xtts:
                        source_path = synthesize_neural_speech(
                            text=input_text,
                            edge_voice=voice_info.get("edge_voice", "en-US-ChristopherNeural"),
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

                    engine_label = "XTTS v2 clone" if used_xtts else "Neural engine"
                    st.session_state.history.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "voice": voice_info["name"],
                        "text": input_text,
                        "file_path": output_file,
                        "engine": engine_label,
                        "metrics": metrics,
                        "similarity": similarity,
                    })

                    st.success(f"🎉 Speech synthesized with the {engine_label}.")
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
# TAB 2: VOICE PROFILES (reference-based)
# ==============================================================================
with tabs[1]:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #fff;">Voice Profiles</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Upload a clean reference recording (10–60s) to register a reusable voice profile.
            Profiles are stored per session. When the XTTS v2 engine is available, a profile is
            used for true zero-shot cloning; otherwise it maps to a built-in neural voice.
        </p>
    </div>
    """, unsafe_allow_html=True)

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

        if st.button("🧬 Save Voice Profile", type="primary", use_container_width=True):
            if not clone_name.strip():
                st.warning("⚠️ Please provide a name for your voice profile.")
            elif not uploaded_sample:
                st.warning("⚠️ Please upload a reference audio sample.")
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
                        )
                        st.success(f"✅ Voice profile '{clone_name}' added to your library!")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Error saving voice profile: {err}")

    with clone_col2:
        st.subheader("2. Your Voice Profiles")
        my_profiles = load_cloned_voices_metadata(USER_ID)

        if not my_profiles:
            st.info("ℹ️ No voice profiles yet. Upload a reference sample on the left to create one.")
        else:
            for voice_id, profile in my_profiles.items():
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 0.9rem; margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <b style="color: #fff; font-size: 1rem;">🧬 {profile['name']}</b>
                        <span style="font-size: 0.72rem; color: #10b981; background: rgba(16, 185, 129, 0.15); padding: 0.15rem 0.5rem; border-radius: 4px;">Saved</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin: 0.3rem 0;">{profile.get('desc')}</div>
                    <div style="font-size: 0.72rem; color: #64748b;">#{profile.get('role')} • {profile.get('accent')} ({profile.get('gender')})</div>
                </div>
                """, unsafe_allow_html=True)

                if os.path.exists(profile.get("audio_path", "")):
                    st.audio(profile["audio_path"])

                if st.button(f"🗑️ Delete {profile['name']}", key=f"del_{voice_id}"):
                    delete_cloned_voice(voice_id, user_id=USER_ID)
                    st.success(f"Deleted {profile['name']}")
                    st.rerun()


# ==============================================================================
# TAB 3: SPEECH-TO-SPEECH DUBBING
# ==============================================================================
with tabs[2]:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #fff;">Speech-to-Speech Dubbing</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Transcribe an uploaded recording and re-synthesize the words in a different voice.
            This is a transcribe-then-speak pipeline, so tone and timing are not preserved —
            only the spoken content is.
        </p>
    </div>
    """, unsafe_allow_html=True)

    vc_col1, vc_col2 = st.columns(2)

    with vc_col1:
        st.subheader("1. Input Speech")
        source_audio = st.file_uploader("Upload Speech Recording (WAV)", type=["wav"], key="vc_source_audio")
        if source_audio:
            st.audio(source_audio)

    with vc_col2:
        st.subheader("2. Target Voice Persona")
        target_voice_choice = st.selectbox(
            "Select Target Voice",
            list(available_voices.keys()),
            key="vc_target_voice",
        )
        target_info = available_voices[target_voice_choice]
        st.caption(f"Role: {target_info['role']} | Accent: {target_info['accent']} ({target_info['gender']})")

    if st.button("🔄 Dub into Target Voice", type="primary", use_container_width=True):
        if not source_audio:
            st.warning("⚠️ Please upload a speech recording.")
        else:
            with st.spinner("Transcribing with Whisper and re-synthesizing..."):
                start_ts = time.time()
                try:
                    asr = load_asr_model()
                    recognized_text, _temp_wav, err = transcribe_audio_file(asr, source_audio)

                    if not recognized_text:
                        st.warning(f"⚠️ {err or 'Could not detect clear words in this recording.'}")
                        st.stop()

                    st.info(f"📝 Transcribed Input: *\"{recognized_text}\"*")

                    source_path = synthesize_neural_speech(
                        text=recognized_text,
                        edge_voice=target_info.get("edge_voice", "en-US-ChristopherNeural"),
                        speed_factor=sidebar_settings["speed_factor"],
                    )
                    output_file = postprocess_audio(
                        source_path,
                        pitch_semitones=sidebar_settings["pitch_semitones"],
                        gain_db=sidebar_settings["gain_db"],
                    )
                    metrics = generation_metrics(start_ts, time.time(), output_file)

                    st.success("🎉 Dubbed audio generated!")
                    st.audio(output_file)
                    render_metrics(metrics)

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
# TAB 4: MULTILINGUAL TRANSLATOR
# ==============================================================================
with tabs[3]:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #fff;">Multilingual Voice Translator</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Translate English text (typed or transcribed from audio) into Hindi, French, Japanese,
            Spanish or German, then speak it with a native voice.
        </p>
    </div>
    """, unsafe_allow_html=True)

    ml_col1, ml_col2 = st.columns([1.5, 1])

    with ml_col1:
        target_lang = st.selectbox("Select Target Language", list(TRANSLATION_LANGS.keys()))

        ml_audio = st.file_uploader(
            "Upload English voice (WAV) — optional, used for transcription",
            type=["wav"],
            key="ml_voice_upload",
        )

        ml_text = st.text_area(
            "Or type / paste text to translate",
            placeholder="Type your sentence here...",
            key="ml_text_input",
        )

    with ml_col2:
        st.subheader("Synthesis Engine")
        engine_mode = st.radio(
            "Voice Delivery Mode",
            [
                "🚀 Neural translation (instant)",
                "🧬 XTTS v2 zero-shot clone (needs GPU)",
            ],
        )

        lang_voices = TRANSLATION_LANGS[target_lang]["voices"]
        chosen_voice_label = st.selectbox(f"Select {target_lang} Voice Persona", list(lang_voices.keys()))
        chosen_neural_voice = lang_voices[chosen_voice_label]

        st.caption("• **Neural:** speaks the translation with a native-language voice.")
        st.caption("• **XTTS v2:** clones your own reference audio into the new language (GPU only).")

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
                    if has_audio:
                        asr = load_asr_model()
                        transcribed_text, speaker_path, err = transcribe_audio_file(asr, ml_audio)
                        if has_text:
                            english_text = ml_text.strip()
                        elif transcribed_text:
                            english_text = transcribed_text
                        else:
                            st.warning(f"⚠️ {err or 'Could not detect clear words in the audio.'}")
                            st.info("💡 You can type your sentence directly into the text box above instead.")
                            st.stop()
                    else:
                        english_text = ml_text.strip()

                    lang_info = TRANSLATION_LANGS[target_lang]
                    translator = load_translator_model(lang_info["model"], lang_info["code"])
                    translated_text = translate_text(translator, english_text, lang_info["code"])

                    res_c1, res_c2 = st.columns(2)
                    with res_c1:
                        st.markdown(f"**Recognized English:**\n\n> {english_text}")
                    with res_c2:
                        st.markdown(f"**{target_lang} Translation:**\n\n> {translated_text}")

                    used_xtts = False
                    if "XTTS" in engine_mode and speaker_path:
                        xtts = load_xtts_model()
                        if xtts is not None:
                            source_path = synthesize_xtts_clone(
                                xtts_model=xtts,
                                text=translated_text,
                                speaker_wav_path=speaker_path,
                                language=lang_info["code"],
                            )
                            used_xtts = True
                        else:
                            st.info("💡 XTTS v2 needs a GPU; using the neural voice instead.")

                    if not used_xtts:
                        source_path = synthesize_neural_speech(
                            text=translated_text,
                            edge_voice=chosen_neural_voice,
                            speed_factor=sidebar_settings["speed_factor"],
                        )

                    output_file = postprocess_audio(
                        source_path,
                        pitch_semitones=sidebar_settings["pitch_semitones"],
                        gain_db=sidebar_settings["gain_db"],
                    )
                    metrics = generation_metrics(start_ts, time.time(), output_file)
                    similarity = (
                        speaker_similarity(speaker_path, output_file)
                        if used_xtts and speaker_path
                        else None
                    )

                    st.success(f"🎉 Generated {target_lang} speech!")
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
# TAB 5: RULE-BASED VOICE ASSISTANT
# ==============================================================================
with tabs[4]:
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
            transcribed, _path, err = transcribe_audio_file(asr, voice_note)
            if transcribed:
                user_message = transcribed
            else:
                st.warning(f"⚠️ {err or 'Could not transcribe the recording.'}")

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
# TAB 6: VOICE LIBRARY
# ==============================================================================
with tabs[5]:
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
                    preview_path = synthesize_neural_speech(
                        text=preview_text,
                        edge_voice=v.get("edge_voice", "en-US-ChristopherNeural"),
                        speed_factor=sidebar_settings["speed_factor"],
                    )
                    preview_path = postprocess_audio(
                        preview_path,
                        pitch_semitones=sidebar_settings["pitch_semitones"],
                        gain_db=sidebar_settings["gain_db"],
                    )
                    st.audio(preview_path)
        idx += 1


# ==============================================================================
# TAB 7: GENERATION HISTORY
# ==============================================================================
with tabs[6]:
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
