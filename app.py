# app.py - ElevenVoice Studio (AI Voice Generator & Voice Agents Platform)
import os
import time
import tempfile
import streamlit as st

from ui import (
    inject_custom_styles,
    render_header,
    render_sidebar,
    render_voice_card,
    render_footer
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
    generate_agent_response
)

# -------- Streamlit Page Configuration --------
st.set_page_config(
    page_title="ElevenVoice Studio | AI Voice Platform",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject ElevenLabs Styles
inject_custom_styles()

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

if "agent_chat" not in st.session_state:
    st.session_state.agent_chat = []

# Top Header Navigation
render_header()

# Sidebar Configuration
sidebar_settings = render_sidebar()


# -------- Model Loading Cached Functions --------
@st.cache_resource
def load_asr_model():
    """Load Whisper ASR for Speech-to-Text if available."""
    try:
        from transformers import pipeline
        return pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=-1
        )
    except Exception:
        return None


@st.cache_resource
def load_translator_model(model_name, target_lang):
    """Load HuggingFace translation pipeline if available."""
    try:
        from transformers import pipeline
        if model_name.startswith("facebook/m2m100"):
            return pipeline(
                "translation",
                model=model_name,
                src_lang="en",
                tgt_lang=target_lang,
                device=-1
            )
        return pipeline(
            "translation",
            model=model_name,
            device=-1
        )
    except Exception:
        return None


@st.cache_resource
def load_xtts_model():
    """Load Coqui XTTS v2 for deep zero-shot voice cloning if available."""
    try:
        from TTS.api import TTS
        return TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=False
        )
    except Exception:
        return None


# Multilingual Translation Config
TRANSLATION_LANGS = {
    "Hindi": {
        "model": "Helsinki-NLP/opus-mt-en-hi",
        "code": "hi",
        "voices": {
            "Madhur (Deep Male)": "hi-IN-MadhurNeural",
            "Swara / Kavya (Expressive Female)": "hi-IN-SwaraNeural"
        },
        "default_voice": "hi-IN-MadhurNeural",
        "file_suffix": "hindi_speech.mp3"
    },
    "French": {
        "model": "Helsinki-NLP/opus-mt-en-fr",
        "code": "fr",
        "voices": {
            "Henri (Sophisticated Male)": "fr-FR-HenriNeural",
            "Denise (Chic Female)": "fr-FR-DeniseNeural"
        },
        "default_voice": "fr-FR-HenriNeural",
        "file_suffix": "french_speech.mp3"
    },
    "Japanese": {
        "model": "facebook/m2m100_418M",
        "code": "ja",
        "voices": {
            "Keita (Natural Male)": "ja-JP-KeitaNeural",
            "Nanami (Gentle Female)": "ja-JP-NanamiNeural"
        },
        "default_voice": "ja-JP-KeitaNeural",
        "file_suffix": "japanese_speech.mp3"
    },
    "Spanish": {
        "model": "Helsinki-NLP/opus-mt-en-es",
        "code": "es",
        "voices": {
            "Alvaro (Confident Male)": "es-ES-AlvaroNeural",
            "Elvira (Warm Female)": "es-ES-ElviraNeural"
        },
        "default_voice": "es-ES-AlvaroNeural",
        "file_suffix": "spanish_speech.mp3"
    },
    "German": {
        "model": "Helsinki-NLP/opus-mt-en-de",
        "code": "de",
        "voices": {
            "Killian (Professional Male)": "de-DE-KillianNeural",
            "Katja (Clear Female)": "de-DE-KatjaNeural"
        },
        "default_voice": "de-DE-KillianNeural",
        "file_suffix": "german_speech.mp3"
    }
}


def get_all_available_voices():
    """Combine preset voices and user's custom cloned voices."""
    all_voices = {}
    
    # 1. Preset ElevenLabs Voices
    for key, v in PRESET_VOICES.items():
        all_voices[key] = {**v, "is_cloned": False}
        
    # 2. Custom Cloned Voices
    cloned_metadata = load_cloned_voices_metadata()
    for v_id, c in cloned_metadata.items():
        label = f"🧬 {c['name']} (Custom Cloned Voice)"
        all_voices[label] = {
            "voice_id": v_id,
            "name": c["name"],
            "gender": c.get("gender", "Custom"),
            "accent": c.get("accent", "Cloned"),
            "role": c.get("role", "Cloned Persona"),
            "desc": c.get("desc", "Custom cloned voice"),
            "edge_voice": "en-US-ChristopherNeural",
            "audio_path": c.get("audio_path"),
            "is_cloned": True,
            "avatar": "🧬"
        }
    return all_voices


available_voices = get_all_available_voices()

# -------- Main Navigation Tabs --------
tabs = st.tabs([
    "🎙️ Text to Speech",
    "🧬 Instant Voice Cloning",
    "🔄 Voice Changer (Speech to Speech)",
    "🌍 Multilingual Voice Clone Translator",
    "🤖 Conversational AI Voice Agent",
    "📚 Voice Library",
    "📜 Generation History"
])


# ==============================================================================
# TAB 1: TEXT TO SPEECH (TTS)
# ==============================================================================
with tabs[0]:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #fff;">Text to Speech Synthesis</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Transform written text into natural, emotional and expressive speech using studio-grade AI voices.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_voice, col_model = st.columns([2, 1])
    with col_voice:
        selected_voice_key = st.selectbox(
            "Select Voice Persona",
            list(available_voices.keys()),
            index=0
        )
    with col_model:
        tts_engine = st.selectbox(
            "Speech Engine",
            ["Neural Studio Engine (Instant)", "XTTS v2 Deep Cloning Engine (Slow)"],
            index=0
        )

    voice_info = available_voices[selected_voice_key]

    # Voice Detail Card
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

    # Prompt Template Buttons
    st.caption("Quick Inspiration Templates:")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    preset_text = ""
    if p_col1.button("🎬 Documentary", use_container_width=True):
        preset_text = "In the heart of the ancient forest, towering redwood trees whisper secrets that have survived for thousands of years."
    if p_col2.button("🚀 Tech Announcement", use_container_width=True):
        preset_text = "Today, we are thrilled to introduce ElevenVoice Studio: the next leap in voice synthesis, real-time cloning, and conversational agents."
    if p_col3.button("☕ Relaxed Greeting", use_container_width=True):
        preset_text = "Good morning! I hope you are having a wonderful start to your day. Let's make today productive and inspiring."
    if p_col4.button("⚡ Gaming Dialogue", use_container_width=True):
        preset_text = "System alert: core reactor temperature stabilizing at ninety-eight percent. Prepare for hyperspace jump in three, two, one."

    # Text Input
    default_input = preset_text if preset_text else "Welcome to ElevenVoice Studio. Experience next-generation voice cloning and human-like conversational intelligence."
    input_text = st.text_area(
        "Enter text to synthesize",
        value=default_input,
        height=140,
        placeholder="Type or paste text here..."
    )

    char_count = len(input_text)
    st.markdown(f'<div class="char-counter">{char_count} / 2,500 characters</div>', unsafe_allow_html=True)

    # Generation Button
    if st.button("✨ Generate Speech", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("⚠️ Please provide text to synthesize.")
        else:
            with st.spinner("Synthesizing audio with studio precision..."):
                output_file = f"tts_{int(time.time())}.mp3"
                
                try:
                    # If user chose XTTS engine and voice is a cloned voice
                    if "XTTS" in tts_engine and voice_info.get("is_cloned") and voice_info.get("audio_path"):
                        xtts = load_xtts_model()
                        if xtts is not None:
                            output_file = f"cloned_{int(time.time())}.wav"
                            synthesize_xtts_clone(
                                xtts_model=xtts,
                                text=input_text,
                                speaker_wav_path=voice_info["audio_path"],
                                output_path=output_file
                            )
                        else:
                            st.info("💡 Running in Studio Neural mode (XTTS v2 is reserved for local GPU/high-RAM machines).")
                            edge_voice = voice_info.get("edge_voice", "en-US-ChristopherNeural")
                            synthesize_neural_speech(
                                text=input_text,
                                edge_voice=edge_voice,
                                speed_factor=sidebar_settings["speed_factor"],
                                output_path=output_file
                            )
                    else:
                        # High-quality neural engine
                        edge_voice = voice_info.get("edge_voice", "en-US-ChristopherNeural")
                        synthesize_neural_speech(
                            text=input_text,
                            edge_voice=edge_voice,
                            speed_factor=sidebar_settings["speed_factor"],
                            output_path=output_file
                        )

                    # Log to history
                    st.session_state.history.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "voice": voice_info["name"],
                        "text": input_text,
                        "file_path": output_file
                    })

                    st.success("🎉 Speech synthesized successfully!")
                    st.audio(output_file)
                    
                    with open(output_file, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Audio File",
                            data=f,
                            file_name=f"elevenvoice_{voice_info['name'].lower()}.mp3" if output_file.endswith(".mp3") else f"elevenvoice_{voice_info['name'].lower()}.wav",
                            mime="audio/mpeg" if output_file.endswith(".mp3") else "audio/wav",
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"Synthesis error: {str(e)}")


# ==============================================================================
# TAB 2: INSTANT VOICE CLONING (IVC)
# ==============================================================================
with tabs[1]:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #fff;">Instant Voice Cloning (IVC)</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Clone your own voice in seconds. Upload a clean recording of speech (10–60 seconds) to create a reusable voice profile.
        </p>
    </div>
    """, unsafe_allow_html=True)

    clone_col1, clone_col2 = st.columns([1.2, 1])

    with clone_col1:
        st.subheader("1. Voice Profile Details")
        clone_name = st.text_input("Voice Name", placeholder="e.g., Ganesh Signature Voice")
        
        detail_c1, detail_c2 = st.columns(2)
        with detail_c1:
            clone_gender = st.selectbox("Gender", ["Male", "Female", "Non-Binary", "Other"])
        with detail_c2:
            clone_accent = st.selectbox("Accent / Language", ["Indian English", "American English", "British English", "Hindi", "French", "Japanese", "Other"])

        clone_role = st.selectbox("Primary Use Case / Category", ["Narration", "Conversational", "YouTube & Social", "Education & Training", "Corporate"])
        clone_desc = st.text_area("Description (Optional)", placeholder="Warm tone, natural pacing, clear articulation...")

        uploaded_sample = st.file_uploader(
            "Upload Audio Sample (WAV or MP3)",
            type=["wav", "mp3"],
            help="For best quality, upload a clear recording with minimal background noise."
        )

        if st.button("🧬 Create Cloned Voice Profile", type="primary", use_container_width=True):
            if not clone_name.strip():
                st.warning("⚠️ Please provide a name for your cloned voice.")
            elif not uploaded_sample:
                st.warning("⚠️ Please upload an audio sample.")
            else:
                with st.spinner("Extracting vocal characteristics and saving profile..."):
                    try:
                        voice_record = save_cloned_voice_file(
                            name=clone_name.strip(),
                            audio_bytes=uploaded_sample.read(),
                            description=clone_desc.strip(),
                            role=clone_role,
                            gender=clone_gender,
                            accent=clone_accent
                        )
                        st.success(f"✅ Voice '{clone_name}' cloned and added to your Voice Library!")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Error cloning voice: {str(err)}")

    with clone_col2:
        st.subheader("2. Your Cloned Voices")
        my_clones = load_cloned_voices_metadata()
        
        if not my_clones:
            st.info("ℹ️ No cloned voices created yet. Upload a sample on the left to create your first clone.")
        else:
            for v_id, c in my_clones.items():
                with st.container():
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 0.9rem; margin-bottom: 0.75rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <b style="color: #fff; font-size: 1rem;">🧬 {c['name']}</b>
                            <span style="font-size: 0.72rem; color: #10b981; background: rgba(16, 185, 129, 0.15); padding: 0.15rem 0.5rem; border-radius: 4px;">Active</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #94a3b8; margin: 0.3rem 0;">{c.get('desc')}</div>
                        <div style="font-size: 0.72rem; color: #64748b;">#{c.get('role')} • {c.get('accent')} ({c.get('gender')})</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if os.path.exists(c.get("audio_path", "")):
                        st.audio(c["audio_path"])
                    
                    if st.button(f"🗑️ Delete {c['name']}", key=f"del_{v_id}"):
                        delete_cloned_voice(v_id)
                        st.success(f"Deleted {c['name']}")
                        st.rerun()


# ==============================================================================
# TAB 3: VOICE CHANGER (SPEECH TO SPEECH)
# ==============================================================================
with tabs[2]:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #fff;">Voice Changer (Speech to Speech)</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Transform your speech into any target voice while preserving emotion and message.
        </p>
    </div>
    """, unsafe_allow_html=True)

    vc_col1, vc_col2 = st.columns(2)

    with vc_col1:
        st.subheader("1. Input Speech")
        source_audio = st.file_uploader(
            "Upload Speech Recording (WAV)",
            type=["wav"],
            key="vc_source_audio"
        )
        if source_audio:
            st.audio(source_audio)

    with vc_col2:
        st.subheader("2. Target Voice Persona")
        target_voice_choice = st.selectbox(
            "Select Target Voice",
            list(available_voices.keys()),
            key="vc_target_voice"
        )
        target_info = available_voices[target_voice_choice]
        st.caption(f"Role: {target_info['role']} | Accent: {target_info['accent']} ({target_info['gender']})")

    if st.button("🔄 Transform Voice", type="primary", use_container_width=True):
        if not source_audio:
            st.warning("⚠️ Please upload a speech recording.")
        else:
            with st.spinner("Transcribing speech with Whisper and resynthesizing into target voice..."):
                try:
                    asr = load_asr_model()
                    recognized_text, temp_wav = transcribe_audio_file(asr, source_audio)
                    
                    st.info(f"📝 Transcribed Input: *\"{recognized_text}\"*")
                    
                    vc_out = f"vc_{int(time.time())}.mp3"
                    edge_voice = target_info.get("edge_voice", "en-US-ChristopherNeural")
                    
                    synthesize_neural_speech(
                        text=recognized_text,
                        edge_voice=edge_voice,
                        speed_factor=sidebar_settings["speed_factor"],
                        output_path=vc_out
                    )

                    st.success("🎉 Voice transformed successfully!")
                    st.audio(vc_out)

                    with open(vc_out, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Converted Voice",
                            data=f,
                            file_name=f"voice_changed_{target_info['name'].lower()}.mp3",
                            mime="audio/mpeg",
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"Voice Changer error: {str(e)}")


# ==============================================================================
# TAB 4: MULTILINGUAL VOICE CLONE TRANSLATOR
# ==============================================================================
with tabs[3]:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #fff;">Multilingual Voice Clone Translator</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Speak English and hear yourself speaking fluent Hindi, French, Japanese, Spanish, or German in your own voice!
        </p>
    </div>
    """, unsafe_allow_html=True)

    ml_col1, ml_col2 = st.columns([1.5, 1])

    with ml_col1:
        target_lang = st.selectbox(
            "Select Target Language",
            list(TRANSLATION_LANGS.keys())
        )
        
        ml_audio = st.file_uploader(
            "Upload English voice (WAV) — used for transcription & voice cloning",
            type=["wav"],
            key="ml_voice_upload"
        )
        
        ml_text = st.text_area(
            "Or type English text (if you don't have audio)",
            placeholder="Type your English statement here...",
            key="ml_text_input"
        )

    with ml_col2:
        st.subheader("Synthesis Engine")
        engine_mode = st.radio(
            "Voice Delivery Mode",
            [
                "🚀 Studio Neural Translation (Instant, High Fidelity)",
                "🧬 XTTS v2 Zero-Shot Voice Clone (Your exact voice timbre - CPU slow)"
            ]
        )
        
        # Speaker Persona Selection (Male / Female)
        lang_voices = TRANSLATION_LANGS[target_lang]["voices"]
        chosen_voice_label = st.selectbox(
            f"Select {target_lang} Voice Persona",
            list(lang_voices.keys())
        )
        chosen_neural_voice = lang_voices[chosen_voice_label]

        st.caption("• **Studio Neural:** Translates and speaks with natural fluent native accent.")
        st.caption("• **XTTS v2:** Deep learning model that clones your acoustic fingerprint into the new language.")

    if st.button("🌐 Translate & Clone Speech", type="primary", use_container_width=True):
        has_audio = ml_audio is not None
        has_text = bool(ml_text.strip())

        if not has_audio and not has_text:
            st.warning("⚠️ Please upload an audio file or type English text.")
        else:
            with st.spinner("Translating and generating multilingual voice synthesis..."):
                try:
                    speaker_path = None
                    if has_audio:
                        asr = load_asr_model()
                        english_text, speaker_path = transcribe_audio_file(asr, ml_audio)
                    else:
                        english_text = ml_text.strip()

                    # If user provided both audio AND typed custom text, prioritize typed text for translation
                    if has_audio and has_text:
                        english_text = ml_text.strip()

                    lang_info = TRANSLATION_LANGS[target_lang]
                    
                    # Translation step
                    translator = load_translator_model(lang_info["model"], lang_info["code"])
                    translated_text = translate_text(translator, english_text, lang_info["code"])

                    # Display translated text side by side
                    res_c1, res_c2 = st.columns(2)
                    with res_c1:
                        st.markdown(f"**Recognized English:**\n\n> {english_text}")
                    with res_c2:
                        st.markdown(f"**{target_lang} Translation:**\n\n> {translated_text}")

                    out_path = f"translated_{int(time.time())}.wav" if "XTTS" in engine_mode else f"translated_{int(time.time())}.mp3"

                    if "XTTS" in engine_mode and speaker_path:
                        xtts = load_xtts_model()
                        if xtts is not None:
                            synthesize_xtts_clone(
                                xtts_model=xtts,
                                text=translated_text,
                                speaker_wav_path=speaker_path,
                                language=lang_info["code"],
                                output_path=out_path
                            )
                        else:
                            st.info(f"💡 Synthesized via Studio Neural Voice ({chosen_voice_label}).")
                            out_path = f"translated_{int(time.time())}.mp3"
                            synthesize_neural_speech(
                                text=translated_text,
                                edge_voice=chosen_neural_voice,
                                speed_factor=sidebar_settings["speed_factor"],
                                output_path=out_path
                            )
                    else:
                        synthesize_neural_speech(
                            text=translated_text,
                            edge_voice=chosen_neural_voice,
                            speed_factor=sidebar_settings["speed_factor"],
                            output_path=out_path
                        )

                    st.success(f"🎉 Generated {target_lang} voice!")
                    st.audio(out_path)

                    with open(out_path, "rb") as f:
                        st.download_button(
                            label=f"⬇️ Download {target_lang} Audio",
                            data=f,
                            file_name=f"translated_{target_lang.lower()}.mp3" if out_path.endswith(".mp3") else f"translated_{target_lang.lower()}.wav",
                            mime="audio/mpeg" if out_path.endswith(".mp3") else "audio/wav",
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"Translation & Voice Clone error: {str(e)}")


# ==============================================================================
# TAB 5: CONVERSATIONAL AI VOICE AGENT
# ==============================================================================
with tabs[4]:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #fff;">Conversational AI Voice Agents</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Interactive, autonomous AI voice personas capable of natural dialogue and instant spoken replies.
        </p>
    </div>
    """, unsafe_allow_html=True)

    agent_persona_choice = st.selectbox(
        "Choose Voice Agent Persona",
        list(AGENT_PERSONAS.keys())
    )
    
    agent_info = AGENT_PERSONAS[agent_persona_choice]
    st.info(f"💡 **Agent Greeting:** \"{agent_info['intro']}\"")

    agent_input = st.text_input("Talk to your Voice Agent:", placeholder="Ask anything or propose a scenario...")
    
    col_ag1, col_ag2 = st.columns([1, 4])
    with col_ag1:
        agent_send = st.button("💬 Send & Speak", type="primary", use_container_width=True)
    with col_ag2:
        if st.button("🧹 Clear Conversation"):
            st.session_state.agent_chat = []
            st.rerun()

    if agent_send and agent_input.strip():
        with st.spinner("Agent thinking & speaking..."):
            reply_text, voice_key = generate_agent_response(agent_persona_choice, agent_input.strip())
            
            agent_audio = f"agent_{int(time.time())}.mp3"
            synthesize_neural_speech(
                text=reply_text,
                edge_voice=voice_key,
                output_path=agent_audio
            )
            
            st.session_state.agent_chat.append({
                "user": agent_input.strip(),
                "agent": reply_text,
                "audio": agent_audio
            })

    # Display Conversation History
    for chat in reversed(st.session_state.agent_chat):
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["agent"])
            if os.path.exists(chat["audio"]):
                st.audio(chat["audio"])


# ==============================================================================
# TAB 6: VOICE LIBRARY & SHOWCASE
# ==============================================================================
with tabs[5]:
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h3 style="margin: 0; color: #fff;">Voice Library & Community Showcase</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            Explore premium curated personas and your custom cloned voices.
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
                is_cloned=v.get("is_cloned", False)
            )
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Quick preview button
            if st.button(f"▶️ Sample: {v['name']}", key=f"preview_{v['voice_id']}"):
                with st.spinner(f"Generating preview for {v['name']}..."):
                    preview_path = f"sample_{v['voice_id']}.mp3"
                    preview_text = f"Hello! My name is {v['name']}. This is an authentic sample of my voice generated by ElevenVoice Studio."
                    edge_v = v.get("edge_voice", "en-US-ChristopherNeural")
                    synthesize_neural_speech(preview_text, edge_v, output_path=preview_path)
                    st.audio(preview_path)
        idx += 1


# ==============================================================================
# TAB 7: GENERATION HISTORY
# ==============================================================================
with tabs[6]:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #fff;">Generation History & Studio Logs</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
            All synthesized audio clips from this session.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No audio generated yet in this session.")
    else:
        for item in reversed(st.session_state.history):
            with st.container():
                st.markdown(f"""
                <div style="background: rgba(17, 24, 39, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 1rem; margin-bottom: 0.8rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                        <b style="color: #fff;">🎙️ Voice: {item['voice']}</b>
                        <span style="font-size: 0.75rem; color: #64748b;">{item['timestamp']}</span>
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-bottom: 0.8rem;">\"{item['text']}\"</div>
                </div>
                """, unsafe_allow_html=True)
                if os.path.exists(item["file_path"]):
                    st.audio(item["file_path"])


# Footer
render_footer()
