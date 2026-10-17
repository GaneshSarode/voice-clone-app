# ui.py - VoxStudio UI and styling system
import streamlit as st

VOXSTUDIO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background-color: #0b0f17;
    color: #e2e8f0;
}

/* Header & Branding */
.vox-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.25rem 1.75rem;
    background: linear-gradient(180deg, rgba(19, 26, 38, 0.8) 0%, rgba(11, 15, 23, 0.4) 100%);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.vox-brand {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}

.vox-logo-badge {
    background: linear-gradient(135deg, #2563eb, #38bdf8);
    color: white;
    font-weight: 800;
    font-size: 1.15rem;
    padding: 0.35rem 0.7rem;
    border-radius: 8px;
    letter-spacing: -0.5px;
    box-shadow: 0 0 15px rgba(37, 99, 235, 0.5);
}

.vox-brand-name {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
}

.vox-tagline {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-left: 0.5rem;
}

.vox-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.85rem;
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 500;
    color: #38bdf8;
}

.vox-status-dot {
    width: 7px;
    height: 7px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}

/* Cards */
.vox-card {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}

.voice-card {
    background: rgba(17, 24, 39, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}

.voice-card:hover {
    border-color: #38bdf8;
    background: rgba(30, 41, 59, 0.6);
    transform: translateY(-2px);
}

.voice-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.4rem;
}

.voice-name {
    font-weight: 600;
    font-size: 0.98rem;
    color: #ffffff;
}

.voice-badge {
    font-size: 0.7rem;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    background: rgba(37, 99, 235, 0.2);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.3);
    font-weight: 500;
}

.voice-desc {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-bottom: 0.4rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #111827;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: 500;
    font-size: 0.88rem;
    color: #94a3b8;
    background-color: transparent;
    border: none;
    transition: all 0.15s ease;
}

.stTabs [aria-selected="true"] {
    background-color: #1e293b !important;
    color: #38bdf8 !important;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: #ffffff;
    font-weight: 600;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 0.6rem 1.4rem;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5);
    transform: translateY(-1px);
    border-color: rgba(255, 255, 255, 0.3);
}

/* Text Area */
.stTextArea textarea {
    background-color: #111827 !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    font-size: 0.95rem;
    line-height: 1.6;
    padding: 1rem;
}

.stTextArea textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
}

/* File Uploader */
.stFileUploader {
    background-color: rgba(17, 24, 39, 0.5);
    border-radius: 12px;
    padding: 0.5rem;
}

/* Character Counter */
.char-counter {
    display: flex;
    justify-content: flex-end;
    font-size: 0.78rem;
    color: #64748b;
    margin-top: -8px;
    margin-bottom: 12px;
}

/* Footer */
.vox-footer {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
    color: #64748b;
    font-size: 0.82rem;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 3rem;
}
</style>
"""


def inject_custom_styles():
    """Inject the app's custom CSS."""
    st.markdown(VOXSTUDIO_CSS, unsafe_allow_html=True)


def render_header():
    """Render the top branding bar."""
    st.markdown(
        """
    <div class="vox-navbar">
        <div class="vox-brand">
            <span class="vox-logo-badge">VS</span>
            <div>
                <span class="vox-brand-name">VoxStudio</span>
                <span class="vox-tagline">Multilingual AI Voice &amp; Speech Toolkit</span>
            </div>
        </div>
        <div style="display: flex; gap: 0.6rem; align-items: center;">
            <div class="vox-status-pill">
                <span class="vox-status-dot"></span>
                <span>Neural TTS online</span>
            </div>
            <div class="vox-status-pill" style="color: #a78bfa; border-color: rgba(167, 139, 250, 0.3);">
                <span>12 voices &middot; 6 languages</span>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar controls and return the audio settings."""
    with st.sidebar:
        st.markdown(
            """
        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1rem;">
            <span style="background: #2563eb; color: white; padding: 0.2rem 0.5rem; border-radius: 6px; font-weight: bold; font-size: 0.85rem;">VS</span>
            <span style="font-weight: 700; font-size: 1.1rem; color: #fff;">Audio Settings</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.caption("The engine is Microsoft edge-tts neural voices. Rate is applied "
                   "during synthesis; pitch and gain are applied as post-processing.")

        speed_factor = st.slider(
            "Speech Rate",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.05,
            help="Speaking speed. Applied by the TTS engine as a rate change.",
        )
        pitch_semitones = st.slider(
            "Pitch Shift (semitones)",
            min_value=-6.0,
            max_value=6.0,
            value=0.0,
            step=0.5,
            help="Shifts pitch without changing speed (librosa post-processing).",
        )
        gain_db = st.slider(
            "Volume Gain (dB)",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            help="Loudness adjustment applied after synthesis.",
        )

        st.markdown("---")
        st.markdown(
            """
        <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.5;">
            <b>⚡ Capabilities:</b><br/>
            • 12 built-in neural voices<br/>
            • Voice profile library (reference-based)<br/>
            • Speech-to-speech dubbing<br/>
            • Translation across 6 languages<br/>
            • Rule-based voice assistant<br/>
            • Metrics: latency, RTF, spectral similarity
        </div>
        """,
            unsafe_allow_html=True,
        )

        return {
            "speed_factor": speed_factor,
            "pitch_semitones": pitch_semitones,
            "gain_db": gain_db,
        }


def render_voice_card(voice_name, role_tag, accent, gender, desc, is_cloned=False):
    """Render a card for the voice library grid."""
    badge_color = "#10b981" if is_cloned else "#38bdf8"
    bg_color = "rgba(16, 185, 129, 0.15)" if is_cloned else "rgba(56, 189, 248, 0.15)"
    label = "🧬 Voice Profile" if is_cloned else f"{gender} • {accent}"

    return f"""
    <div class="voice-card">
        <div class="voice-card-header">
            <span class="voice-name">🎙️ {voice_name}</span>
            <span class="voice-badge" style="color: {badge_color}; background: {bg_color}; border-color: {badge_color}44;">
                {label}
            </span>
        </div>
        <div class="voice-desc">{desc}</div>
        <div style="display: flex; gap: 0.4rem; font-size: 0.72rem; color: #64748b;">
            <span>#{role_tag}</span>
            <span>•</span>
            <span>{accent}</span>
        </div>
    </div>
    """


def render_footer():
    """Render the page footer."""
    st.markdown(
        """
    <div class="vox-footer">
        <b>VoxStudio</b> — multilingual TTS, voice profiles, translation and a rule-based
        voice assistant in one Streamlit app.<br/>
        Speech synthesis by edge-tts (Microsoft neural voices) · optional Coqui XTTS v2 cloning ·
        Whisper / Google Web Speech ASR · librosa audio metrics.<br/>
        <span style="font-size: 0.75rem;">Built as a learning/portfolio project. Not affiliated with any commercial voice API.</span>
    </div>
    """,
        unsafe_allow_html=True,
    )
