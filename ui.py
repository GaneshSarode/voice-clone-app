# ui.py
import streamlit as st


def render_header():
    st.markdown(
        """
    <h1 style="text-align:center;">
        🎙️ Multi-Language Voice Cloning
    </h1>
    <p style="text-align:center; color:gray;">
        Clone your voice and speak new languages — CPU-only
    </p>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("## 📊 Project Dashboard")
        st.markdown(
            """
        **Engine:** XTTS v2
        **ASR:** Whisper
        **Translation:** HuggingFace
        **Hardware:** CPU only
        """
        )
        st.markdown("---")
        st.markdown("🧠 *Educational / Research use*")


def render_status():
    col1, col2, col3 = st.columns(3)
    col1.metric("Mode", "CPU")
    col2.metric("Voice Clone", "XTTS v2")
    col3.metric("Latency", "Slow ⏳")
