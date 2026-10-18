# 🎙️ VoxStudio

**Multilingual AI Voice & Speech Toolkit**

A local-first speech platform that combines zero-shot voice cloning, speech-to-speech conversion, multilingual translation, and conversational voice agents — built entirely on open-source models, with no API keys required.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://voice-clone-app-ueulpkpnzcqupxtrutqbxj.streamlit.app/)

> **Note:** The hosted demo runs on Streamlit Cloud's 1 GB free tier, so XTTS v2 cloning is disabled there. Run locally for full voice cloning — see [Local Setup](#-local-setup) below.

---

## ✨ Features

### 🗣️ Text to Speech
- 12 curated neural voice personas across 5 accents (American, British, Indian English, Hindi, French, Japanese)
- Adjustable **speech rate**, **pitch shift** (semitones), and **volume gain** (dB)
- One-click inspiration templates (Documentary, Tech Announcement, Relaxed Greeting, Gaming Dialogue)
- Live character counter and real-time generation metrics

### 🧬 Zero-Shot Voice Cloning
- Upload a 10–60 second reference recording to create a reusable voice profile
- Profiles are cloned with **Coqui XTTS v2** — no fine-tuning, no training, just conditioning on the reference clip
- Consent verification required before any profile is created (timestamped record stored per profile)
- When cloning is unavailable, profiles fall back to neural voices with an explicit label — no silent substitution

### 🎛️ Voice Changer (Speech-to-Speech)
Two conversion modes:
- **Voice Morph (DSP):** Keeps original words, timing, and pauses. Shifts pitch using YIN estimation and transfers the target speaker's spectral envelope via STFT, preserving phase for intelligibility
- **Dub:** Transcribes the source audio with Whisper, then re-speaks the text in the target voice (neural or XTTS clone)
- Before/after spectral similarity is measured and displayed

### 🌍 Multilingual Translation & Dubbing
- Translate English speech or text into **5 languages**: Hindi, French, Japanese, Spanish, German
- Translation models: **Helsinki-NLP MarianMT** (opus-mt) for 4 languages, **Meta M2M-100** for Japanese
- Speak the translation with a native neural voice or clone it in your own voice cross-lingually
- Local-first: cloud translation (MyMemory API) is only used as an opt-in fallback

### 🤖 Conversational Voice Agents
- 4 interactive voice personas with spoken responses:
  - **Alex** — Tech Support Specialist
  - **Sarah** — Executive Career Coach
  - **Maya** — Language & Culture Tutor
  - **Leo** — Creative Storyteller & Roleplay
- Chat history with audio playback for each response

### 📚 Voice Library
- Visual gallery of all voice personas with gender, accent, role tags, and preview samples
- Cloned voice profiles appear alongside built-in voices

### 📜 Generation History
- Session-based log of every generated audio clip with timestamps, engine labels, and audio players

---

## 🏗️ Architecture

```
app.py      Controller & Streamlit UI orchestration
logic.py    Core engines: synthesis, cloning, ASR, translation, DSP, metrics, consent
ui.py       Presentation layer: styles, sidebar, cards, footer
tests/      31 automated tests (unittest + Streamlit AppTest)
```

### Engine Fallback Chain

The app probes each engine at startup (importability check + XTTS weight cache check) and labels every output with the engine that produced it:

```
Voice Profile selected
  ├─ XTTS v2 installed + enabled + model loaded + reference exists → XTTS zero-shot clone
  ├─ XTTS not installed                                            → Neural TTS (edge-tts) with explicit label
  ├─ User turned cloning off in sidebar                            → Neural TTS with explicit label
  ├─ XTTS model failed to load                                     → Neural TTS with explicit label
  └─ No reference audio on disk                                    → Neural TTS with explicit label
```

The XTTS model (~1.8 GB) is loaded lazily — only when the user enables the checkbox — and cached with `st.cache_resource` for the server's lifetime.

### Privacy Mode

A **Local only** toggle in the sidebar disables all cloud services (Google Web Speech for ASR, MyMemory for translation). When enabled, audio never leaves the machine.

---

## 🚀 Local Setup

### Prerequisites
- **Python 3.10** (XTTS v2 requires Python ≤ 3.11)
- **FFmpeg** installed and on PATH
- ~4 GB RAM available (for XTTS v2 model loading)

### Install & Run

```bash
git clone https://github.com/GaneshSarode/voice-clone-app.git
cd voice-clone-app

pip install -r requirements.txt

# For full voice cloning (optional, ~1.8 GB download on first use):
pip install TTS

streamlit run app.py
```

Open **http://localhost:8501** and enable **"Enable XTTS v2 cloning engine"** in the sidebar.

### System Packages (Linux / Streamlit Cloud)

Listed in `packages.txt`:
```
libsndfile1-dev
espeak-ng
ffmpeg
```

---

## 🧪 Tests

31 automated tests across three modules:

| Module | Tests | Coverage |
|---|---|---|
| `test_logic_core.py` | 18 | Consent gates, plan_synthesis fallback branches, profile CRUD, metrics, agent dialogue |
| `test_voice_conversion.py` | 7 | YIN pitch estimation, bounded pitch shifts, STFT envelope transfer, phase preservation |
| `test_app_smoke.py` | 6 | Headless Streamlit rendering, sidebar controls, consent UI, end-to-end synthesis |

```bash
python -m unittest discover -s tests -v
```

---

## 📦 Tech Stack

| Layer | Tools |
|---|---|
| **Voice Cloning** | Coqui XTTS v2 (zero-shot, CPU) |
| **Speech Recognition** | faster-whisper (int8 quantized CTranslate2) |
| **Neural TTS** | edge-tts (Microsoft Neural Voices) |
| **Translation** | Hugging Face Transformers (MarianMT, M2M-100) |
| **DSP & Audio** | Librosa (YIN, STFT, pitch shifting, mel spectrograms), SoundFile, FFmpeg/PyAV |
| **Metrics** | Real-time factor (RTF), latency, log-mel spectral similarity |
| **Frontend** | Streamlit |
| **Testing** | Python unittest, Streamlit AppTest |

---

## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit application
├── logic.py                # Core engine: synthesis, ASR, translation, DSP, metrics, consent
├── ui.py                   # UI components: styles, sidebar, voice cards, footer
├── requirements.txt        # Python dependencies
├── packages.txt            # System packages (Debian/Ubuntu)
├── runtime.txt             # Python version pin
├── .streamlit/
│   └── config.toml         # Streamlit server config (50 MB upload limit, no telemetry)
├── cloned_voices/          # Per-session voice profile storage
├── outputs/                # Generated audio (auto-cleaned after 120 min)
└── tests/
    ├── test_logic_core.py
    ├── test_voice_conversion.py
    └── test_app_smoke.py
```

---

## ⚖️ Responsible AI

- **Consent gate:** `save_cloned_voice_file()` raises `ConsentRequired` without explicit confirmation. Each profile stores a timestamped consent record with the declared owner
- **No silent substitution:** Every synthesis output is labeled with the engine that produced it. A fallback from cloning to neural TTS is always visible to the user
- **Per-session isolation:** Voice profiles are stored in session-scoped directories with sanitized paths
- **TTL cleanup:** Generated audio is automatically deleted after 120 minutes

---

## 📄 License

This project is for educational and personal use.

---

## 🙏 Acknowledgments

- [Coqui TTS](https://github.com/coqui-ai/TTS) — XTTS v2 zero-shot voice cloning
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 Whisper inference
- [edge-tts](https://github.com/rany2/edge-tts) — Microsoft Neural TTS
- [Hugging Face Transformers](https://huggingface.co/) — MarianMT & M2M-100 translation
- [Librosa](https://librosa.org/) — Audio analysis and DSP
- [Streamlit](https://streamlit.io/) — Web application framework
