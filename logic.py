# logic.py - Voice Cloning, Neural Synthesis & Platform Engine
import os
import json
import asyncio
import tempfile
import edge_tts
import librosa
from pathlib import Path

CLONED_VOICES_DIR = Path("cloned_voices")
CLONED_VOICES_DIR.mkdir(exist_ok=True)
METADATA_FILE = CLONED_VOICES_DIR / "voices_metadata.json"

# Curated ElevenLabs Style Voice Library
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
        "avatar": "👩‍💼"
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
        "avatar": "🎙️"
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
        "avatar": "👨‍💼"
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
        "avatar": "✨"
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
        "avatar": "👨‍💻"
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
        "avatar": "👩‍🎓"
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
        "avatar": "🌸"
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
        "avatar": "📻"
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
        "avatar": "🥐"
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
        "avatar": "🎨"
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
        "avatar": "🎐"
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
        "avatar": "🎮"
    }
}


def load_cloned_voices_metadata():
    """Load metadata of user-cloned custom voices."""
    if not METADATA_FILE.exists():
        return {}
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cloned_voice_file(name, audio_bytes, description="", role="Custom Voice", gender="Unknown", accent="Custom"):
    """Save an uploaded voice recording and register it in the voice library."""
    voice_id = f"custom_{name.lower().replace(' ', '_')}_{int(len(audio_bytes))}"
    file_path = CLONED_VOICES_DIR / f"{voice_id}.wav"

    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    metadata = load_cloned_voices_metadata()
    metadata[voice_id] = {
        "voice_id": voice_id,
        "name": name,
        "gender": gender,
        "accent": accent,
        "role": role,
        "desc": description or f"Custom cloned voice generated from uploaded audio sample.",
        "audio_path": str(file_path),
        "is_cloned": True,
        "avatar": "🧬"
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata[voice_id]


def delete_cloned_voice(voice_id):
    """Delete a cloned voice from disk and metadata."""
    metadata = load_cloned_voices_metadata()
    if voice_id in metadata:
        audio_path = Path(metadata[voice_id].get("audio_path", ""))
        if audio_path.exists():
            try:
                audio_path.unlink()
            except Exception:
                pass
        del metadata[voice_id]
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return True
    return False


async def _generate_edge_tts(text, voice_name, rate_str, output_path):
    communicate = edge_tts.Communicate(text, voice_name, rate=rate_str)
    await communicate.save(output_path)


def synthesize_neural_speech(text, edge_voice, speed_factor=1.0, output_path="output.mp3"):
    """Generate high-quality ElevenLabs-grade speech using Neural TTS."""
    rate_percent = int((speed_factor - 1.0) * 100)
    rate_str = f"{'+' if rate_percent >= 0 else ''}{rate_percent}%"

    asyncio.run(_generate_edge_tts(text, edge_voice, rate_str, output_path))
    return output_path


def synthesize_xtts_clone(xtts_model, text, speaker_wav_path, language="en", output_path="cloned_output.wav"):
    """Generate cloned speech using Coqui XTTS v2 with user's voice reference."""
    xtts_model.tts_to_file(
        text=text,
        speaker_wav=speaker_wav_path,
        language=language,
        file_path=output_path,
        split_sentences=True
    )
    return output_path


def transcribe_audio_file(asr_pipeline, audio_file):
    """Transcribe uploaded audio file to English text using Whisper ASR or Google Web Speech."""
    audio_bytes = audio_file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name

    # 1. Try local Whisper pipeline if available
    if asr_pipeline is not None:
        try:
            audio, _ = librosa.load(temp_path, sr=16000)
            result = asr_pipeline(audio)
            text = result.get("text", "").strip()
            if text:
                return text, temp_path
        except Exception:
            pass

    # 2. Cloud-native SpeechRecognition (Google Web Speech API - free & fast)
    try:
        import speech_recognition as sr
        data, sr_rate = librosa.load(temp_path, sr=16000, mono=True)
        converted_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        soundfile.write(converted_wav, data, 16000)

        recognizer = sr.Recognizer()
        with sr.AudioFile(converted_wav) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        try:
            os.remove(converted_wav)
        except Exception:
            pass

        if text and text.strip():
            return text.strip(), temp_path
    except Exception as e:
        print(f"Speech recognition error: {e}")

    # 3. Fallback message only if recognition could not detect clear words
    return "Could not transcribe audio clearly. Please ensure your recording is clear and audible.", temp_path


def translate_text(translator_pipeline, text, target_lang_code="hi"):
    """Translate text to target language using HuggingFace or fast cloud API fallback."""
    if translator_pipeline is not None:
        try:
            res = translator_pipeline(text)
            return res[0]["translation_text"]
        except Exception:
            pass

    # Cloud lightweight fallback (uses zero RAM, works everywhere)
    try:
        import urllib.request
        import urllib.parse
        import json
        url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(text)}&langpair=en|{target_lang_code}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            translated = data.get("responseData", {}).get("translatedText")
            if translated:
                return translated
    except Exception:
        pass

    return text


# Voice Agent Personalities
AGENT_PERSONAS = {
    "Alex (Tech Support Specialist)": {
        "voice": "en-US-ChristopherNeural",
        "intro": "Hello! I am Alex, your Technical Support Engineer. How can I assist you with your systems today?",
        "prompt": "You are a friendly, highly competent Senior Technical Support Engineer who troubleshoots software, cloud, and AI systems with clear, concise guidance."
    },
    "Sarah (Executive Career Coach)": {
        "voice": "en-US-JennyNeural",
        "intro": "Welcome! I'm Sarah, your Executive Career Coach. Let's discuss your career goals, interview preparation, or leadership strategies.",
        "prompt": "You are an empathetic, insightful Executive Career and Leadership Coach providing high-impact professional advice."
    },
    "Maya (Language & Culture Tutor)": {
        "voice": "en-GB-SoniaNeural",
        "intro": "Hello there! I'm Maya, your language and cultural tutor. Which language or dialect would you like to explore today?",
        "prompt": "You are a warm, multilingual educator helping students master pronunciation, idioms, and natural conversation."
    },
    "Leo (Creative Storyteller & Roleplay)": {
        "voice": "en-GB-RyanNeural",
        "intro": "Greetings adventurer! I am Leo, your master storyteller. Give me a premise or world, and let's craft an epic tale together.",
        "prompt": "You are an imaginative fantasy storyteller who speaks with rich descriptive prose and immersive roleplay."
    }
}


def generate_agent_response(persona_name, user_input, conversation_history=None):
    """Generate an intelligent conversational response for the Voice Agent."""
    persona = AGENT_PERSONAS.get(persona_name, AGENT_PERSONAS["Alex (Tech Support Specialist)"])
    
    # Smart conversational generation logic
    user_lower = user_input.lower()
    
    if any(greet in user_lower for greet in ["hello", "hi", "hey", "greetings"]):
        response = f"{persona['intro']} What is on your mind?"
    elif "who are you" in user_lower or "what can you do" in user_lower:
        response = f"I am {persona_name.split(' ')[0]}, an AI Voice Agent powered by ElevenVoice. I can engage in live natural voice dialogues, answer technical or creative questions, and explain complex concepts."
    elif "interview" in user_lower:
        response = "The key to acing any interview is the STAR method: Situation, Task, Action, and Result. Make sure your narrative highlights your direct contribution and quantifiable impact."
    elif "voice clone" in user_lower or "elevenlabs" in user_lower:
        response = "ElevenLabs pioneered instant voice cloning by extracting speaker embeddings from a brief audio reference, allowing seamless text synthesis while preserving unique vocal timbre."
    elif "python" in user_lower or "code" in user_lower or "bug" in user_lower:
        response = "When debugging in Python, always isolate variables, inspect stack traces carefully, and leverage type annotations and unit tests to prevent silent regressions."
    else:
        response = f"That is a great inquiry. Regarding '{user_input.strip()}', the most effective approach is to examine the core parameters, test iteratively with small benchmarks, and scale based on verified metrics. Would you like me to elaborate on the details?"

    return response, persona["voice"]
