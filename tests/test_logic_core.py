# tests/test_logic_core.py
"""Phase 1 tests for the VoxStudio engine layer.

Stdlib unittest only (no new dependency). Run from the repository root:

    python -m unittest discover -s tests -t . -v
"""
import io
import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile

import logic


def wav_bytes(seconds=2.0, sample_rate=16000, frequency=220.0):
    """A small decodable WAV payload for upload-path tests."""
    t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
    samples = (0.2 * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)
    buffer = io.BytesIO()
    soundfile.write(buffer, samples, sample_rate, format="WAV")
    return buffer.getvalue()


class _StubASR:
    """Stands in for faster-whisper's WhisperModel.transcribe()."""

    def transcribe(self, path, **kwargs):
        class _Segment:
            text = " hello world"

        class _Info:
            language = "en"
            language_probability = 0.99

        return [_Segment()], _Info()


class LogicCoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._saved_cloned = logic.CLONED_VOICES_DIR
        self._saved_output = logic.OUTPUT_DIR
        logic.CLONED_VOICES_DIR = root / "cloned_voices"
        logic.OUTPUT_DIR = root / "outputs"
        logic.CLONED_VOICES_DIR.mkdir(parents=True, exist_ok=True)
        logic.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.user = "unittest_user"

    def tearDown(self):
        logic.CLONED_VOICES_DIR = self._saved_cloned
        logic.OUTPUT_DIR = self._saved_output
        self._tmp.cleanup()

    # ---- capabilities -------------------------------------------------
    def test_detect_capabilities_reports_expected_keys(self):
        caps = logic.detect_capabilities()
        keys = {c.key for c in caps}
        self.assertEqual(
            keys, {"neural_tts", "cloning", "asr_local", "translation_local"}
        )
        for cap in caps:
            self.assertTrue(cap.label)
            self.assertTrue(cap.detail)
            self.assertIsInstance(cap.available, bool)

    # ---- synthesis planning -------------------------------------------
    def test_plan_preset_uses_neural(self):
        plan = logic.plan_synthesis(
            {"is_cloned": False, "edge_voice": "en-US-JennyNeural"},
            cloning_enabled=True,
            cloning_loaded=True,
        )
        self.assertEqual(plan.engine, "neural")
        self.assertFalse(plan.is_true_clone)

    def test_plan_clone_without_reference_falls_back_and_explains(self):
        plan = logic.plan_synthesis(
            {"is_cloned": True, "audio_path": "does-not-exist.wav"},
            cloning_enabled=True,
            cloning_loaded=True,
        )
        self.assertEqual(plan.engine, "neural")
        self.assertIn("no usable reference", plan.reason)

    def test_plan_clone_engine_disabled_is_explicit(self):
        reference = self._write_reference()
        plan = logic.plan_synthesis(
            {"is_cloned": True, "audio_path": reference},
            cloning_enabled=False,
            cloning_loaded=False,
        )
        self.assertEqual(plan.engine, "neural")
        self.assertIn("turned off", plan.reason)

    def test_plan_clone_engine_unavailable_is_explicit(self):
        reference = self._write_reference()
        plan = logic.plan_synthesis(
            {"is_cloned": True, "audio_path": reference},
            cloning_enabled=True,
            cloning_loaded=False,
        )
        self.assertEqual(plan.engine, "neural")
        self.assertIn("unavailable", plan.label)
        self.assertIn("could not be loaded", plan.reason)

    def test_plan_true_clone_when_engine_ready(self):
        reference = self._write_reference()
        plan = logic.plan_synthesis(
            {"is_cloned": True, "audio_path": reference},
            cloning_enabled=True,
            cloning_loaded=True,
        )
        self.assertEqual(plan.engine, "xtts")
        self.assertTrue(plan.is_true_clone)
        self.assertEqual(plan.reference_path, reference)

    def _write_reference(self):
        path = logic.CLONED_VOICES_DIR / "ref.wav"
        soundfile.write(str(path), np.zeros(16000, dtype=np.float32), 16000)
        return str(path)

    # ---- consent ------------------------------------------------------
    def test_consent_required_without_confirmation(self):
        with self.assertRaises(logic.ConsentRequired):
            logic.save_cloned_voice_file("No Consent", wav_bytes(), user_id=self.user)

    def test_save_records_consent_and_normalises_audio(self):
        profile = logic.save_cloned_voice_file(
            "Consented",
            wav_bytes(seconds=3.0),
            user_id=self.user,
            consent_confirmed=True,
            consent_note="my own recording",
            consent_owner="self",
        )
        self.assertTrue(profile["consent"]["confirmed"])
        self.assertEqual(profile["consent"]["declared_owner"], "self")
        self.assertIn("confirmed_at", profile["consent"])
        self.assertAlmostEqual(profile["duration_s"], 3.0, places=1)
        self.assertTrue(logic.profile_has_consent(profile))

        # Stored as real 16 kHz mono WAV, regardless of upload format.
        data, rate = soundfile.read(profile["audio_path"])
        self.assertEqual(rate, 16000)
        self.assertEqual(data.ndim, 1)

        # Round-trips through the metadata store.
        stored = logic.load_cloned_voices_metadata(self.user)
        self.assertIn(profile["voice_id"], stored)
        self.assertTrue(logic.profile_has_consent(stored[profile["voice_id"]]))

    def test_save_rejects_too_short_audio(self):
        with self.assertRaises(ValueError):
            logic.save_cloned_voice_file(
                "Too Short",
                wav_bytes(seconds=0.2),
                user_id=self.user,
                consent_confirmed=True,
            )

    def test_save_rejects_undecodable_audio(self):
        with self.assertRaises(ValueError):
            logic.save_cloned_voice_file(
                "Garbage",
                b"not audio at all",
                user_id=self.user,
                consent_confirmed=True,
            )

    def test_delete_removes_profile_and_file(self):
        profile = logic.save_cloned_voice_file(
            "Deletable", wav_bytes(), user_id=self.user, consent_confirmed=True
        )
        self.assertTrue(os.path.exists(profile["audio_path"]))
        self.assertTrue(logic.delete_cloned_voice(profile["voice_id"], user_id=self.user))
        self.assertFalse(os.path.exists(profile["audio_path"]))
        self.assertNotIn(profile["voice_id"], logic.load_cloned_voices_metadata(self.user))

    def test_legacy_profile_without_consent_is_flagged(self):
        legacy = {"voice_id": "old", "name": "Legacy"}
        self.assertFalse(logic.profile_has_consent(legacy))

    # ---- storage side effects -----------------------------------------
    def test_loading_metadata_has_no_filesystem_side_effect(self):
        session = "never_saved_session"
        self.assertFalse((logic.CLONED_VOICES_DIR / session).exists())
        self.assertEqual(logic.load_cloned_voices_metadata(session), {})
        self.assertFalse(
            (logic.CLONED_VOICES_DIR / session).exists(),
            "loading metadata must not create a session directory",
        )

    def test_saving_creates_the_session_directory_on_demand(self):
        profile = logic.save_cloned_voice_file(
            "On Demand", wav_bytes(), user_id=self.user, consent_confirmed=True
        )
        self.assertTrue((logic.CLONED_VOICES_DIR / self.user).is_dir())
        self.assertTrue(os.path.exists(profile["audio_path"]))

    # ---- fallback voice -----------------------------------------------
    def test_fallback_voice_matches_declared_gender(self):
        self.assertEqual(logic.fallback_edge_voice({"gender": "Female"}), "en-US-JennyNeural")
        self.assertEqual(logic.fallback_edge_voice({"gender": "Male"}), "en-US-ChristopherNeural")
        self.assertEqual(logic.fallback_edge_voice({}), logic.DEFAULT_FALLBACK_EDGE_VOICE)

    def test_prepare_reference_wav_normalises_to_16k_mono(self):
        source = logic.CLONED_VOICES_DIR / "source_44k.wav"
        t = np.linspace(0.0, 1.5, int(44100 * 1.5), endpoint=False)
        soundfile.write(str(source), (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), 44100)

        prepared = logic.prepare_reference_wav(str(source))
        data, rate = soundfile.read(prepared)
        self.assertEqual(rate, 16000)
        self.assertEqual(data.ndim, 1)
        self.assertAlmostEqual(len(data) / 16000, 1.5, places=1)

    # ---- translation --------------------------------------------------
    def test_translation_result_behaves_like_a_string(self):
        result = logic.TranslationResult("bonjour", "test")
        self.assertIsInstance(result, str)
        self.assertEqual(result.backend, "test")
        self.assertEqual(f"{result}!", "bonjour!")
        self.assertEqual(result.upper(), "BONJOUR")

    def test_translate_without_cloud_does_not_call_network(self):
        result = logic.translate_text(None, "hello world", "fr", allow_cloud=False)
        self.assertEqual(str(result), "hello world")
        self.assertIn("cloud disabled", result.backend)

    # ---- transcription ------------------------------------------------
    def test_transcription_result_unpacks_as_legacy_tuple(self):
        text, audio_path, error = logic.TranscriptionResult(
            text="hi", audio_path="p.wav", error=None
        )
        self.assertEqual((text, audio_path, error), ("hi", "p.wav", None))

    def test_transcribe_reports_local_backend_and_language(self):
        result = logic.transcribe_audio_file(_StubASR(), io.BytesIO(wav_bytes()))
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.backend, "faster-whisper (local)")
        self.assertEqual(result.language, "en")
        self.assertIsNone(result.error)

    def test_transcribe_cloud_disabled_reports_clearly(self):
        result = logic.transcribe_audio_file(None, io.BytesIO(wav_bytes()), allow_cloud=False)
        self.assertIsNone(result.text)
        self.assertEqual(result.backend, "none")
        self.assertIn("cloud transcription is disabled", result.error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
