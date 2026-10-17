# tests/test_app_smoke.py
"""End-to-end smoke tests for the Streamlit app (streamlit.testing.v1.AppTest).

These run the real app script in-process. The text-to-speech test performs an
actual network synthesis, so it is an integration test: if edge-tts is
unreachable it is skipped rather than failed.

Run from the repository root:

    python -m unittest discover -s tests -t . -v
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

import logic  # noqa: E402

APP_PATH = str(ROOT / "app.py")


def run_app(timeout=180):
    at = AppTest.from_file(APP_PATH, default_timeout=timeout)
    at.run()
    return at


class AppSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(ROOT)

    def test_app_renders_without_exception(self):
        at = run_app()
        self.assertEqual(list(at.exception), [], "app raised during initial render")
        self.assertEqual(len(at.tabs), 8, "expected eight top-level sections")

    def test_sidebar_exposes_engine_and_privacy_controls(self):
        at = run_app()
        labels = [c.label.lower() for c in at.checkbox]
        self.assertTrue(any("cloning engine" in label for label in labels), labels)
        self.assertTrue(any("local only" in label for label in labels), labels)

    def test_capabilities_are_reported_to_the_user(self):
        at = run_app()
        markup = " ".join(m.value for m in at.sidebar.markdown)
        for expected in (
            "Neural TTS (edge-tts)",
            "Voice cloning (XTTS v2)",
            "Local ASR (faster-whisper)",
        ):
            self.assertIn(expected, markup)

    def test_local_only_toggle_rerenders_cleanly(self):
        at = run_app()
        box = next(c for c in at.checkbox if "local only" in c.label.lower())
        box.set_value(True).run()
        self.assertEqual(list(at.exception), [])

    def test_voice_profile_save_gate(self):
        """The save button must not create a profile without a reference sample, and
        the consent gate must be present with the hint text owned by the domain."""
        at = AppTest.from_file(APP_PATH, default_timeout=180).run()

        consent_labels = [c.label for c in at.checkbox]
        self.assertIn(
            logic.CONSENT_REQUIRED_HINT,
            consent_labels,
            "the consent gate is not rendered on the Voice Profiles tab",
        )

        at.text_input[0].set_value("Gate Test").run()
        save_button = next(b for b in at.button if "Save Voice Profile" in b.label)
        save_button.click().run()

        self.assertEqual(list(at.exception), [])
        warnings = " ".join(w.value for w in at.warning)
        self.assertIn("upload a reference audio sample", warnings)

        user_id = at.session_state.user_id
        self.assertEqual(
            logic.load_cloned_voices_metadata(user_id),
            {},
            "no voice profile should be created by a blocked save",
        )

    def test_text_to_speech_generates_and_records_audio(self):
        at = AppTest.from_file(APP_PATH, default_timeout=300).run()
        at.text_area[0].set_value("This is a short synthesis integration test.").run()

        generate = next(b for b in at.button if "Generate Speech" in b.label)
        generate.click().run()

        self.assertEqual(list(at.exception), [], "generation raised")
        if at.error:
            self.skipTest(f"network synthesis unavailable: {at.error[0].value}")

        history = at.session_state.history
        self.assertEqual(len(history), 1, "generation was not recorded in history")
        entry = history[0]
        self.assertTrue(os.path.exists(entry["file_path"]), entry["file_path"])
        self.assertGreater(entry["metrics"]["latency_ms"], 0)
        self.assertFalse(entry["engine"].lower().startswith("xtts"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
