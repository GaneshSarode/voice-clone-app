# tests/test_voice_conversion.py
"""Tests for the speech-to-speech voice morph in logic.py.

Stdlib unittest only. Run from the repository root:

    python -m unittest discover -s tests -t . -v
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import numpy as np
import soundfile

import logic


def _harmonic_tone(frequency, seconds, sample_rate=22050):
    """A voiced-like tone: a fundamental plus decaying harmonics."""
    t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)
    signal += 0.5 * np.sin(2 * np.pi * frequency * 2 * t)
    signal += 0.25 * np.sin(2 * np.pi * frequency * 3 * t)
    return (0.3 * signal / np.max(np.abs(signal))).astype(np.float32)


class VoiceConversionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name, samples, sample_rate=22050):
        path = os.path.join(self.root, name)
        soundfile.write(path, samples, sample_rate)
        return path

    def test_estimate_pitch_tracks_a_known_frequency(self):
        path = self._write("a.wav", _harmonic_tone(180.0, 1.0))
        samples, _ = librosa.load(path, sr=22050, mono=True)
        estimate = logic.estimate_pitch_hz(samples, 22050)
        self.assertIsNotNone(estimate)
        self.assertLess(abs(estimate - 180.0), 12.0)

    def test_morph_preserves_duration_and_moves_pitch_toward_target(self):
        source = self._write("source.wav", _harmonic_tone(150.0, 1.0))
        target = self._write("target.wav", _harmonic_tone(220.0, 0.8))

        result = logic.morph_voice(source, target)

        self.assertTrue(os.path.exists(result.output_path))
        # Timing is preserved because the waveform is transformed, not re-spoken.
        self.assertAlmostEqual(result.duration_s, 1.0, delta=0.05)
        # The shift is bounded and points from 150 Hz up toward 220 Hz.
        self.assertGreater(result.semitones, 3.0)
        self.assertLess(result.semitones, 8.0)
        self.assertEqual(result.sample_rate, logic.DEFAULT_CONVERSION_SR)

        converted, rate = soundfile.read(result.output_path)
        self.assertEqual(rate, logic.DEFAULT_CONVERSION_SR)
        self.assertGreater(converted.size, 0)

    def test_morph_rejects_audio_that_is_too_short(self):
        source = self._write("source.wav", _harmonic_tone(150.0, 0.05))
        target = self._write("target.wav", _harmonic_tone(200.0, 0.8))
        with self.assertRaises(ValueError):
            logic.morph_voice(source, target)


if __name__ == "__main__":
    unittest.main()
