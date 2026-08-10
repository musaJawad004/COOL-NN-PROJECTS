import tempfile
import unittest
from pathlib import Path

import numpy as np

from emotion_ai.audio import SAMPLE_COUNT, SAMPLE_RATE, audio_to_spectrogram, load_wav, normalize_audio, save_wav


class AudioTests(unittest.TestCase):
    def test_normalization_has_fixed_length(self):
        result = normalize_audio(np.ones(100, dtype=np.float32))
        self.assertEqual(result.shape, (SAMPLE_COUNT,))

    def test_spectrogram_shape(self):
        time = np.arange(SAMPLE_COUNT) / SAMPLE_RATE
        audio = np.sin(2 * np.pi * 440 * time).astype(np.float32)
        feature = audio_to_spectrogram(audio)
        self.assertEqual(tuple(feature.shape), (1, 64, 128))
        self.assertTrue(np.isfinite(feature.numpy()).all())

    def test_wav_round_trip(self):
        audio = np.random.default_rng(2).normal(0, 0.1, SAMPLE_COUNT).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            save_wav(path, audio)
            loaded = load_wav(path)
            self.assertEqual(loaded.shape, (SAMPLE_COUNT,))


if __name__ == "__main__":
    unittest.main()
