import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import subprocess
import sys

from emotion_ai.audio import SAMPLE_COUNT
from emotion_ai.model import EmotionCNN
from emotion_ai.training import predict, training_dataset_files, validate_dataset


class ModelTests(unittest.TestCase):
    def test_forward_shape(self):
        model = EmotionCNN()
        output = model(torch.zeros(2, 1, 64, 128))
        self.assertEqual(tuple(output.shape), (2, 5))

    def test_prediction_probabilities_sum_to_one(self):
        probabilities = predict(EmotionCNN().eval(), np.zeros(SAMPLE_COUNT, dtype=np.float32))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)

    def test_temperature_reduces_overconfidence(self):
        model = EmotionCNN().eval()
        audio = np.zeros(SAMPLE_COUNT, dtype=np.float32)
        original = max(predict(model, audio).values())
        model.temperature = 5.0
        calibrated = max(predict(model, audio).values())
        self.assertLessEqual(calibrated, original)

    def test_training_worker_has_json_helpful_interface(self):
        result = subprocess.run(
            [sys.executable, "-m", "emotion_ai.train_worker", "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--checkpoint", result.stdout)

    def test_checkpoint_round_trip(self):
        model = EmotionCNN()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save(path, {"sample_count": 10})
            restored, metadata = EmotionCNN.load(path)
            self.assertIsInstance(restored, EmotionCNN)
            self.assertEqual(metadata["sample_count"], 10)

    def test_training_requires_every_emotion(self):
        incomplete = {emotion: [] for emotion in ("happy", "sad", "angry", "calm", "nervous")}
        incomplete["happy"] = [Path("one.wav"), Path("two.wav")]
        with self.assertRaisesRegex(ValueError, "every emotion"):
            validate_dataset(incomplete)

    def test_real_speech_replaces_demo_tones_when_balanced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for emotion in ("happy", "sad", "angry", "calm", "nervous"):
                folder = root / emotion
                folder.mkdir()
                for name in ("demo_01.wav", "real_01.wav", "real_02.wav"):
                    (folder / name).touch()
            files, source = training_dataset_files(root)
            self.assertEqual(source, "human speech")
            self.assertTrue(all(len(paths) == 2 for paths in files.values()))


if __name__ == "__main__":
    unittest.main()
