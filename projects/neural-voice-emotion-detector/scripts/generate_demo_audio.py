"""Generate deterministic synthetic audio for testing the training pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion_ai.audio import EMOTIONS, SAMPLE_COUNT, SAMPLE_RATE, save_wav  # noqa: E402


def syllable_envelope(time: np.ndarray, pace: float, phase: float) -> np.ndarray:
    pulse = np.maximum(0.0, np.sin(2 * np.pi * pace * time + phase))
    return 0.18 + 0.82 * pulse**1.7


def make_demo(emotion: str, index: int) -> np.ndarray:
    rng = np.random.default_rng(10_000 + EMOTIONS.index(emotion) * 100 + index)
    time = np.arange(SAMPLE_COUNT, dtype=np.float32) / SAMPLE_RATE
    variation = index - 4.5

    if emotion == "happy":
        frequency = 255 + variation * 3 + 28 * np.sin(2 * np.pi * 1.7 * time)
        phase = 2 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
        signal = 0.52 * np.sin(phase) + 0.20 * np.sin(phase * 2)
        signal *= syllable_envelope(time, 3.1, index * 0.2)
    elif emotion == "sad":
        frequency = 155 + variation * 2 - 18 * time / time[-1]
        phase = 2 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
        signal = (0.48 * np.sin(phase) + 0.10 * np.sin(phase * 2))
        signal *= syllable_envelope(time, 1.35, index * 0.15)
    elif emotion == "angry":
        frequency = 205 + variation * 4 + 12 * np.sin(2 * np.pi * 7 * time)
        phase = 2 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
        signal = 0.58 * np.sin(phase) + 0.27 * np.sin(phase * 2) + 0.14 * np.sin(phase * 3)
        signal = np.tanh(signal * 2.4)
        signal += rng.normal(0.0, 0.055, SAMPLE_COUNT)
        signal *= syllable_envelope(time, 4.0, index * 0.3)
    elif emotion == "calm":
        frequency = 185 + variation * 1.5 + 4 * np.sin(2 * np.pi * 0.55 * time)
        phase = 2 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
        signal = 0.5 * np.sin(phase) + 0.12 * np.sin(phase * 2)
        signal *= 0.72 + 0.12 * np.sin(2 * np.pi * 0.7 * time + index * 0.1)
    else:  # nervous
        jitter = rng.normal(0.0, 9.0, SAMPLE_COUNT)
        frequency = 230 + variation * 3 + 20 * np.sin(2 * np.pi * 6.5 * time) + jitter
        phase = 2 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
        signal = 0.42 * np.sin(phase) + 0.14 * np.sin(phase * 2)
        tremolo = 0.48 + 0.42 * np.maximum(0.0, np.sin(2 * np.pi * 7.5 * time + index))
        signal *= tremolo
        signal += rng.normal(0.0, 0.018, SAMPLE_COUNT)

    fade_samples = int(SAMPLE_RATE * 0.06)
    fade = np.linspace(0.0, 1.0, fade_samples)
    signal[:fade_samples] *= fade
    signal[-fade_samples:] *= fade[::-1]
    return np.asarray(signal, dtype=np.float32)


def main() -> None:
    for emotion in EMOTIONS:
        for index in range(10):
            destination = ROOT / "data" / emotion / f"demo_{emotion}_{index + 1:02d}.wav"
            save_wav(destination, make_demo(emotion, index))
    print(f"Generated {len(EMOTIONS) * 10} demo WAV files in {ROOT / 'data'}")


if __name__ == "__main__":
    main()
