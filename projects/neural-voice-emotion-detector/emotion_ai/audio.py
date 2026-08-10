"""WAV I/O and dependency-light log-mel feature extraction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import wavfile
import torch
from torch.nn import functional as F


EMOTIONS = ("happy", "sad", "angry", "calm", "nervous")
SAMPLE_RATE = 16_000
DURATION_SECONDS = 3
SAMPLE_COUNT = SAMPLE_RATE * DURATION_SECONDS
N_FFT = 512
HOP_LENGTH = 160
MEL_BINS = 64
FEATURE_FRAMES = 128


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        return np.zeros(SAMPLE_COUNT, dtype=np.float32)
    audio = audio - float(audio.mean())
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio / peak
    if len(audio) < SAMPLE_COUNT:
        audio = np.pad(audio, (0, SAMPLE_COUNT - len(audio)))
    else:
        audio = audio[:SAMPLE_COUNT]
    return audio.astype(np.float32)


def save_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = np.clip(normalize_audio(audio), -1.0, 1.0)
    wavfile.write(path, SAMPLE_RATE, (normalized * 32767).astype(np.int16))


def load_wav(path: Path) -> np.ndarray:
    sample_rate, audio = wavfile.read(path)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        old_positions = np.linspace(0.0, 1.0, len(audio), endpoint=False)
        new_length = max(1, round(len(audio) * SAMPLE_RATE / sample_rate))
        new_positions = np.linspace(0.0, 1.0, new_length, endpoint=False)
        audio = np.interp(new_positions, old_positions, audio).astype(np.float32)
    return normalize_audio(audio)


def hz_to_mel(frequency: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + frequency / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


@lru_cache(maxsize=1)
def mel_filterbank() -> torch.Tensor:
    minimum = hz_to_mel(torch.tensor(20.0))
    maximum = hz_to_mel(torch.tensor(SAMPLE_RATE / 2.0))
    mel_points = torch.linspace(minimum, maximum, MEL_BINS + 2)
    frequencies = mel_to_hz(mel_points)
    bins = torch.floor((N_FFT + 1) * frequencies / SAMPLE_RATE).long()
    filters = torch.zeros(MEL_BINS, N_FFT // 2 + 1)
    for index in range(MEL_BINS):
        left, center, right = int(bins[index]), int(bins[index + 1]), int(bins[index + 2])
        if center > left:
            filters[index, left:center] = torch.linspace(0.0, 1.0, center - left)
        if right > center:
            filters[index, center:right] = torch.linspace(1.0, 0.0, right - center)
    return filters


def audio_to_spectrogram(audio: np.ndarray) -> torch.Tensor:
    waveform = torch.from_numpy(normalize_audio(audio))
    spectrum = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=torch.hann_window(N_FFT),
        return_complex=True,
    ).abs().pow(2)
    mel = torch.matmul(mel_filterbank(), spectrum)
    log_mel = torch.log1p(mel)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    resized = F.interpolate(
        log_mel.unsqueeze(0).unsqueeze(0),
        size=(MEL_BINS, FEATURE_FRAMES),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def augment_audio(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = normalize_audio(audio).copy()
    result *= rng.uniform(0.7, 1.1)
    result = np.roll(result, int(rng.integers(-1_600, 1_601)))
    noise_level = rng.uniform(0.0, 0.018)
    result += rng.normal(0.0, noise_level, result.shape).astype(np.float32)
    return np.clip(result, -1.0, 1.0)
