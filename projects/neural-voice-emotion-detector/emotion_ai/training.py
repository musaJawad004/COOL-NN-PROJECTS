"""Dataset discovery, augmented training, and prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from .audio import EMOTIONS, audio_to_spectrogram, augment_audio, load_wav
from .model import EmotionCNN


def dataset_files(data_root: Path) -> dict[str, list[Path]]:
    return {emotion: sorted((data_root / emotion).glob("*.wav")) for emotion in EMOTIONS}


def validate_dataset(files: dict[str, list[Path]]) -> None:
    missing = [emotion for emotion, paths in files.items() if len(paths) < 2]
    if missing:
        raise ValueError(
            "Record at least two samples for every emotion. Still needed: "
            + ", ".join(missing)
            + "."
        )


def train_model(
    data_root: Path,
    checkpoint: Path,
    epochs: int = 35,
    progress: Callable[[int, float, float], None] | None = None,
) -> tuple[EmotionCNN, dict]:
    files = dataset_files(data_root)
    validate_dataset(files)
    raw_items = [(path, EMOTIONS.index(emotion)) for emotion, paths in files.items() for path in paths]
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    model = EmotionCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(raw_items)
        losses = []
        correct = 0
        total = 0
        for path, label in raw_items:
            audio = load_wav(path)
            versions = (audio, augment_audio(audio, rng))
            features = torch.stack([audio_to_spectrogram(version) for version in versions])
            labels = torch.tensor([label, label], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += len(labels)
        accuracy = correct / max(total, 1)
        average_loss = sum(losses) / max(len(losses), 1)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if progress:
            progress(epoch, average_loss, accuracy)

    assert best_state is not None
    model.load_state_dict(best_state)
    metadata = {"epochs": epochs, "training_accuracy": best_accuracy, "sample_count": len(raw_items)}
    model.save(checkpoint, metadata)
    model.eval()
    return model, metadata


@torch.inference_mode()
def predict(model: EmotionCNN, audio: np.ndarray) -> dict[str, float]:
    feature = audio_to_spectrogram(audio).unsqueeze(0)
    probabilities = torch.softmax(model(feature), dim=1).squeeze(0)
    return {emotion: float(probabilities[index]) for index, emotion in enumerate(EMOTIONS)}
