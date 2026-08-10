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


def training_dataset_files(data_root: Path) -> tuple[dict[str, list[Path]], str]:
    """Prefer genuine speech and exclude demo tones once every label has speech."""
    all_files = dataset_files(data_root)
    real_files = {
        emotion: [path for path in paths if not path.name.startswith("demo_")]
        for emotion, paths in all_files.items()
    }
    if all(len(paths) >= 2 for paths in real_files.values()):
        return real_files, "human speech"
    return all_files, "demo audio"


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
    progress: Callable[[int, float, float, float], None] | None = None,
) -> tuple[EmotionCNN, dict]:
    files, data_source = training_dataset_files(data_root)
    validate_dataset(files)
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    training_items = []
    validation_items = []
    personal_sample_count = 0
    for emotion, paths in files.items():
        label = EMOTIONS.index(emotion)
        if data_source == "demo audio":
            shuffled = list(paths)
            rng.shuffle(shuffled)
            validation_count = max(1, round(len(shuffled) * 0.2))
            validation_items.extend((path, label) for path in shuffled[:validation_count])
            training_items.extend((path, label) for path in shuffled[validation_count:])
            continue
        public = [path for path in paths if path.name.startswith("ravdess_")]
        personal = [path for path in paths if not path.name.startswith(("ravdess_", "demo_"))]
        rng.shuffle(public)
        validation_count = max(1, round(len(public) * 0.2))
        validation_items.extend((path, label) for path in public[:validation_count])
        training_items.extend((path, label) for path in public[validation_count:])
        # Personal microphone samples are the most valuable domain adaptation
        # data. Keep all of them in training and repeat them without duplicating files.
        personal_sample_count += len(personal)
        for path in personal:
            training_items.extend([(path, label)] * 5)
    model = EmotionCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.0001)
    class_counts = np.bincount([label for _, label in training_items], minlength=len(EMOTIONS))
    class_weights = class_counts.sum() / (len(EMOTIONS) * np.maximum(class_counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    best_accuracy = 0.0
    best_state = None
    validation_features = [
        (audio_to_spectrogram(load_wav(path)), label)
        for path, label in validation_items
    ]
    audio_cache = {path: load_wav(path) for path, _ in training_items}

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(training_items)
        losses = []
        correct = 0
        total = 0
        epoch_features = []
        epoch_labels = []
        for path, label in training_items:
            audio = audio_cache[path]
            versions = (audio, augment_audio(audio, rng))
            epoch_features.extend(audio_to_spectrogram(version) for version in versions)
            epoch_labels.extend((label, label))
        order = rng.permutation(len(epoch_labels))
        for start in range(0, len(order), 32):
            indices = order[start : start + 32]
            features = torch.stack([epoch_features[int(index)] for index in indices])
            labels = torch.tensor([epoch_labels[int(index)] for index in indices], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += len(labels)
        training_accuracy = correct / max(total, 1)
        average_loss = sum(losses) / max(len(losses), 1)
        model.eval()
        validation_correct = 0
        with torch.inference_mode():
            for feature, label in validation_features:
                prediction = int(model(feature.unsqueeze(0)).argmax(dim=1).item())
                validation_correct += int(prediction == label)
        validation_accuracy = validation_correct / max(len(validation_features), 1)
        if validation_accuracy >= best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if progress:
            progress(epoch, average_loss, training_accuracy, validation_accuracy)

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    validation_logits = []
    validation_labels = []
    with torch.inference_mode():
        for feature, label in validation_features:
            validation_logits.append(model(feature.unsqueeze(0)).squeeze(0))
            validation_labels.append(label)
    stacked_logits = torch.stack(validation_logits)
    stacked_labels = torch.tensor(validation_labels, dtype=torch.long)
    # Simple held-out temperature scaling prevents overconfident 100% outputs.
    candidates = torch.linspace(0.75, 5.0, 86)
    losses = [
        float(nn.functional.cross_entropy(stacked_logits / candidate, stacked_labels).item())
        for candidate in candidates
    ]
    temperature = float(candidates[int(np.argmin(losses))].item())
    model.temperature = temperature
    metadata = {
        "epochs": epochs,
        "validation_accuracy": best_accuracy,
        "sample_count": sum(len(paths) for paths in files.values()),
        "training_samples": len(training_items),
        "validation_samples": len(validation_items),
        "data_source": data_source,
        "personal_samples": personal_sample_count,
        "temperature": temperature,
    }
    model.save(checkpoint, metadata)
    model.eval()
    return model, metadata


@torch.inference_mode()
def predict(model: EmotionCNN, audio: np.ndarray) -> dict[str, float]:
    feature = audio_to_spectrogram(audio).unsqueeze(0)
    temperature = max(float(getattr(model, "temperature", 1.0)), 0.1)
    probabilities = torch.softmax(model(feature) / temperature, dim=1).squeeze(0)
    return {emotion: float(probabilities[index]) for index, emotion in enumerate(EMOTIONS)}
