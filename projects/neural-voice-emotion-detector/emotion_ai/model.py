"""Convolutional neural network for log-mel emotion classification."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .audio import EMOTIONS


class EmotionCNN(nn.Module):
    def __init__(self, class_count: int = len(EMOTIONS)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, class_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(features))

    def save(self, path: Path, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.state_dict(), "emotions": EMOTIONS, "metadata": metadata or {}}, path)

    @classmethod
    def load(cls, path: Path) -> tuple["EmotionCNN", dict]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if tuple(payload["emotions"]) != EMOTIONS:
            raise ValueError("Checkpoint emotion labels do not match this application.")
        model = cls()
        model.load_state_dict(payload["model_state"])
        model.eval()
        return model, payload.get("metadata", {})
