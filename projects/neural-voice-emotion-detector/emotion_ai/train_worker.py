"""Subprocess entry point for crash-isolated neural network training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import torch

from .training import train_model


def emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    warnings.filterwarnings("ignore", message="Chunk .* not understood")

    def progress(epoch: int, loss: float, training_accuracy: float, validation_accuracy: float) -> None:
        emit(
            {
                "event": "progress",
                "epoch": epoch,
                "loss": loss,
                "training_accuracy": training_accuracy,
                "validation_accuracy": validation_accuracy,
            }
        )

    try:
        _, metadata = train_model(
            args.data,
            args.checkpoint,
            epochs=args.epochs,
            progress=progress,
        )
        emit({"event": "trained", "metadata": metadata})
    except Exception as error:
        emit({"event": "error", "message": str(error)})
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
