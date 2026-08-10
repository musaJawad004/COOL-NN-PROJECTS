"""Train the character-level phrase generator."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from model import TextCorrector, Vocabulary

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "content" / "training_pairs.json"
DEFAULT_CHECKPOINT = ROOT / "work" / "checkpoints" / "best_model.pt"


def make_typo(text: str, rng: random.Random) -> str:
    """Apply one plausible character-level typo."""
    if len(text) < 2:
        return text
    operation = rng.choice(("delete", "swap", "duplicate"))
    index = rng.randrange(len(text) - 1)
    if operation == "delete":
        return text[:index] + text[index + 1 :]
    if operation == "swap":
        return text[:index] + text[index + 1] + text[index] + text[index + 2 :]
    return text[:index] + text[index] + text[index:]


def load_pairs(path: Path, augmentations: int, seed: int) -> list[tuple[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    pairs = []
    rng = random.Random(seed)
    for row in raw:
        source, target = row["input"].strip().lower(), row["output"].strip().lower()
        if not source or not target:
            raise ValueError("Every training pair needs non-empty input and output text.")
        pairs.append((source, target))
        for _ in range(augmentations):
            noisy = source
            for _ in range(rng.randint(1, 3)):
                noisy = make_typo(noisy, rng)
            pairs.append((noisy, target))
    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], vocabulary: Vocabulary):
        self.items = [
            (
                torch.tensor(vocabulary.encode(source), dtype=torch.long),
                torch.tensor(vocabulary.encode(target), dtype=torch.long),
            )
            for source, target in pairs
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.items[index]


def collate_batch(batch, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    sources, targets = zip(*batch)
    return (
        pad_sequence(sources, batch_first=True, padding_value=pad_id),
        pad_sequence(targets, batch_first=True, padding_value=pad_id),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the gibberish-to-phrase neural network.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--augmentations", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    pairs = load_pairs(args.data, args.augmentations, args.seed)
    vocabulary = Vocabulary.build([text for pair in pairs for text in pair])
    dataset = PairDataset(pairs, vocabulary)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, vocabulary.pad_id),
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = TextCorrector(len(vocabulary), vocabulary.pad_id, args.embedding_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_id)
    best_loss = float("inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(f"Training {len(dataset)} examples on {device}; vocabulary: {len(vocabulary)} symbols")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for source, target in loader:
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, target, vocabulary.pad_id)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(loader)
        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocabulary": vocabulary.to_dict(),
                    "config": {"embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim},
                    "epoch": epoch,
                    "loss": best_loss,
                },
                args.checkpoint,
            )
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:>3}/{args.epochs} | loss {average_loss:.4f} | best {best_loss:.4f}")
    print(f"Saved best model to {args.checkpoint}")


if __name__ == "__main__":
    main()
