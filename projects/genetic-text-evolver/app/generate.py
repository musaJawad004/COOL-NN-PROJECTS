"""Load a trained checkpoint and generate corrected text."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import TextCorrector, Vocabulary

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "work" / "checkpoints" / "best_model.pt"


class PhraseGenerator:
    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT):
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No trained model at {checkpoint_path}. Run: python train.py --epochs 80"
            )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.vocabulary = Vocabulary.from_dict(payload["vocabulary"])
        config = payload["config"]
        self.model = TextCorrector(
            len(self.vocabulary),
            self.vocabulary.pad_id,
            config["embedding_dim"],
            config["hidden_dim"],
        )
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()

    @torch.inference_mode()
    def generate(self, text: str, max_characters: int = 160) -> str:
        cleaned = text.strip().lower()
        if not cleaned:
            return ""
        source = torch.tensor([self.vocabulary.encode(cleaned)], dtype=torch.long)
        encoder_outputs, hidden = self.model.encode(source)
        source_mask = source.ne(self.vocabulary.pad_id)
        token = torch.tensor([self.vocabulary.sos_id])
        generated: list[int] = []
        for _ in range(max_characters):
            logits, hidden = self.model.decode_step(token, hidden, encoder_outputs, source_mask)
            token = logits.argmax(dim=-1)
            token_id = int(token.item())
            if token_id == self.vocabulary.eos_id:
                break
            generated.append(token_id)
        return self.vocabulary.decode(generated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn noisy text into a trained phrase.")
    parser.add_argument("text", help="Text for the model to correct")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-characters", type=int, default=160)
    args = parser.parse_args()
    generator = PhraseGenerator(args.checkpoint)
    result = generator.generate(args.text, args.max_characters)
    print(result)
    print(f"Generated characters: {len(result)}")


if __name__ == "__main__":
    main()
