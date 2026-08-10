"""Character vocabulary used by training and inference."""

from __future__ import annotations

from dataclasses import dataclass


SPECIAL_TOKENS = ("<pad>", "<sos>", "<eos>", "<unk>")


@dataclass
class Vocabulary:
    char_to_id: dict[str, int]
    id_to_char: list[str]

    @classmethod
    def build(cls, texts: list[str]) -> "Vocabulary":
        characters = sorted(set("".join(texts)))
        symbols = list(SPECIAL_TOKENS) + characters
        return cls({symbol: i for i, symbol in enumerate(symbols)}, symbols)

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        symbols = list(data["id_to_char"])
        return cls({symbol: i for i, symbol in enumerate(symbols)}, symbols)

    def to_dict(self) -> dict:
        return {"id_to_char": self.id_to_char}

    def __len__(self) -> int:
        return len(self.id_to_char)

    @property
    def pad_id(self) -> int:
        return self.char_to_id["<pad>"]

    @property
    def sos_id(self) -> int:
        return self.char_to_id["<sos>"]

    @property
    def eos_id(self) -> int:
        return self.char_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.char_to_id["<unk>"]

    def encode(self, text: str, boundaries: bool = True) -> list[int]:
        ids = [self.char_to_id.get(char, self.unk_id) for char in text.lower()]
        return [self.sos_id, *ids, self.eos_id] if boundaries else ids

    def decode(self, ids: list[int]) -> str:
        ignored = set(SPECIAL_TOKENS)
        return "".join(
            self.id_to_char[index]
            for index in ids
            if 0 <= index < len(self) and self.id_to_char[index] not in ignored
        )
