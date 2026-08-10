"""Small evolvable neural-network genome."""

from __future__ import annotations

import random

import torch


class NeuralGenome:
    INPUTS = 9
    HIDDEN = 16
    OUTPUTS = 2

    def __init__(self, tensors: list[torch.Tensor] | None = None) -> None:
        self.tensors = tensors or [
            torch.randn(self.HIDDEN, self.INPUTS) * 0.55,
            torch.randn(self.HIDDEN) * 0.15,
            torch.randn(self.OUTPUTS, self.HIDDEN) * 0.55,
            torch.randn(self.OUTPUTS) * 0.15,
        ]

    def forward(self, inputs: list[float]) -> tuple[float, float]:
        w1, b1, w2, b2 = self.tensors
        x = torch.tensor(inputs, dtype=torch.float32)
        hidden = torch.tanh(torch.mv(w1, x) + b1)
        output = torch.tanh(torch.mv(w2, hidden) + b2)
        return float(output[0]), float(output[1])

    def clone(self) -> "NeuralGenome":
        return NeuralGenome([tensor.clone() for tensor in self.tensors])

    def crossover(self, partner: "NeuralGenome", rng: random.Random) -> "NeuralGenome":
        children = []
        for mine, theirs in zip(self.tensors, partner.tensors):
            mask = torch.rand(mine.shape) < 0.5
            children.append(torch.where(mask, mine, theirs))
        return NeuralGenome(children)

    def mutate(self, rate: float, strength: float, rng: random.Random) -> None:
        # rng is accepted so evolution remains reproducible at the selection level.
        _ = rng
        for index, tensor in enumerate(self.tensors):
            mask = torch.rand(tensor.shape) < rate
            noise = torch.randn(tensor.shape) * strength
            self.tensors[index] = tensor + mask * noise

    def to_dict(self) -> dict:
        return {"tensors": [tensor.clone() for tensor in self.tensors]}

    @classmethod
    def from_dict(cls, data: dict) -> "NeuralGenome":
        tensors = [tensor.clone() for tensor in data["tensors"]]
        expected = [
            (cls.HIDDEN, cls.INPUTS),
            (cls.HIDDEN,),
            (cls.OUTPUTS, cls.HIDDEN),
            (cls.OUTPUTS,),
        ]
        if [tuple(tensor.shape) for tensor in tensors] != expected:
            raise ValueError(
                "This champion uses the older neural-network design. "
                "Reset and train a new champion for the upgraded simulator."
            )
        return cls(tensors)
