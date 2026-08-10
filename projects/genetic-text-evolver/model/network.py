"""Attention-based character sequence-to-sequence model."""

from __future__ import annotations

import random

import torch
from torch import nn


class TextCorrector(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, embedding_dim: int = 96, hidden_dim: int = 192):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.bridge = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_query = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.decoder = nn.GRU(embedding_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim + hidden_dim * 2, vocab_size)

    def encode(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(source)
        encoder_outputs, encoder_hidden = self.encoder(embedded)
        joined = torch.cat((encoder_hidden[-2], encoder_hidden[-1]), dim=-1)
        decoder_hidden = torch.tanh(self.bridge(joined)).unsqueeze(0)
        return encoder_outputs, decoder_hidden

    def decode_step(
        self,
        token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.attention_query(hidden[-1])
        scores = torch.bmm(encoder_outputs, query.unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(~source_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        embedded = self.embedding(token).unsqueeze(1)
        decoder_input = torch.cat((embedded, context), dim=-1)
        decoder_output, hidden = self.decoder(decoder_input, hidden)
        logits = self.output(torch.cat((decoder_output.squeeze(1), context.squeeze(1)), dim=-1))
        return logits, hidden

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        pad_id: int,
        teacher_forcing: float = 0.6,
    ) -> torch.Tensor:
        encoder_outputs, hidden = self.encode(source)
        source_mask = source.ne(pad_id)
        token = target[:, 0]
        outputs = []
        for position in range(1, target.size(1)):
            logits, hidden = self.decode_step(token, hidden, encoder_outputs, source_mask)
            outputs.append(logits)
            predicted = logits.argmax(dim=-1)
            token = target[:, position] if random.random() < teacher_forcing else predicted
        return torch.stack(outputs, dim=1)
