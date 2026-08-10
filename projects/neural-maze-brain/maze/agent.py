"""Deep Q-Network agent with experience replay."""

from __future__ import annotations

from collections import deque
import random
from pathlib import Path

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, input_size: int, action_count: int = 4) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_count),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


class ReplayBuffer:
    def __init__(self, capacity: int = 30_000) -> None:
        self.items: deque = deque(maxlen=capacity)

    def add(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        next_action_mask: torch.Tensor | None = None,
    ) -> None:
        if next_action_mask is None:
            next_action_mask = torch.ones(4, dtype=torch.bool)
        self.items.append(
            (state.clone(), action, reward, next_state.clone(), done, next_action_mask.clone())
        )

    def sample(self, size: int):
        return random.sample(self.items, size)

    def __len__(self) -> int:
        return len(self.items)


class DQNAgent:
    def __init__(self, state_size: int, seed: int = 42) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        self.state_size = state_size
        self.online = QNetwork(state_size)
        self.target = QNetwork(state_size)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=0.0007)
        self.memory = ReplayBuffer()
        self.gamma = 0.97
        self.epsilon = 1.0
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.992
        self.batch_size = 64
        self.training_steps = 0
        self.last_loss = 0.0

    def act(
        self,
        state: torch.Tensor,
        explore: bool = True,
        valid_actions: list[int] | None = None,
    ) -> int:
        choices = valid_actions if valid_actions else list(range(4))
        if explore and random.random() < self.epsilon:
            return random.choice(choices)
        with torch.inference_mode():
            q_values = self.online(state.unsqueeze(0)).squeeze(0)
            mask = torch.zeros(4, dtype=torch.bool)
            mask[choices] = True
            q_values = q_values.masked_fill(~mask, -torch.inf)
            return int(q_values.argmax().item())

    def learn(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        batch = self.memory.sample(self.batch_size)
        states = torch.stack([item[0] for item in batch])
        actions = torch.tensor([item[1] for item in batch], dtype=torch.long)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        next_states = torch.stack([item[3] for item in batch])
        dones = torch.tensor([item[4] for item in batch], dtype=torch.float32)
        next_action_masks = torch.stack([item[5] for item in batch])

        current_q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: online model selects; target model evaluates.
            online_next = self.online(next_states).masked_fill(~next_action_masks, -torch.inf)
            next_actions = online_next.argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            next_q = torch.where(dones.bool(), torch.zeros_like(next_q), next_q)
            expected_q = rewards + self.gamma * next_q * (1.0 - dones)
        loss = nn.functional.smooth_l1_loss(current_q, expected_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()

        self.training_steps += 1
        self.last_loss = float(loss.item())
        if self.training_steps % 250 == 0:
            self.target.load_state_dict(self.online.state_dict())
        return self.last_loss

    def finish_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @torch.inference_mode()
    def policy(
        self,
        states: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_values = self.online(states)
        if action_masks is not None:
            q_values = q_values.masked_fill(~action_masks, -torch.inf)
        probabilities = torch.softmax(q_values, dim=1)
        return q_values.argmax(dim=1), probabilities.max(dim=1).values

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_size": self.state_size,
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_steps": self.training_steps,
            },
            path,
        )

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["state_size"] != self.state_size:
            raise ValueError("This checkpoint was trained for a different maze size.")
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload["target"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.epsilon = float(payload["epsilon"])
        self.training_steps = int(payload["training_steps"])
