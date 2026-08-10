import tempfile
import unittest
from pathlib import Path

import torch

from maze.agent import DQNAgent


class AgentTests(unittest.TestCase):
    def test_action_is_valid(self):
        agent = DQNAgent(12)
        self.assertIn(agent.act(torch.zeros(12), explore=False), range(4))

    def test_learning_updates_loss(self):
        agent = DQNAgent(12)
        agent.batch_size = 4
        state = torch.zeros(12)
        next_state = torch.ones(12)
        for action in range(4):
            agent.memory.add(state, action, 0.1, next_state, False)
        loss = agent.learn()
        self.assertIsNotNone(loss)
        self.assertGreaterEqual(agent.last_loss, 0.0)

    def test_checkpoint_round_trip(self):
        agent = DQNAgent(12)
        agent.epsilon = 0.25
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brain.pt"
            agent.save(path)
            restored = DQNAgent(12)
            restored.load(path)
            self.assertEqual(restored.epsilon, 0.25)

    def test_action_mask_is_respected(self):
        agent = DQNAgent(12)
        state = torch.zeros(12)
        for _ in range(20):
            self.assertEqual(agent.act(state, explore=True, valid_actions=[2]), 2)


if __name__ == "__main__":
    unittest.main()
