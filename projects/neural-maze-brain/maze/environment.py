"""Grid-world reinforcement learning environment."""

from __future__ import annotations

import heapq
import random

import torch


ACTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # up, right, down, left


class GridMaze:
    def __init__(self, width: int = 12, height: int = 12) -> None:
        if width < 4 or height < 4:
            raise ValueError("Maze dimensions must be at least 4×4.")
        self.width = width
        self.height = height
        self.start = (0, 0)
        self.goal = (width - 1, height - 1)
        self.walls: set[tuple[int, int]] = set()
        self.agent = self.start
        self.steps = 0
        self.max_steps = width * height * 3

    def reset(self) -> torch.Tensor:
        self.agent = self.start
        self.steps = 0
        return self.observation()

    def observation_at(self, position: tuple[int, int]) -> torch.Tensor:
        """Three image channels: walls, agent position, and goal."""
        state = torch.zeros((3, self.height, self.width), dtype=torch.float32)
        for x, y in self.walls:
            state[0, y, x] = 1.0
        state[1, position[1], position[0]] = 1.0
        state[2, self.goal[1], self.goal[0]] = 1.0
        return state.flatten()

    def observation(self) -> torch.Tensor:
        return self.observation_at(self.agent)

    def step(self, action: int) -> tuple[torch.Tensor, float, bool]:
        if action not in range(len(ACTIONS)):
            raise ValueError("Action must be 0, 1, 2, or 3.")
        old_distance = self._distance(self.agent, self.goal)
        dx, dy = ACTIONS[action]
        candidate = (self.agent[0] + dx, self.agent[1] + dy)
        self.steps += 1

        if not self.in_bounds(candidate) or candidate in self.walls:
            reward = -0.15
        else:
            self.agent = candidate
            new_distance = self._distance(self.agent, self.goal)
            reward = -0.01 + 0.015 * (old_distance - new_distance)

        reached_goal = self.agent == self.goal
        timed_out = self.steps >= self.max_steps
        if reached_goal:
            reward = 1.0
        elif timed_out:
            reward = -0.3
        return self.observation(), reward, reached_goal or timed_out

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def valid_actions(self, position: tuple[int, int] | None = None) -> list[int]:
        """Return moves that stay inside the grid and do not hit a wall."""
        x, y = self.agent if position is None else position
        return [
            action
            for action, (dx, dy) in enumerate(ACTIONS)
            if self.in_bounds((x + dx, y + dy)) and (x + dx, y + dy) not in self.walls
        ]

    def action_mask(self, position: tuple[int, int] | None = None) -> torch.Tensor:
        mask = torch.zeros(len(ACTIONS), dtype=torch.bool)
        mask[self.valid_actions(position)] = True
        return mask

    def set_wall(self, cell: tuple[int, int], enabled: bool = True) -> None:
        if not self.in_bounds(cell) or cell in (self.start, self.goal):
            return
        if enabled:
            self.walls.add(cell)
        else:
            self.walls.discard(cell)

    def set_start(self, cell: tuple[int, int]) -> None:
        if self.in_bounds(cell) and cell != self.goal:
            self.walls.discard(cell)
            self.start = cell
            self.reset()

    def set_goal(self, cell: tuple[int, int]) -> None:
        if self.in_bounds(cell) and cell != self.start:
            self.walls.discard(cell)
            self.goal = cell
            self.reset()

    def clear(self) -> None:
        self.walls.clear()
        self.reset()

    def randomize(self, density: float = 0.24, seed: int | None = None) -> None:
        rng = random.Random(seed)
        candidates = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in (self.start, self.goal)
        ]
        # Retry until the generated maze has a valid route.
        for _ in range(200):
            self.walls = {cell for cell in candidates if rng.random() < density}
            if self.shortest_path():
                self.reset()
                return
        self.clear()

    def shortest_path(self) -> list[tuple[int, int]]:
        """Find an exact shortest route with A* and a Manhattan heuristic."""
        frontier = [(self._distance(self.start, self.goal), 0, self.start)]
        costs = {self.start: 0}
        previous: dict[tuple[int, int], tuple[int, int] | None] = {self.start: None}
        while frontier:
            _, cost, cell = heapq.heappop(frontier)
            if cost != costs[cell]:
                continue
            if cell == self.goal:
                path = []
                current: tuple[int, int] | None = cell
                while current is not None:
                    path.append(current)
                    current = previous[current]
                return list(reversed(path))
            for dx, dy in ACTIONS:
                neighbor = (cell[0] + dx, cell[1] + dy)
                new_cost = cost + 1
                if (
                    self.in_bounds(neighbor)
                    and neighbor not in self.walls
                    and new_cost < costs.get(neighbor, 10**9)
                ):
                    costs[neighbor] = new_cost
                    previous[neighbor] = cell
                    priority = new_cost + self._distance(neighbor, self.goal)
                    heapq.heappush(frontier, (priority, new_cost, neighbor))
        return []

    @staticmethod
    def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
