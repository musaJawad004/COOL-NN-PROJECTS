"""Deterministic traffic cars that follow the circuit center line."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .track import Point, Track, distance


@dataclass
class TrafficCar:
    position: Point
    target_index: int
    speed: float
    color: str
    heading: float = 0.0

    @classmethod
    def create(cls, track: Track, point_index: int, speed: float, color: str) -> "TrafficCar":
        position = track.points[point_index % len(track.points)]
        return cls(position, (point_index + 1) % len(track.points), speed, color)

    def update(self, track: Track) -> None:
        target = track.points[self.target_index]
        dx, dy = target[0] - self.position[0], target[1] - self.position[1]
        remaining = distance(self.position, target)
        if remaining <= self.speed:
            self.position = target
            self.target_index = (self.target_index + 1) % len(track.points)
            target = track.points[self.target_index]
            dx, dy = target[0] - self.position[0], target[1] - self.position[1]
            remaining = max(distance(self.position, target), 0.001)
        self.heading = math.atan2(dy, dx)
        self.position = (
            self.position[0] + dx / max(remaining, 0.001) * self.speed,
            self.position[1] + dy / max(remaining, 0.001) * self.speed,
        )
