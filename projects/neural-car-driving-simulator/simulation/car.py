"""Car physics and neural controller integration."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .network import NeuralGenome
from .track import Point, Track, distance


SENSOR_ANGLES = (-1.15, -0.55, 0.0, 0.55, 1.15)
MAX_SENSOR = 145.0


@dataclass
class NeuralCar:
    genome: NeuralGenome
    position: Point
    heading: float
    speed: float = 0.0
    alive: bool = True
    fitness: float = 0.0
    checkpoints: int = 0
    laps: int = 0
    next_checkpoint: int = 1
    steps_alive: int = 0
    steps_without_progress: int = 0
    sensors: list[float] = field(default_factory=lambda: [0.0] * 5)
    best_checkpoint_approach: float = 0.0

    @classmethod
    def spawn(cls, genome: NeuralGenome, track: Track) -> "NeuralCar":
        position, heading = track.starting_pose
        return cls(genome, position, heading)

    def update(self, track: Track, obstacles: list[Point] | None = None) -> None:
        if not self.alive:
            return
        obstacles = obstacles or []
        self.sensors = [
            self._sensor_distance(track, self.heading + offset, obstacles)
            for offset in SENSOR_ANGLES
        ]
        checkpoint = track.points[self.next_checkpoint]
        target_angle = math.atan2(checkpoint[1] - self.position[1], checkpoint[0] - self.position[0])
        relative_angle = (target_angle - self.heading + math.pi) % (2 * math.pi) - math.pi
        target_distance = min(distance(self.position, checkpoint) / 180.0, 1.0)
        inputs = [value / MAX_SENSOR for value in self.sensors] + [
            self.speed / 5.2,
            math.sin(relative_angle),
            math.cos(relative_angle),
            target_distance,
        ]
        steering, throttle = self.genome.forward(inputs)
        self.speed += (throttle + 1.0) * 0.055 - 0.025
        self.speed = max(0.35, min(5.2, self.speed))
        self.heading += steering * 0.075 * (0.35 + self.speed / 5.2)
        self.position = (
            self.position[0] + math.cos(self.heading) * self.speed,
            self.position[1] + math.sin(self.heading) * self.speed,
        )
        self.steps_alive += 1
        self.steps_without_progress += 1

        if not track.on_road(self.position, margin=5.0):
            self.alive = False
            return

        if any(distance(self.position, obstacle) < 14.0 for obstacle in obstacles):
            self.alive = False
            return

        checkpoint = track.points[self.next_checkpoint]
        if distance(self.position, checkpoint) < track.road_width * 0.58:
            self.checkpoints += 1
            self.next_checkpoint = (self.next_checkpoint + 1) % len(track.points)
            if self.next_checkpoint == 1:
                self.laps += 1
            self.steps_without_progress = 0
            self.best_checkpoint_approach = 0.0

        checkpoint_distance = distance(self.position, track.points[self.next_checkpoint])
        approach = max(0.0, 1.0 - checkpoint_distance / 180.0)
        self.best_checkpoint_approach = max(self.best_checkpoint_approach, approach)
        self.fitness = (
            self.checkpoints * 1_000.0
            + self.laps * 5_000.0
            + self.steps_alive * 0.03
            + self.best_checkpoint_approach * 350.0
        )
        if self.steps_without_progress > 420:
            self.alive = False

    def _sensor_distance(self, track: Track, angle: float, obstacles: list[Point]) -> float:
        nearest = track.ray_distance(self.position, angle, MAX_SENSOR)
        ray_x, ray_y = math.cos(angle), math.sin(angle)
        obstacle_radius = 12.0
        for obstacle in obstacles:
            dx = obstacle[0] - self.position[0]
            dy = obstacle[1] - self.position[1]
            projection = dx * ray_x + dy * ray_y
            if projection <= 0 or projection >= nearest:
                continue
            lateral_squared = dx * dx + dy * dy - projection * projection
            if lateral_squared <= obstacle_radius * obstacle_radius:
                entry = projection - math.sqrt(max(0.0, obstacle_radius**2 - lateral_squared))
                nearest = max(0.0, entry)
        return nearest
