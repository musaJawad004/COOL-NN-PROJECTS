"""Population lifecycle for neural car evolution."""

from __future__ import annotations

from pathlib import Path
import random

import torch

from .car import NeuralCar
from .network import NeuralGenome
from .track import Track
from .traffic import TrafficCar


class EvolutionEngine:
    def __init__(
        self,
        track: Track,
        population_size: int = 80,
        mutation_rate: float = 0.10,
        mutation_strength: float = 0.28,
        seed: int = 42,
    ) -> None:
        if not 10 <= population_size <= 500:
            raise ValueError("Population must be between 10 and 500.")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("Mutation rate must be between 0 and 1.")
        self.track = track
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.generation = 1
        self.generation_steps = 0
        self.best_history: list[float] = []
        self.log: list[str] = []
        self.global_best: NeuralGenome | None = None
        self.global_best_fitness = 0.0
        self.genomes = [NeuralGenome() for _ in range(population_size)]
        self.traffic_enabled = True
        self.traffic: list[TrafficCar] = []
        self._reset_traffic()
        self.cars = [NeuralCar.spawn(genome, track) for genome in self.genomes]

    @property
    def alive_count(self) -> int:
        return sum(car.alive for car in self.cars)

    @property
    def leader(self) -> NeuralCar:
        return max(self.cars, key=lambda car: car.fitness)

    def step(self) -> bool:
        if self.traffic_enabled:
            for traffic_car in self.traffic:
                traffic_car.update(self.track)
        obstacles = [car.position for car in self.traffic] if self.traffic_enabled else []
        for car in self.cars:
            car.update(self.track, obstacles)
        self.generation_steps += 1
        if self.alive_count == 0 or self.generation_steps >= 2_200:
            self.evolve()
            return True
        return False

    def evolve(self) -> None:
        ranked = sorted(self.cars, key=lambda car: car.fitness, reverse=True)
        champion = ranked[0]
        self.best_history.append(champion.fitness)
        self.log.append(
            f"Generation {self.generation:>4}  fitness {champion.fitness:>9.1f}  "
            f"checkpoints {champion.checkpoints:>3}  laps {champion.laps}"
        )
        if champion.fitness > self.global_best_fitness:
            self.global_best_fitness = champion.fitness
            self.global_best = champion.genome.clone()

        elite_count = max(2, self.population_size // 20)
        parent_count = max(4, self.population_size // 4)
        next_genomes = [car.genome.clone() for car in ranked[:elite_count]]
        parents = ranked[:parent_count]
        weights = list(range(parent_count, 0, -1))
        while len(next_genomes) < self.population_size:
            parent_a, parent_b = self.rng.choices(parents, weights=weights, k=2)
            child = parent_a.genome.crossover(parent_b.genome, self.rng)
            child.mutate(self.mutation_rate, self.mutation_strength, self.rng)
            next_genomes.append(child)

        self.genomes = next_genomes
        self.generation += 1
        self.restart_generation()

    def restart_generation(self) -> None:
        self.cars = [NeuralCar.spawn(genome, self.track) for genome in self.genomes]
        self._reset_traffic()
        self.generation_steps = 0

    def _reset_traffic(self) -> None:
        colors = ("#ffd166", "#8ec5ff", "#e78cff", "#ff9f6e", "#a6e36f", "#f4f4f4")
        count = 6
        spacing = max(2, len(self.track.points) // count)
        self.traffic = [
            TrafficCar.create(
                self.track,
                (index + 1) * spacing,
                1.15 + (index % 3) * 0.18,
                colors[index],
            )
            for index in range(count)
        ]

    def change_track(self, track: Track) -> None:
        self.track = track
        self.restart_generation()

    def reset(self) -> None:
        self.generation = 1
        self.best_history.clear()
        self.log.clear()
        self.global_best = None
        self.global_best_fitness = 0.0
        self.genomes = [NeuralGenome() for _ in range(self.population_size)]
        self.restart_generation()

    def save_champion(self, path: Path) -> None:
        genome = self.global_best or self.leader.genome
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"genome": genome.to_dict(), "fitness": self.global_best_fitness}, path)

    def load_champion(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        champion = NeuralGenome.from_dict(payload["genome"])
        self.global_best = champion.clone()
        self.global_best_fitness = float(payload.get("fitness", 0.0))
        self.genomes = [champion.clone()]
        while len(self.genomes) < self.population_size:
            child = champion.clone()
            child.mutate(self.mutation_rate, self.mutation_strength, self.rng)
            self.genomes.append(child)
        self.restart_generation()
