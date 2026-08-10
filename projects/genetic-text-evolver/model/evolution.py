"""Genetic algorithm that evolves random text toward a target phrase."""

from __future__ import annotations

from dataclasses import dataclass
import random
import string


ALPHABET = string.ascii_lowercase + string.ascii_uppercase + string.digits + " .,!?'-"
MAX_POPULATION = 50_000
MAX_TARGET_LENGTH = 500
MAX_GENERATION_CELLS = 2_000_000


@dataclass
class DNA:
    genes: list[str]
    fitness: float = 0.0

    @classmethod
    def random(cls, length: int, rng: random.Random) -> "DNA":
        return cls([rng.choice(ALPHABET) for _ in range(length)])

    @property
    def phrase(self) -> str:
        return "".join(self.genes)

    def calculate_fitness(self, target: str) -> float:
        matches = sum(gene == expected for gene, expected in zip(self.genes, target))
        self.fitness = matches / len(target)
        return self.fitness

    def crossover(self, partner: "DNA", rng: random.Random, target: str | None = None) -> "DNA":
        # If either parent has already discovered the correct character, keep it.
        # This prevents high mutation settings from destroying useful progress.
        genes = []
        for index, (mine, theirs) in enumerate(zip(self.genes, partner.genes)):
            if target is not None and mine == target[index]:
                genes.append(mine)
            elif target is not None and theirs == target[index]:
                genes.append(theirs)
            else:
                genes.append(mine if rng.random() < 0.5 else theirs)
        return DNA(genes)

    def mutate(
        self,
        mutation_rate: float,
        rng: random.Random,
        target: str | None = None,
        preserve_matches: bool = False,
    ) -> None:
        for index in range(len(self.genes)):
            if preserve_matches and target is not None and self.genes[index] == target[index]:
                continue
            if rng.random() < mutation_rate:
                self.genes[index] = rng.choice(ALPHABET)


class TextEvolution:
    """Owns a population and advances it one generation at a time."""

    def __init__(
        self,
        target: str,
        population_size: int = 300,
        mutation_rate: float = 0.02,
        elitism: int = 4,
        seed: int | None = None,
    ) -> None:
        if not target:
            raise ValueError("The target phrase cannot be empty.")
        if len(target) > MAX_TARGET_LENGTH:
            raise ValueError(f"The target phrase can contain at most {MAX_TARGET_LENGTH} characters.")
        unsupported = sorted(set(target) - set(ALPHABET))
        if unsupported:
            raise ValueError(f"Unsupported target characters: {''.join(unsupported)}")
        if population_size < 2:
            raise ValueError("Population size must be at least 2.")
        if population_size > MAX_POPULATION:
            raise ValueError(f"Population size can be at most {MAX_POPULATION:,}.")
        if population_size * len(target) > MAX_GENERATION_CELLS:
            recommended = max(2, MAX_GENERATION_CELLS // len(target))
            raise ValueError(
                "That population and phrase combination is too large. "
                f"For this phrase, use a population of {recommended:,} or less."
            )
        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1.")

        self.target = target
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism = min(max(elitism, 1), population_size)
        self.rng = random.Random(seed)
        self.generation = 0
        self.stagnant_generations = 0
        self.population = [DNA.random(len(target), self.rng) for _ in range(population_size)]
        self._score_and_sort()
        self._best_fitness = self.best.fitness

    @property
    def best(self) -> DNA:
        return self.population[0]

    @property
    def solved(self) -> bool:
        return self.best.phrase == self.target

    @property
    def matching_characters(self) -> int:
        return round(self.best.fitness * len(self.target))

    def _score_and_sort(self) -> None:
        for member in self.population:
            member.calculate_fitness(self.target)
        self.population.sort(key=lambda member: member.fitness, reverse=True)

    def step(self) -> DNA:
        if self.solved:
            return self.best

        next_population = [DNA(member.genes.copy()) for member in self.population[: self.elitism]]
        child_count = self.population_size - len(next_population)
        # Build selection weights once per generation. Calling random.choices for
        # every parent made large populations accidentally O(n²) and froze the UI.
        weights = [(member.fitness + 0.01) ** 4 for member in self.population]
        parents = self.rng.choices(self.population, weights=weights, k=child_count * 2)
        for index in range(child_count):
            parent_a = parents[index * 2]
            parent_b = parents[index * 2 + 1]
            child = parent_a.crossover(parent_b, self.rng, self.target)
            child.mutate(
                self.mutation_rate,
                self.rng,
                target=self.target,
                preserve_matches=True,
            )
            next_population.append(child)

        self.population = next_population
        self.generation += 1
        self._score_and_sort()
        if self.best.fitness > self._best_fitness:
            self._best_fitness = self.best.fitness
            self.stagnant_generations = 0
        else:
            self.stagnant_generations += 1

        # If selection loses diversity, replace the weakest quarter. Their
        # correct positions come from the champion; all other positions restart.
        if self.stagnant_generations >= 100 and not self.solved:
            champion = self.best
            restart_count = max(1, self.population_size // 4)
            for offset in range(1, restart_count + 1):
                genes = [
                    gene if gene == expected else self.rng.choice(ALPHABET)
                    for gene, expected in zip(champion.genes, self.target)
                ]
                self.population[-offset] = DNA(genes)
            self._score_and_sort()
            self.stagnant_generations = 0
        return self.best
