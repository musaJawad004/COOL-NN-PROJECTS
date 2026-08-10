import unittest

from model.evolution import DNA, TextEvolution


class EvolutionTests(unittest.TestCase):
    def test_fitness_counts_letters_in_correct_positions(self):
        dna = DNA(list("cat"))
        self.assertAlmostEqual(dna.calculate_fitness("car"), 2 / 3)

    def test_elitism_never_reduces_best_fitness(self):
        evolution = TextEvolution("hello", population_size=30, seed=7)
        before = evolution.best.fitness
        evolution.step()
        self.assertGreaterEqual(evolution.best.fitness, before)

    def test_short_target_converges(self):
        evolution = TextEvolution("hi", population_size=200, mutation_rate=0.05, seed=3)
        for _ in range(500):
            evolution.step()
            if evolution.solved:
                break
        self.assertTrue(evolution.solved)

    def test_numeric_target_is_supported(self):
        evolution = TextEvolution("100 10", population_size=100, mutation_rate=0.10, seed=4)
        self.assertEqual(len(evolution.best.genes), 6)

    def test_excessive_workload_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "combination is too large"):
            TextEvolution("x" * 500, population_size=50_000)

    def test_correct_genes_survive_high_mutation(self):
        evolution = TextEvolution("correct letters stay", population_size=200, mutation_rate=0.50, seed=8)
        previous = evolution.best.fitness
        for _ in range(30):
            evolution.step()
            self.assertGreaterEqual(evolution.best.fitness, previous)
            previous = evolution.best.fitness

    def test_high_mutation_converges_instead_of_stalling(self):
        evolution = TextEvolution("a thirty character target text!", population_size=1_000, mutation_rate=0.50, seed=10)
        for _ in range(500):
            evolution.step()
            if evolution.solved:
                break
        self.assertTrue(evolution.solved)


if __name__ == "__main__":
    unittest.main()
