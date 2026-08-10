import tempfile
import unittest
from pathlib import Path
import math

from simulation.car import NeuralCar
from simulation.evolution import EvolutionEngine
from simulation.network import NeuralGenome
from simulation.track import preset_track
from simulation.traffic import TrafficCar


class EvolutionTests(unittest.TestCase):
    def test_network_outputs_controls(self):
        steering, throttle = NeuralGenome().forward([0.5] * 9)
        self.assertGreaterEqual(steering, -1.0)
        self.assertLessEqual(steering, 1.0)
        self.assertGreaterEqual(throttle, -1.0)
        self.assertLessEqual(throttle, 1.0)

    def test_car_moves(self):
        track = preset_track(1)
        car = NeuralCar.spawn(NeuralGenome(), track)
        before = car.position
        car.update(track)
        self.assertNotEqual(car.position, before)

    def test_generation_evolves(self):
        engine = EvolutionEngine(preset_track(1), population_size=10)
        engine.evolve()
        self.assertEqual(engine.generation, 2)
        self.assertEqual(len(engine.cars), 10)
        self.assertEqual(len(engine.best_history), 1)

    def test_champion_round_trip(self):
        engine = EvolutionEngine(preset_track(1), population_size=10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "champion.pt"
            engine.save_champion(path)
            self.assertTrue(path.exists())
            engine.load_champion(path)
            self.assertEqual(len(engine.genomes), 10)

    def test_traffic_moves_and_is_detected_by_sensor(self):
        track = preset_track(1)
        car = NeuralCar.spawn(NeuralGenome(), track)
        obstacle = (
            car.position[0] + math.cos(car.heading) * 30,
            car.position[1] + math.sin(car.heading) * 30,
        )
        sensor = car._sensor_distance(track, car.heading, [obstacle])
        self.assertLess(sensor, 30)
        traffic = TrafficCar.create(track, 5, 1.5, "yellow")
        before = traffic.position
        traffic.update(track)
        self.assertNotEqual(traffic.position, before)


if __name__ == "__main__":
    unittest.main()
