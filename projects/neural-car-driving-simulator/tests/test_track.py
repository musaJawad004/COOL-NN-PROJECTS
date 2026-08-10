import unittest

from simulation.track import Track, distance_to_segment, preset_track


class TrackTests(unittest.TestCase):
    def test_segment_distance(self):
        self.assertAlmostEqual(distance_to_segment((5, 3), (0, 0), (10, 0)), 3.0)

    def test_center_point_is_on_road(self):
        track = preset_track(1)
        self.assertTrue(track.on_road(track.points[2]))

    def test_distant_point_is_off_road(self):
        track = preset_track(1)
        self.assertFalse(track.on_road((750, 10)))

    def test_short_track_is_rejected(self):
        with self.assertRaises(ValueError):
            Track([(0, 0)] * 5)


if __name__ == "__main__":
    unittest.main()
