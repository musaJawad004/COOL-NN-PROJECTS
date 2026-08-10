import unittest

from maze.environment import GridMaze


class GridMazeTests(unittest.TestCase):
    def test_observation_shape(self):
        maze = GridMaze(6, 5)
        self.assertEqual(tuple(maze.observation().shape), (6 * 5 * 3,))

    def test_wall_blocks_agent(self):
        maze = GridMaze(5, 5)
        maze.set_wall((1, 0))
        _, reward, done = maze.step(1)
        self.assertEqual(maze.agent, (0, 0))
        self.assertEqual(reward, -0.15)
        self.assertFalse(done)

    def test_goal_completes_episode(self):
        maze = GridMaze(4, 4)
        maze.set_goal((1, 0))
        _, reward, done = maze.step(1)
        self.assertTrue(done)
        self.assertEqual(reward, 1.0)

    def test_shortest_path_avoids_wall(self):
        maze = GridMaze(4, 4)
        maze.set_wall((1, 0))
        path = maze.shortest_path()
        self.assertNotIn((1, 0), path)
        self.assertEqual(path[0], maze.start)
        self.assertEqual(path[-1], maze.goal)

    def test_random_maze_is_solvable(self):
        maze = GridMaze()
        maze.randomize(density=0.35, seed=12)
        self.assertTrue(maze.shortest_path())

    def test_action_mask_excludes_walls_and_boundaries(self):
        maze = GridMaze(4, 4)
        maze.set_wall((1, 0))
        self.assertEqual(maze.valid_actions(), [2])
        self.assertEqual(maze.action_mask().tolist(), [False, False, True, False])


if __name__ == "__main__":
    unittest.main()
