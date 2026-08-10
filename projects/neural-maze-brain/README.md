# Neural Maze Brain

A desktop reinforcement-learning laboratory. Draw a maze and train a real Deep
Q-Network (DQN) to move from the blue start square to the green goal.

## Features

- PyTorch DQN with replay memory and a target network
- Editable 12×12 maze
- Random solvable maze generator
- Live training and animated watch mode
- Reward, loss, exploration, and success-rate statistics
- Confidence arrows showing the neural network's preferred move
- Reward-history chart
- A* path overlay for comparison
- Save and load the trained brain
- Invalid-action masking and automatic local-loop escape

## Run

Double-click `launch.command`, or open Terminal in this folder and run:

```bash
python3 -m app.gui
```

If PyTorch is not installed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m app.gui
```

## How to use

1. Pick **Wall**, **Erase**, **Start**, or **Goal**, then click the maze.
2. Click **Train Brain**. Early movement is intentionally random.
3. Watch the success rate rise and epsilon fall as the network learns.
4. Pause and click **Watch AI** to see the learned policy without exploration.
5. Click **Show A*** to compare the neural policy with an exact pathfinding algorithm.

Editing the maze resets the current episode but keeps the brain. A very different
maze may require additional training or **Reset Brain**.

The network is never allowed to select a wall as its next move. If it falls into
an A-B-A-B cycle or revisits one square repeatedly, the controller temporarily
selects the least-visited valid exit and stores that escape in replay memory so
the DQN can learn from it.

## Tests

```bash
python3 -m unittest discover -s tests
```
