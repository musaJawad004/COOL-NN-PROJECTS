# Neural Car Driving Simulator

A visual neuroevolution laboratory. Every car is controlled by a small neural
network. Five distance sensors observe the road; two network outputs control
steering and acceleration. The best drivers breed, cross over, and mutate to
create the next generation.

## Features

- Real PyTorch feed-forward neural-network controllers
- Population-based evolution with elitism, crossover, and mutation
- Five live ray sensors per car
- Steering, acceleration, drag, collision, and stuck detection
- Three built-in tracks plus freehand track drawing
- 1×, 5×, and 20× simulation speeds
- Generation, alive-car, fitness, checkpoint, and lap statistics
- Fitness-history graph and generation log
- Save/load the all-time champion brain
- Automatic generation advancement
- Six moving traffic cars with sensor detection and collision avoidance
- Smooth circuit presets with curbs, lane markings, grass, and a finish line

## Run

Double-click `launch.command`, or run:

```bash
python3 -m app.gui
```

If PyTorch is missing:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m app.gui
```

## Use

1. Click **Start Evolution**.
2. Let several generations run. Early cars will crash quickly.
3. Increase simulation speed while training and return to 1× to watch closely.
4. Change mutation percentage if the population stops improving.
5. Click **Draw Track**, drag a closed route, and release.
6. Save the best brain once a capable driver evolves.

The green car is the current generation leader. Its sensor rays are shown in
blue. Yellow, blue, purple, orange, and white cars are moving traffic. Cars score
fitness by reaching ordered checkpoints, so circling one small area cannot
produce a high score. The neural network also receives the direction and distance
to the next checkpoint, allowing it to learn technical circuits instead of
behaving like a maze walker.

## Tests

```bash
python3 -m unittest discover -s tests
```
