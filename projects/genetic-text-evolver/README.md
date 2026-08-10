# Genetic Text Evolver + Gibberish to Phrase AI

## Main app: genetic evolution

The main desktop app follows a genetic algorithm like the classic “infinite
monkey” example. It creates a population of random strings, calculates fitness
from correctly positioned characters, selects parents, performs crossover and
mutation, and repeats until the target phrase is reached.

Double-click `launch.command`, or run:

```bash
python3 -m app.evolution_gui
```

You can change the target phrase, population size, and mutation percentage. For
example, enter `10` for a 10% mutation chance. The app
shows the best string, generation number, fitness, matching letters, and a live
generation log. Every generation is shown; green characters match the target
and orange characters still need to evolve. Slow, Normal, and Fast display
speeds change the delay without skipping generations. This mode uses only
Python's standard library.

Matched characters are preserved, so even unusually high mutation values such
as 50% cannot destroy the best discoveries. If fitness does not improve for 100
generations, the weakest quarter of the population is automatically refreshed.

## Optional neural-network experiment

A small, local character-level neural network that learns from paired examples:

```text
noisy or coded text  ->  intended sentence
```

The desktop app shows the generated sentence one character at a time, together
with the current and final character counts.

## Important limitation

A neural network cannot discover meaning in truly random letters. For example,
`hfdshfsid` could mean anything unless training data says what it means. Add
several examples for every kind of input you want the model to understand. For
ordinary misspellings, the trainer can automatically create extra typo examples.

## Project folders

- `content/` — editable input/output training pairs
- `model/` — vocabulary and encoder/decoder neural network
- `app/` — desktop GUI and command-line generator
- `work/checkpoints/` — trained model files (created by training)
- `tests/` — lightweight tests

## Setup

Open Terminal in this folder, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Add your examples

Edit `content/training_pairs.json`. Every item needs an `input` (the scrambled
text) and an `output` (the sentence it should become):

```json
{"input": "how aer yuo", "output": "how are you?"}
```

More varied examples produce better results. Keep at least 50-100 examples for
a useful experiment; a real product normally needs thousands.

## Train

```bash
source .venv/bin/activate
python train.py --epochs 80
```

The best model is saved as `work/checkpoints/best_model.pt`. Training also
creates typo variations by default. Use `--augmentations 0` if every input is an
arbitrary code whose characters have no relationship to its output.

## Run the desktop app

```bash
source .venv/bin/activate
python -m app.gui
```

Or double-click `launch.command` after setup and training.

## Command-line generation

```bash
python -m app.generate "how aer yuo"
```

## Test

```bash
python -m unittest discover -s tests
```

This is an educational prototype. The model is a bidirectional GRU encoder with
an attention-based GRU decoder and greedy character-by-character decoding.
