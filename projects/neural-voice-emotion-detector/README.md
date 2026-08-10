# Neural Voice Emotion Detector

A local desktop application that learns emotional patterns from voice recordings.
It converts audio into log-mel spectrograms and trains a convolutional neural
network to classify **happy**, **sad**, **angry**, **calm**, and **nervous** speech.

## Features

- Three-second microphone recording
- Labeled local audio dataset
- Waveform and spectrogram visualization
- Real PyTorch CNN training
- Noise, gain, and time-shift augmentation
- Confidence bars for all five emotions
- Dataset counts and training progress
- Save/load best model checkpoint
- WAV import for existing recordings
- Entirely local processing

## Run

Double-click `launch.command`, or run:

```bash
python3 -m app.gui
```

If dependencies are missing:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m app.gui
```

macOS may ask for microphone access the first time. Allow access for Terminal or
Python. The application does not upload recordings anywhere.

## Training workflow

1. Select an emotion.
2. Click **Record labeled sample** and speak naturally for three seconds.
3. Record at least five varied samples for each emotion. Ten or more is better.
4. Click **Train neural network**.
5. Make a new recording and click **Analyze current audio**.

For useful results, vary sentences, speaking volume, and distance from the
microphone. The model learns acoustic patterns from your data; it cannot reliably
recognize speakers or emotions it has never encountered.

## Included demo audio

Ten generated WAV files per emotion are included so training works immediately.
They contain distinct synthetic tone patterns, not real emotional speech. Use
them to verify the pipeline, then add genuine labeled recordings for meaningful
predictions. Recreate them with `python3 scripts/generate_demo_audio.py`.

A balanced 50-file human-speech subset of RAVDESS is also included: ten samples
per label. When at least two genuine speech files exist for every emotion, the
trainer automatically excludes synthetic tones and trains only on speech. See
`data/RAVDESS_ATTRIBUTION.md` for the required attribution and non-commercial
ShareAlike license terms.

## Tests

```bash
python3 -m unittest discover -s tests
```

This is an educational classifier, not a medical or psychological assessment.
