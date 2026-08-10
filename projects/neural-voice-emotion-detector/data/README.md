# Demo audio dataset

This directory contains ten generated WAV files for each application label:
happy, sad, angry, calm, and nervous.

The files are deterministic synthetic tone patterns. They are useful for testing
record loading, spectrogram generation, CNN training, checkpoint creation, and
prediction. They are **not recordings of genuine human emotion** and must not be
used to claim real-world emotion-recognition accuracy.

For meaningful results, replace or supplement these files with varied labeled
human speech. Run `python3 scripts/generate_demo_audio.py` to recreate the demo
files.
