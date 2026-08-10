"""Audio feature extraction and neural emotion classification."""

from .audio import EMOTIONS, SAMPLE_RATE, audio_to_spectrogram, load_wav, save_wav
from .model import EmotionCNN

__all__ = ["EMOTIONS", "SAMPLE_RATE", "EmotionCNN", "audio_to_spectrogram", "load_wav", "save_wav"]
