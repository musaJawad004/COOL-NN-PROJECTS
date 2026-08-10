"""Tkinter desktop application for voice emotion training and inference."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import sounddevice as sd

from emotion_ai.audio import DURATION_SECONDS, EMOTIONS, SAMPLE_COUNT, SAMPLE_RATE, audio_to_spectrogram, load_wav, save_wav
from emotion_ai.model import EmotionCNN
from emotion_ai.training import dataset_files, predict, train_model, training_dataset_files


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
CHECKPOINT = ROOT / "work" / "checkpoints" / "emotion_cnn.pt"
COLORS = {"happy": "#ffd166", "sad": "#6ea8ff", "angry": "#ff6b6b", "calm": "#65e6b4", "nervous": "#c990ff"}


class EmotionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Neural Voice Emotion Detector")
        self.geometry("1120x760")
        self.minsize(980, 680)
        self.configure(bg="#090e1a")
        self.current_audio: np.ndarray | None = None
        self.model: EmotionCNN | None = None
        self.events: queue.Queue[tuple] = queue.Queue()
        self.emotion = tk.StringVar(value="happy")
        self._build_ui()
        self._load_checkpoint_if_present()
        self._update_dataset_counts()
        self.after(80, self._poll_events)

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Primary.TButton", font=("Helvetica", 11, "bold"), padding=9)
        shell = tk.Frame(self, bg="#090e1a", padx=24, pady=20)
        shell.pack(fill="both", expand=True)
        header = tk.Frame(shell, bg="#090e1a")
        header.pack(fill="x", pady=(0, 16))
        tk.Label(header, text="Neural Voice Emotion Detector", bg="#090e1a", fg="#f5f8ff", font=("Helvetica", 25, "bold")).pack(side="left")
        self.status = tk.Label(header, text="READY", bg="#15213a", fg="#78c5ff", padx=12, pady=6, font=("Helvetica", 10, "bold"))
        self.status.pack(side="right")

        top = tk.Frame(shell, bg="#090e1a")
        top.pack(fill="x")
        controls = tk.Frame(top, bg="#111a2c", padx=16, pady=14)
        controls.pack(side="left", fill="y")
        tk.Label(controls, text="SAMPLE LABEL", bg="#111a2c", fg="#7f8eaa", font=("Helvetica", 9, "bold")).pack(anchor="w")
        emotion_box = ttk.Combobox(controls, values=EMOTIONS, textvariable=self.emotion, state="readonly", width=14)
        emotion_box.pack(fill="x", pady=(6, 12))
        self.record_button = ttk.Button(controls, text="● Record labeled sample", style="Primary.TButton", command=self.record_labeled)
        self.record_button.pack(fill="x")
        self.test_record_button = ttk.Button(controls, text="Record test audio", command=self.record_test)
        self.test_record_button.pack(fill="x", pady=(7, 0))
        ttk.Button(controls, text="Import WAV sample", command=self.import_wav).pack(fill="x", pady=(7, 0))
        ttk.Button(controls, text="Analyze current audio", command=self.analyze).pack(fill="x", pady=(14, 0))
        self.timer = tk.Label(controls, text=f"{DURATION_SECONDS}.0 seconds", bg="#111a2c", fg="#9aa8c1", font=("Helvetica", 11))
        self.timer.pack(pady=(12, 0))

        visuals = tk.Frame(top, bg="#090e1a", padx=15)
        visuals.pack(side="left", fill="both", expand=True)
        tk.Label(visuals, text="WAVEFORM", bg="#090e1a", fg="#71809f", font=("Helvetica", 9, "bold")).pack(anchor="w")
        self.waveform = tk.Canvas(visuals, height=135, bg="#10192b", highlightthickness=0)
        self.waveform.pack(fill="x", pady=(5, 12))
        tk.Label(visuals, text="LOG-MEL SPECTROGRAM", bg="#090e1a", fg="#71809f", font=("Helvetica", 9, "bold")).pack(anchor="w")
        self.spectrogram = tk.Canvas(visuals, height=175, bg="#10192b", highlightthickness=0)
        self.spectrogram.pack(fill="x", pady=(5, 0))

        bottom = tk.Frame(shell, bg="#090e1a")
        bottom.pack(fill="both", expand=True, pady=(16, 0))
        train_panel = tk.Frame(bottom, bg="#111a2c", padx=14, pady=13)
        train_panel.pack(side="left", fill="both", expand=True)
        tk.Label(train_panel, text="LOCAL TRAINING DATA", bg="#111a2c", fg="#71809f", font=("Helvetica", 9, "bold")).pack(anchor="w")
        self.dataset_label = tk.Label(train_panel, text="", bg="#111a2c", fg="#d4dbeb", font=("Menlo", 11), justify="left")
        self.dataset_label.pack(anchor="w", pady=(8, 12))
        self.train_button = ttk.Button(train_panel, text="Train neural network", style="Primary.TButton", command=self.train)
        self.train_button.pack(fill="x")
        self.training_label = tk.Label(train_panel, text="Record at least two samples for every emotion to begin.", bg="#111a2c", fg="#8d9ab3", wraplength=350, justify="left")
        self.training_label.pack(anchor="w", pady=(10, 0))

        result = tk.Frame(bottom, bg="#111a2c", padx=14, pady=13)
        result.pack(side="left", fill="both", expand=True, padx=(14, 0))
        tk.Label(result, text="NEURAL PREDICTION", bg="#111a2c", fg="#71809f", font=("Helvetica", 9, "bold")).pack(anchor="w")
        self.prediction_title = tk.Label(result, text="No audio analyzed", bg="#111a2c", fg="#f5f8ff", font=("Helvetica", 21, "bold"))
        self.prediction_title.pack(anchor="w", pady=(8, 10))
        self.confidence_canvas = tk.Canvas(result, height=175, bg="#111a2c", highlightthickness=0)
        self.confidence_canvas.pack(fill="both", expand=True)
        self._draw_confidences({emotion: 0.0 for emotion in EMOTIONS})

    def record_labeled(self) -> None:
        self._start_recording(save_label=self.emotion.get())

    def record_test(self) -> None:
        self._start_recording(save_label=None)

    def _start_recording(self, save_label: str | None) -> None:
        self.record_button.configure(state="disabled")
        self.test_record_button.configure(state="disabled")
        self.status.configure(text="RECORDING", fg="#ff7c89")
        self._countdown(DURATION_SECONDS * 10)

        def worker() -> None:
            try:
                audio = sd.rec(SAMPLE_COUNT, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
                sd.wait()
                recording = audio[:, 0].copy()
                if float(np.sqrt(np.mean(recording**2))) < 0.003:
                    raise ValueError("The recording is nearly silent. Check microphone access and try again.")
                path = None
                if save_label:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    path = DATA_ROOT / save_label / f"{stamp}.wav"
                    save_wav(path, recording)
                self.events.put(("recorded", recording, path))
            except Exception as error:
                self.events.put(("error", f"Microphone recording failed: {error}"))
        threading.Thread(target=worker, daemon=True).start()

    def _countdown(self, ticks: int) -> None:
        if ticks >= 0 and str(self.record_button["state"]) == "disabled":
            self.timer.configure(text=f"{ticks / 10:.1f} seconds")
            self.after(100, self._countdown, ticks - 1)

    def import_wav(self) -> None:
        source = filedialog.askopenfilename(filetypes=[("WAV audio", "*.wav")])
        if not source:
            return
        try:
            audio = load_wav(Path(source))
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            destination = DATA_ROOT / self.emotion.get() / f"imported_{stamp}.wav"
            save_wav(destination, audio)
        except Exception as error:
            messagebox.showerror("Import failed", str(error))
            return
        self.current_audio = audio
        self._draw_audio()
        self._update_dataset_counts()
        self.status.configure(text="SAMPLE IMPORTED", fg="#70e5aa")

    def train(self) -> None:
        self.train_button.configure(state="disabled")
        self.status.configure(text="TRAINING", fg="#c490ff")
        self.training_label.configure(text="Preparing spectrograms…")

        def progress(epoch: int, loss: float, training_accuracy: float, validation_accuracy: float) -> None:
            self.events.put(("progress", epoch, loss, training_accuracy, validation_accuracy))

        def worker() -> None:
            try:
                model, metadata = train_model(DATA_ROOT, CHECKPOINT, progress=progress)
                self.events.put(("trained", model, metadata))
            except Exception as error:
                self.events.put(("error", str(error)))
        threading.Thread(target=worker, daemon=True).start()

    def analyze(self) -> None:
        if self.current_audio is None:
            messagebox.showinfo("No audio", "Record or import audio first.")
            return
        if self.model is None:
            messagebox.showinfo("No trained model", "Record labeled samples and train the neural network first.")
            return
        probabilities = predict(self.model, self.current_audio)
        ranked = sorted(probabilities, key=probabilities.get, reverse=True)
        winner, runner_up = ranked[:2]
        confidence = probabilities[winner]
        margin = confidence - probabilities[runner_up]
        if confidence < 0.48 or margin < 0.12:
            title = f"Uncertain · leaning {winner.title()} {confidence:.1%}"
            color = "#ffd479"
        else:
            title = f"{winner.title()} · {confidence:.1%}"
            color = COLORS[winner]
        self.prediction_title.configure(text=title, fg=color)
        self._draw_confidences(probabilities)
        self.status.configure(text="ANALYSIS COMPLETE", fg="#70e5aa")

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "recorded":
                    self.current_audio = event[1]
                    self.record_button.configure(state="normal")
                    self.test_record_button.configure(state="normal")
                    self.timer.configure(text=f"{DURATION_SECONDS}.0 seconds")
                    self.status.configure(text="SAMPLE SAVED" if event[2] else "TEST AUDIO READY", fg="#70e5aa")
                    self._draw_audio()
                    self._update_dataset_counts()
                elif kind == "progress":
                    _, epoch, loss, training_accuracy, validation_accuracy = event
                    self.training_label.configure(
                        text=(
                            f"Epoch {epoch:>2}/35   loss {loss:.4f}\n"
                            f"training {training_accuracy:.1%}   validation {validation_accuracy:.1%}"
                        )
                    )
                elif kind == "trained":
                    self.model = event[1]
                    metadata = event[2]
                    self.train_button.configure(state="normal")
                    self.training_label.configure(
                        text=(
                            f"Model ready · {metadata['sample_count']} {metadata['data_source']} samples\n"
                            f"best held-out validation accuracy {metadata['validation_accuracy']:.1%}"
                        )
                    )
                    self.status.configure(text="MODEL READY", fg="#70e5aa")
                else:
                    self.record_button.configure(state="normal")
                    self.test_record_button.configure(state="normal")
                    self.train_button.configure(state="normal")
                    self.status.configure(text="ERROR", fg="#ff7c89")
                    messagebox.showerror("Cannot continue", event[1])
        except queue.Empty:
            pass
        self.after(80, self._poll_events)

    def _draw_audio(self) -> None:
        assert self.current_audio is not None
        self.update_idletasks()
        self.waveform.delete("all")
        width, height = max(self.waveform.winfo_width(), 400), 135
        self.waveform.create_line(0, height / 2, width, height / 2, fill="#2c3a53")
        indices = np.linspace(0, len(self.current_audio) - 1, min(width, len(self.current_audio))).astype(int)
        values = self.current_audio[indices]
        points = []
        for index, value in enumerate(values):
            points.extend((index * width / max(len(values) - 1, 1), height / 2 - float(value) * height * 0.43))
        self.waveform.create_line(*points, fill="#68d8ff", width=1)

        feature = audio_to_spectrogram(self.current_audio).squeeze(0).numpy()
        self.spectrogram.delete("all")
        spec_width, spec_height = max(self.spectrogram.winfo_width(), 500), 175
        columns, rows = 96, 48
        x_indices = np.linspace(0, feature.shape[1] - 1, columns).astype(int)
        y_indices = np.linspace(0, feature.shape[0] - 1, rows).astype(int)
        sampled = feature[np.ix_(y_indices, x_indices)]
        sampled = (sampled - sampled.min()) / (sampled.max() - sampled.min() + 1e-6)
        for row in range(rows):
            for column in range(columns):
                value = float(sampled[rows - row - 1, column])
                color = self._heat_color(value)
                x1, y1 = column * spec_width / columns, row * spec_height / rows
                self.spectrogram.create_rectangle(x1, y1, (column + 1) * spec_width / columns + 1, (row + 1) * spec_height / rows + 1, fill=color, outline="")

    @staticmethod
    def _heat_color(value: float) -> str:
        red = int(30 + 220 * value)
        green = int(30 + 150 * max(0.0, value - 0.35) / 0.65)
        blue = int(70 + 150 * (1.0 - value))
        return f"#{red:02x}{green:02x}{blue:02x}"

    def _draw_confidences(self, probabilities: dict[str, float]) -> None:
        self.confidence_canvas.delete("all")
        width = max(self.confidence_canvas.winfo_width(), 380)
        for index, emotion in enumerate(EMOTIONS):
            y = 8 + index * 32
            probability = probabilities.get(emotion, 0.0)
            self.confidence_canvas.create_text(5, y + 10, text=emotion.title(), fill="#cbd4e5", anchor="w", font=("Helvetica", 10, "bold"))
            self.confidence_canvas.create_rectangle(85, y, width - 50, y + 20, fill="#202c42", outline="")
            self.confidence_canvas.create_rectangle(85, y, 85 + (width - 135) * probability, y + 20, fill=COLORS[emotion], outline="")
            self.confidence_canvas.create_text(width - 5, y + 10, text=f"{probability:.1%}", fill="#eaf0fb", anchor="e", font=("Menlo", 9))

    def _update_dataset_counts(self) -> None:
        files = dataset_files(DATA_ROOT)
        training_files, source = training_dataset_files(DATA_ROOT)
        lines = [
            f"{emotion:<9} {len(files[emotion]):>3} total · {len(training_files[emotion]):>2} training"
            for emotion in EMOTIONS
        ]
        lines.append(f"\nActive source: {source}")
        self.dataset_label.configure(text="\n".join(lines))

    def _load_checkpoint_if_present(self) -> None:
        if not CHECKPOINT.exists():
            return
        try:
            self.model, metadata = EmotionCNN.load(CHECKPOINT)
            accuracy = metadata.get("validation_accuracy")
            suffix = f" · validation {accuracy:.1%}" if isinstance(accuracy, float) else ""
            self.training_label.configure(text=f"Saved model loaded · {metadata.get('sample_count', '?')} samples{suffix}")
            self.status.configure(text="MODEL LOADED", fg="#70e5aa")
        except Exception:
            self.model = None


if __name__ == "__main__":
    EmotionApp().mainloop()
