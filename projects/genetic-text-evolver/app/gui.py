"""Tkinter desktop interface with animated character output."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from app.generate import PhraseGenerator


class GeneratorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Gibberish to Phrase AI")
        self.geometry("780x560")
        self.minsize(640, 480)
        self.configure(bg="#101522")
        self.generator: PhraseGenerator | None = None
        self.results: queue.Queue[tuple[str, str]] = queue.Queue()
        self._build_ui()
        self.after(100, self._load_model)

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Accent.TButton", font=("Helvetica", 13, "bold"), padding=10)

        frame = tk.Frame(self, bg="#101522", padx=32, pady=26)
        frame.pack(fill="both", expand=True)
        tk.Label(
            frame,
            text="Gibberish → Phrase AI",
            bg="#101522",
            fg="#f4f7ff",
            font=("Helvetica", 25, "bold"),
        ).pack(anchor="w")
        tk.Label(
            frame,
            text="Enter noisy text. The trained neural network will generate a phrase character by character.",
            bg="#101522",
            fg="#9eabc5",
            font=("Helvetica", 12),
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(5, 22))

        tk.Label(frame, text="INPUT", bg="#101522", fg="#7f8da8", font=("Helvetica", 10, "bold")).pack(anchor="w")
        self.input_box = tk.Text(frame, height=5, bg="#192235", fg="#ffffff", insertbackground="white", relief="flat", padx=14, pady=12, font=("Menlo", 13))
        self.input_box.pack(fill="x", pady=(6, 14))
        self.input_box.insert("1.0", "how aer yuo")

        action_row = tk.Frame(frame, bg="#101522")
        action_row.pack(fill="x")
        self.generate_button = ttk.Button(action_row, text="Generate phrase", style="Accent.TButton", command=self._start_generation, state="disabled")
        self.generate_button.pack(side="left")
        self.status = tk.Label(action_row, text="Loading model…", bg="#101522", fg="#70b7ff", font=("Helvetica", 11))
        self.status.pack(side="left", padx=14)

        tk.Label(frame, text="GENERATED OUTPUT", bg="#101522", fg="#7f8da8", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(24, 0))
        output_frame = tk.Frame(frame, bg="#192235", padx=14, pady=12)
        output_frame.pack(fill="both", expand=True, pady=(6, 0))
        self.output = tk.Label(output_frame, text="", bg="#192235", fg="#9ff5c8", font=("Menlo", 17), wraplength=680, justify="left", anchor="nw")
        self.output.pack(fill="both", expand=True)
        self.counter = tk.Label(output_frame, text="0 characters", bg="#192235", fg="#7f8da8", font=("Helvetica", 10))
        self.counter.pack(anchor="se")

    def _load_model(self) -> None:
        def load() -> None:
            try:
                self.generator = PhraseGenerator()
                self.results.put(("loaded", ""))
            except Exception as error:
                self.results.put(("error", str(error)))
        threading.Thread(target=load, daemon=True).start()
        self.after(100, self._poll_results)

    def _start_generation(self) -> None:
        text = self.input_box.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Input needed", "Enter some text first.")
            return
        self.generate_button.configure(state="disabled")
        self.output.configure(text="")
        self.counter.configure(text="0 characters")
        self.status.configure(text="Neural network is generating…")

        def run() -> None:
            try:
                assert self.generator is not None
                self.results.put(("result", self.generator.generate(text)))
            except Exception as error:
                self.results.put(("error", str(error)))
        threading.Thread(target=run, daemon=True).start()

    def _poll_results(self) -> None:
        try:
            event, value = self.results.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_results)
            return
        if event == "loaded":
            self.status.configure(text="Model ready", fg="#70e6a8")
            self.generate_button.configure(state="normal")
        elif event == "result":
            self._animate(value)
        else:
            self.status.configure(text="Model unavailable", fg="#ff8f9c")
            messagebox.showerror("Cannot continue", value)
        self.after(100, self._poll_results)

    def _animate(self, text: str, index: int = 0) -> None:
        if index <= len(text):
            self.output.configure(text=text[:index])
            self.counter.configure(text=f"{index} / {len(text)} characters")
            if index < len(text):
                self.after(38, self._animate, text, index + 1)
                return
        self.status.configure(text="Generation complete", fg="#70e6a8")
        self.generate_button.configure(state="normal")


if __name__ == "__main__":
    GeneratorApp().mainloop()
