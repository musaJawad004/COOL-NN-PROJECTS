"""Desktop visualizer for the genetic text evolution algorithm."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from model.evolution import TextEvolution


class EvolutionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Genetic Text Evolver")
        self.geometry("900x650")
        self.minsize(720, 560)
        self.configure(bg="#0b1020")
        self.evolution: TextEvolution | None = None
        self.running = False
        self.max_log_lines = 2_000
        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Run.TButton", font=("Helvetica", 12, "bold"), padding=10)

        shell = tk.Frame(self, bg="#0b1020", padx=30, pady=24)
        shell.pack(fill="both", expand=True)

        tk.Label(shell, text="Genetic Text Evolver", bg="#0b1020", fg="#f7f9ff", font=("Helvetica", 26, "bold")).pack(anchor="w")
        tk.Label(
            shell,
            text="Watch random DNA strings breed and mutate until they become your target phrase.",
            bg="#0b1020",
            fg="#94a3c2",
            font=("Helvetica", 12),
        ).pack(anchor="w", pady=(4, 20))

        controls = tk.Frame(shell, bg="#151d31", padx=18, pady=16)
        controls.pack(fill="x")
        tk.Label(controls, text="TARGET PHRASE", bg="#151d31", fg="#8190ae", font=("Helvetica", 9, "bold")).grid(row=0, column=0, columnspan=5, sticky="w")
        self.target_entry = tk.Entry(controls, bg="#222d46", fg="white", insertbackground="white", relief="flat", font=("Menlo", 15))
        self.target_entry.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(5, 14), ipady=9)
        self.target_entry.insert(0, "like and subscribe")

        tk.Label(controls, text="Population", bg="#151d31", fg="#aab6ce").grid(row=2, column=0, sticky="w")
        self.population_entry = tk.Entry(controls, width=10, bg="#222d46", fg="white", insertbackground="white", relief="flat")
        self.population_entry.grid(row=3, column=0, sticky="w", ipady=6)
        self.population_entry.insert(0, "300")

        tk.Label(controls, text="Mutation (%)", bg="#151d31", fg="#aab6ce").grid(row=2, column=1, sticky="w", padx=(15, 0))
        self.mutation_entry = tk.Entry(controls, width=10, bg="#222d46", fg="white", insertbackground="white", relief="flat")
        self.mutation_entry.grid(row=3, column=1, sticky="w", padx=(15, 0), ipady=6)
        self.mutation_entry.insert(0, "2")

        tk.Label(controls, text="Display speed", bg="#151d31", fg="#aab6ce").grid(row=2, column=2, sticky="w", padx=(15, 0))
        self.speed_choice = ttk.Combobox(controls, values=("Slow", "Normal", "Fast"), state="readonly", width=9)
        self.speed_choice.grid(row=3, column=2, sticky="w", padx=(15, 0), ipady=4)
        self.speed_choice.set("Normal")

        self.run_button = ttk.Button(controls, text="Start evolution", style="Run.TButton", command=self.start)
        self.run_button.grid(row=3, column=3, padx=(24, 8), sticky="ew")
        self.stop_button = ttk.Button(controls, text="Pause", command=self.pause, state="disabled")
        self.stop_button.grid(row=3, column=4, sticky="ew")
        controls.columnconfigure(3, weight=1)

        result = tk.Frame(shell, bg="#10182a", padx=20, pady=18)
        result.pack(fill="x", pady=(18, 12))
        tk.Label(result, text="BEST DNA THIS GENERATION", bg="#10182a", fg="#71809f", font=("Helvetica", 9, "bold")).pack(anchor="w")
        self.best_display = tk.Text(result, height=2, bg="#10182a", fg="#f3b36b", relief="flat", font=("Menlo", 24, "bold"), wrap="char", padx=0, pady=5, state="disabled")
        self.best_display.tag_configure("match", foreground="#70f0b1")
        self.best_display.tag_configure("miss", foreground="#f3b36b")
        self.best_display.tag_configure("empty", foreground="#71809f")
        self.best_display.pack(fill="x", pady=(5, 9))
        self._set_best_display(None)
        self.progress = ttk.Progressbar(result, maximum=100)
        self.progress.pack(fill="x")

        stats = tk.Frame(result, bg="#10182a")
        stats.pack(fill="x", pady=(12, 0))
        self.generation_label = self._stat(stats, "GENERATION", "0", 0)
        self.fitness_label = self._stat(stats, "FITNESS", "0.00%", 1)
        self.matches_label = self._stat(stats, "MATCHING LETTERS", "0 / 0", 2)
        self.status_label = self._stat(stats, "STATUS", "Ready", 3)
        for column in range(4):
            stats.columnconfigure(column, weight=1)

        tk.Label(shell, text="GENERATION LOG", bg="#0b1020", fg="#71809f", font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(5, 5))
        log_frame = tk.Frame(shell, bg="#10182a")
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=11, bg="#10182a", fg="#c1cae0", insertbackground="white", relief="flat", padx=12, pady=10, font=("Menlo", 10), state="disabled", wrap="none")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _stat(self, parent: tk.Widget, title: str, value: str, column: int) -> tk.Label:
        box = tk.Frame(parent, bg="#10182a")
        box.grid(row=0, column=column, sticky="w")
        tk.Label(box, text=title, bg="#10182a", fg="#71809f", font=("Helvetica", 8, "bold")).pack(anchor="w")
        label = tk.Label(box, text=value, bg="#10182a", fg="white", font=("Helvetica", 14, "bold"))
        label.pack(anchor="w")
        return label

    def start(self) -> None:
        if not self.running and self.evolution is not None and not self.evolution.solved:
            self.running = True
            self.run_button.configure(text="Restart", command=self.restart)
            self.stop_button.configure(state="normal")
            self.after(1, self._tick)
            return
        self.restart()

    def restart(self) -> None:
        try:
            target = self.target_entry.get()
            population = int(self.population_entry.get().replace(",", "").strip())
            mutation_percent = float(self.mutation_entry.get().strip().rstrip("%"))
            if not 0 <= mutation_percent <= 100:
                raise ValueError("Mutation must be a percentage from 0 to 100.")
            mutation = mutation_percent / 100
            self.evolution = TextEvolution(target, population, mutation)
        except (ValueError, OverflowError, MemoryError) as error:
            messagebox.showerror("Invalid settings", str(error))
            return
        self.running = True
        self._clear_log()
        self.run_button.configure(text="Restart", command=self.restart)
        self.stop_button.configure(state="normal")
        self._render()
        self.after(1, self._tick)

    def pause(self) -> None:
        self.running = False
        self.run_button.configure(text="Continue", command=self.start)
        self.stop_button.configure(state="disabled")
        self.status_label.configure(text="Paused", fg="#ffd479")

    def _tick(self) -> None:
        if not self.running or self.evolution is None:
            return
        # Exactly one step is processed and painted per frame so no generation
        # disappears from either the evolving phrase or the log.
        self.evolution.step()
        self._render()
        if self.evolution.solved:
            self.running = False
            self.run_button.configure(text="Run again", command=self.restart)
            self.stop_button.configure(state="disabled")
            self.status_label.configure(text="Solved", fg="#70f0b1")
            self._append_log(f"\nTARGET REACHED in generation {self.evolution.generation}!\n")
            return
        delay = {"Slow": 250, "Normal": 65, "Fast": 12}.get(self.speed_choice.get(), 65)
        self.after(delay, self._tick)

    def _render(self) -> None:
        assert self.evolution is not None
        best = self.evolution.best
        self._set_best_display(best.phrase)
        self.progress.configure(value=best.fitness * 100)
        self.generation_label.configure(text=f"{self.evolution.generation:,}")
        self.fitness_label.configure(text=f"{best.fitness:.2%}")
        self.matches_label.configure(text=f"{self.evolution.matching_characters} / {len(self.evolution.target)}")
        if self.evolution.stagnant_generations:
            self.status_label.configure(
                text=f"Searching · {self.evolution.stagnant_generations}",
                fg="#70b7ff",
            )
        else:
            self.status_label.configure(text="Improved", fg="#70f0b1")
        self._append_log(f"Generation {self.evolution.generation:>6}: {best.phrase}  ({best.fitness:.2%})\n")

    def _set_best_display(self, phrase: str | None) -> None:
        self.best_display.configure(state="normal")
        self.best_display.delete("1.0", "end")
        if phrase is None or self.evolution is None:
            self.best_display.insert("end", "Press Start", "empty")
        else:
            for index, character in enumerate(phrase):
                tag = "match" if character == self.evolution.target[index] else "miss"
                self.best_display.insert("end", character, tag)
        self.best_display.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        line_count = int(self.log.index("end-1c").split(".")[0])
        if line_count > self.max_log_lines:
            self.log.delete("1.0", f"{line_count - self.max_log_lines + 1}.0")
        self.log.see("end")
        self.log.configure(state="disabled")


if __name__ == "__main__":
    EvolutionApp().mainloop()
