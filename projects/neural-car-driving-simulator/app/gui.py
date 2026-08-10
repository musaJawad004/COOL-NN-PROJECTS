"""Tkinter UI for the neural car evolution simulator."""

from __future__ import annotations

import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from simulation import EvolutionEngine, Track, preset_track
from simulation.car import MAX_SENSOR, SENSOR_ANGLES


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "work" / "checkpoints" / "champion_car.pt"
CANVAS_WIDTH = 760
CANVAS_HEIGHT = 620


class CarSimulatorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Neural Car Driving Simulator")
        self.geometry("1240x770")
        self.minsize(1100, 700)
        self.configure(bg="#080d17")
        self.track = preset_track(1)
        self.engine = EvolutionEngine(self.track)
        self.running = False
        self.drawing = False
        self.drawn_points: list[tuple[float, float]] = []
        self.speed = tk.StringVar(value="1×")
        self.show_sensors = tk.BooleanVar(value=True)
        self.show_traffic = tk.BooleanVar(value=True)
        self._build_ui()
        self._render()

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Primary.TButton", font=("Helvetica", 11, "bold"), padding=9)
        shell = tk.Frame(self, bg="#080d17", padx=20, pady=18)
        shell.pack(fill="both", expand=True)
        header = tk.Frame(shell, bg="#080d17")
        header.pack(fill="x", pady=(0, 12))
        tk.Label(header, text="Neural Car Driving Simulator", bg="#080d17", fg="#f7f9ff", font=("Helvetica", 25, "bold")).pack(side="left")
        self.status = tk.Label(header, text="READY", bg="#15213a", fg="#77c4ff", padx=12, pady=6, font=("Helvetica", 10, "bold"))
        self.status.pack(side="right")

        body = tk.Frame(shell, bg="#080d17")
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg="#111827", padx=10, pady=10)
        left.pack(side="left", fill="y")
        self.canvas = tk.Canvas(left, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="#183725", highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._draw_start)
        self.canvas.bind("<B1-Motion>", self._draw_move)
        self.canvas.bind("<ButtonRelease-1>", self._draw_end)

        right = tk.Frame(body, bg="#080d17", padx=18)
        right.pack(side="left", fill="both", expand=True)
        self._controls(right)
        self._metrics(right)
        tk.Label(right, text="FITNESS HISTORY", bg="#080d17", fg="#71809f", font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(15, 5))
        self.chart = tk.Canvas(right, height=150, bg="#111827", highlightthickness=0)
        self.chart.pack(fill="x")
        tk.Label(right, text="GENERATION LOG", bg="#080d17", fg="#71809f", font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(15, 5))
        self.log = tk.Text(right, height=8, bg="#111827", fg="#b9c5db", relief="flat", font=("Menlo", 9), padx=10, pady=8, state="disabled")
        self.log.pack(fill="both", expand=True)

    def _controls(self, parent: tk.Widget) -> None:
        tk.Label(parent, text="EVOLUTION CONTROL", bg="#080d17", fg="#71809f", font=("Helvetica", 9, "bold")).pack(anchor="w")
        row = tk.Frame(parent, bg="#080d17")
        row.pack(fill="x", pady=(6, 10))
        self.start_button = ttk.Button(row, text="Start Evolution", style="Primary.TButton", command=self.toggle_running)
        self.start_button.pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Next Generation", command=self.next_generation).pack(side="left", padx=6)
        ttk.Button(row, text="Reset", command=self.reset).pack(side="left")

        settings = tk.Frame(parent, bg="#111827", padx=10, pady=10)
        settings.pack(fill="x")
        tk.Label(settings, text="Speed", bg="#111827", fg="#a9b5cc").grid(row=0, column=0, sticky="w")
        speed_box = ttk.Combobox(settings, values=("1×", "5×", "20×"), textvariable=self.speed, state="readonly", width=6)
        speed_box.grid(row=1, column=0, sticky="w")
        tk.Label(settings, text="Mutation (%)", bg="#111827", fg="#a9b5cc").grid(row=0, column=1, sticky="w", padx=(12, 0))
        self.mutation_entry = tk.Entry(settings, width=8, bg="#202b40", fg="white", insertbackground="white", relief="flat")
        self.mutation_entry.grid(row=1, column=1, sticky="w", padx=(12, 0), ipady=5)
        self.mutation_entry.insert(0, "10")
        ttk.Checkbutton(settings, text="Leader sensors", variable=self.show_sensors).grid(row=1, column=2, padx=(14, 0))
        ttk.Checkbutton(settings, text="Traffic cars", variable=self.show_traffic, command=self._toggle_traffic).grid(row=1, column=3, padx=(10, 0))

        tracks = tk.Frame(parent, bg="#080d17")
        tracks.pack(fill="x", pady=(10, 0))
        for number in (1, 2, 3):
            ttk.Button(tracks, text=f"Track {number}", command=lambda n=number: self.use_preset(n)).pack(side="left", padx=(0, 5))
        ttk.Button(tracks, text="Draw Track", command=self.enable_drawing).pack(side="left", padx=(5, 0))
        ttk.Button(tracks, text="Save Champion", command=self.save).pack(side="right")
        ttk.Button(tracks, text="Load Champion", command=self.load).pack(side="right", padx=5)

    def _metrics(self, parent: tk.Widget) -> None:
        cards = tk.Frame(parent, bg="#080d17")
        cards.pack(fill="x", pady=(14, 0))
        self.generation_value = self._card(cards, "GENERATION", "1", 0)
        self.alive_value = self._card(cards, "ALIVE", "80 / 80", 1)
        self.fitness_value = self._card(cards, "BEST FITNESS", "0.0", 2)
        for index in range(3):
            cards.columnconfigure(index, weight=1)
        cards2 = tk.Frame(parent, bg="#080d17")
        cards2.pack(fill="x", pady=(7, 0))
        self.checkpoint_value = self._card(cards2, "CHECKPOINTS", "0", 0)
        self.lap_value = self._card(cards2, "LAPS", "0", 1)
        self.global_value = self._card(cards2, "ALL-TIME BEST", "0.0", 2)
        for index in range(3):
            cards2.columnconfigure(index, weight=1)

    def _card(self, parent, title: str, value: str, column: int):
        box = tk.Frame(parent, bg="#111827", padx=9, pady=8)
        box.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 5, 0))
        tk.Label(box, text=title, bg="#111827", fg="#71809f", font=("Helvetica", 8, "bold")).pack(anchor="w")
        label = tk.Label(box, text=value, bg="#111827", fg="#f5f8ff", font=("Helvetica", 13, "bold"))
        label.pack(anchor="w")
        return label

    def toggle_running(self) -> None:
        if self.drawing:
            return
        if not self._apply_mutation():
            return
        self.running = not self.running
        self.start_button.configure(text="Pause" if self.running else "Continue")
        self.status.configure(text="EVOLVING" if self.running else "PAUSED", fg="#74e9ad" if self.running else "#ffd479")
        if self.running:
            self.after(1, self._tick)

    def _tick(self) -> None:
        if not self.running:
            return
        steps = {"1×": 1, "5×": 5, "20×": 20}.get(self.speed.get(), 1)
        generation_changed = False
        for _ in range(steps):
            generation_changed = self.engine.step() or generation_changed
        self._render()
        if generation_changed:
            self._update_log()
        self.after(16, self._tick)

    def next_generation(self) -> None:
        self.engine.evolve()
        self._render()
        self._update_log()

    def _render(self) -> None:
        self.canvas.delete("all")
        # Grass stripes give the circuit depth without expensive image assets.
        for y in range(0, CANVAS_HEIGHT, 40):
            self.canvas.create_rectangle(0, y, CANVAS_WIDTH, y + 20, fill="#183725", outline="")
            self.canvas.create_rectangle(0, y + 20, CANVAS_WIDTH, y + 40, fill="#163321", outline="")
        points = self.track.points + [self.track.points[0]]
        flat = [coordinate for point in points for coordinate in point]
        self.canvas.create_line(*flat, fill="#0b111a", width=self.track.road_width + 18, joinstyle="round")
        self.canvas.create_line(*flat, fill="#d85454", width=self.track.road_width + 10, dash=(14, 10), joinstyle="round")
        self.canvas.create_line(*flat, fill="#3d4654", width=self.track.road_width, joinstyle="round")
        self.canvas.create_line(*flat, fill="#c8cdd5", width=2, dash=(13, 15), joinstyle="round")
        start = self.track.points[0]
        following = self.track.points[1]
        heading = math.atan2(following[1] - start[1], following[0] - start[0])
        perpendicular = heading + math.pi / 2
        half = self.track.road_width / 2
        ends = (
            start[0] + math.cos(perpendicular) * half,
            start[1] + math.sin(perpendicular) * half,
            start[0] - math.cos(perpendicular) * half,
            start[1] - math.sin(perpendicular) * half,
        )
        self.canvas.create_line(*ends, fill="#ffffff", width=6)
        self.canvas.create_line(*ends, fill="#111111", width=2, dash=(6, 6))

        leader = self.engine.leader
        if self.engine.traffic_enabled:
            for traffic_car in self.engine.traffic:
                self._draw_vehicle(traffic_car.position, traffic_car.heading, traffic_car.color, False)
        for car in self.engine.cars:
            if car.alive:
                self._draw_car(car, car is leader)
        if leader.alive and self.show_sensors.get():
            self._draw_sensors(leader)
        self.generation_value.configure(text=f"{self.engine.generation:,}")
        self.alive_value.configure(text=f"{self.engine.alive_count} / {self.engine.population_size}")
        self.fitness_value.configure(text=f"{leader.fitness:,.1f}")
        self.checkpoint_value.configure(text=f"{leader.checkpoints}")
        self.lap_value.configure(text=f"{leader.laps}")
        self.global_value.configure(text=f"{self.engine.global_best_fitness:,.1f}")
        self._draw_chart()

    def _draw_car(self, car, leader: bool) -> None:
        self._draw_vehicle(car.position, car.heading, "#52f2a2" if leader else "#ff7069", leader)

    def _draw_vehicle(self, position, heading: float, color: str, leader: bool) -> None:
        x, y = position
        forward = (math.cos(heading), math.sin(heading))
        side = (-forward[1], forward[0])
        front = (x + forward[0] * 11, y + forward[1] * 11)
        rear_left = (x - forward[0] * 9 + side[0] * 6, y - forward[1] * 9 + side[1] * 6)
        rear_right = (x - forward[0] * 9 - side[0] * 6, y - forward[1] * 9 - side[1] * 6)
        self.canvas.create_oval(x - 8, y - 8, x + 9, y + 10, fill="#000000", outline="")
        self.canvas.create_polygon(*front, *rear_left, *rear_right, fill=color, outline="#efffff" if leader else "#202632", width=2)
        windshield = (x + forward[0] * 2, y + forward[1] * 2)
        self.canvas.create_oval(windshield[0] - 3, windshield[1] - 3, windshield[0] + 3, windshield[1] + 3, fill="#bce5ff", outline="")

    def _draw_sensors(self, car) -> None:
        for offset, sensor in zip(SENSOR_ANGLES, car.sensors):
            end = (car.position[0] + math.cos(car.heading + offset) * sensor, car.position[1] + math.sin(car.heading + offset) * sensor)
            self.canvas.create_line(*car.position, *end, fill="#58bfff", width=1)
            self.canvas.create_oval(end[0] - 2, end[1] - 2, end[0] + 2, end[1] + 2, fill="#bce8ff", outline="")

    def _draw_chart(self) -> None:
        self.chart.delete("all")
        values = self.engine.best_history
        width, height = max(self.chart.winfo_width(), 320), 150
        for fraction in (0.25, 0.5, 0.75):
            y = height * fraction
            self.chart.create_line(0, y, width, y, fill="#243047", dash=(3, 5))
        if len(values) < 2:
            self.chart.create_text(width / 2, height / 2, text="Fitness graph appears after two generations", fill="#66758e")
            return
        visible = values[-100:]
        high = max(max(visible), 1.0)
        self.chart.create_text(8, 8, text=f"max {high:,.0f}", fill="#8393ae", anchor="nw", font=("Helvetica", 9))
        coords = []
        for index, value in enumerate(visible):
            coords.extend((index * width / max(len(visible) - 1, 1), height - 10 - value / high * (height - 20)))
        self.chart.create_line(*coords, fill="#6ff0ad", width=2, smooth=True)
        self.chart.create_oval(coords[-2] - 3, coords[-1] - 3, coords[-2] + 3, coords[-1] + 3, fill="#d9ffeb", outline="")
        self.chart.create_text(width - 8, height - 7, text=f"gen {self.engine.generation}", fill="#8393ae", anchor="se", font=("Helvetica", 9))

    def _toggle_traffic(self) -> None:
        self.engine.traffic_enabled = self.show_traffic.get()
        self.engine.restart_generation()
        self._render()

    def _update_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("end", "\n".join(self.engine.log[-100:]))
        self.log.see("end")
        self.log.configure(state="disabled")

    def _apply_mutation(self) -> bool:
        try:
            percent = float(self.mutation_entry.get().strip().rstrip("%"))
            if not 0 <= percent <= 100:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid mutation", "Enter a mutation percentage from 0 to 100.")
            return False
        self.engine.mutation_rate = percent / 100
        return True

    def use_preset(self, number: int) -> None:
        self.running = False
        self.track = preset_track(number)
        self.engine.change_track(self.track)
        self.start_button.configure(text="Start Evolution")
        self.status.configure(text=f"TRACK {number}", fg="#77c4ff")
        self._render()

    def enable_drawing(self) -> None:
        self.running = False
        self.drawing = True
        self.drawn_points = []
        self.status.configure(text="DRAW A CLOSED TRACK", fg="#c89cff")
        self.start_button.configure(text="Start Evolution")

    def _draw_start(self, event) -> None:
        if self.drawing:
            self.drawn_points = [(event.x, event.y)]

    def _draw_move(self, event) -> None:
        if not self.drawing or not self.drawn_points:
            return
        last = self.drawn_points[-1]
        if math.hypot(event.x - last[0], event.y - last[1]) >= 22:
            self.drawn_points.append((event.x, event.y))
            self.canvas.create_line(*last, event.x, event.y, fill="#c89cff", width=4)

    def _draw_end(self, _event) -> None:
        if not self.drawing:
            return
        if len(self.drawn_points) < 6:
            messagebox.showinfo("Track too short", "Draw a longer loop with at least six points.")
            self.drawn_points = []
            return
        self.track = Track(self.drawn_points)
        self.engine.change_track(self.track)
        self.drawing = False
        self.status.configure(text="CUSTOM TRACK READY", fg="#74e9ad")
        self._render()

    def reset(self) -> None:
        if not messagebox.askyesno("Reset evolution", "Delete all learned neural car brains?"):
            return
        self.running = False
        self.engine.reset()
        self.start_button.configure(text="Start Evolution")
        self.status.configure(text="RESET", fg="#ffd479")
        self._update_log()
        self._render()

    def save(self) -> None:
        self.engine.save_champion(CHECKPOINT)
        self.status.configure(text="CHAMPION SAVED", fg="#74e9ad")

    def load(self) -> None:
        if not CHECKPOINT.exists():
            messagebox.showinfo("No champion", "Train and save a champion first.")
            return
        try:
            self.engine.load_champion(CHECKPOINT)
        except Exception as error:
            messagebox.showerror("Load failed", str(error))
            return
        self.status.configure(text="CHAMPION LOADED", fg="#74e9ad")
        self._render()


if __name__ == "__main__":
    CarSimulatorApp().mainloop()
