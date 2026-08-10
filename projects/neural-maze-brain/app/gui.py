"""Tkinter interface for training and inspecting the maze DQN."""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import torch

from maze import ACTIONS, DQNAgent, GridMaze


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "work" / "checkpoints" / "maze_brain.pt"
CELL = 42
ARROWS = ("↑", "→", "↓", "←")


class MazeBrainApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Neural Maze Brain")
        self.geometry("1180x760")
        self.minsize(1000, 690)
        self.configure(bg="#090e1a")
        self.maze = GridMaze()
        self.agent = DQNAgent(self.maze.width * self.maze.height * 3)
        self.mode = "idle"
        self.paint_mode = tk.StringVar(value="Wall")
        self.episode = 0
        self.episode_reward = 0.0
        self.successes: deque[int] = deque(maxlen=100)
        self.reward_history: deque[float] = deque(maxlen=250)
        self.astar_path: list[tuple[int, int]] = []
        self.show_policy = True
        self.visit_counts: Counter[tuple[int, int]] = Counter()
        self.recent_positions: deque[tuple[int, int]] = deque(maxlen=10)
        self.loop_escapes = 0
        self._build_ui()
        self.maze.randomize(seed=8)
        self._draw_all()

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Primary.TButton", font=("Helvetica", 11, "bold"), padding=9)
        shell = tk.Frame(self, bg="#090e1a", padx=24, pady=20)
        shell.pack(fill="both", expand=True)

        header = tk.Frame(shell, bg="#090e1a")
        header.pack(fill="x", pady=(0, 16))
        tk.Label(header, text="Neural Maze Brain", bg="#090e1a", fg="#f5f8ff", font=("Helvetica", 25, "bold")).pack(side="left")
        self.status = tk.Label(header, text="READY", bg="#15213a", fg="#7dc4ff", padx=12, pady=6, font=("Helvetica", 10, "bold"))
        self.status.pack(side="right")

        body = tk.Frame(shell, bg="#090e1a")
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg="#11192b", padx=16, pady=16)
        left.pack(side="left", fill="y")
        self.canvas = tk.Canvas(left, width=self.maze.width * CELL, height=self.maze.height * CELL, bg="#0c1322", highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._edit_cell)

        tools = tk.Frame(left, bg="#11192b")
        tools.pack(fill="x", pady=(12, 0))
        for text in ("Wall", "Erase", "Start", "Goal"):
            ttk.Radiobutton(tools, text=text, variable=self.paint_mode, value=text).pack(side="left", padx=(0, 8))
        ttk.Button(tools, text="Random Maze", command=self.random_maze).pack(side="right")
        ttk.Button(tools, text="Clear", command=self.clear_maze).pack(side="right", padx=7)

        right = tk.Frame(body, bg="#090e1a", padx=22)
        right.pack(side="left", fill="both", expand=True)
        tk.Label(right, text="TRAINING CONTROL", bg="#090e1a", fg="#70809f", font=("Helvetica", 9, "bold")).pack(anchor="w")
        buttons = tk.Frame(right, bg="#090e1a")
        buttons.pack(fill="x", pady=(7, 16))
        self.train_button = ttk.Button(buttons, text="Train Brain", style="Primary.TButton", command=self.start_training)
        self.train_button.pack(side="left", fill="x", expand=True)
        ttk.Button(buttons, text="Pause", command=self.pause).pack(side="left", padx=7)
        ttk.Button(buttons, text="Watch AI", command=self.watch).pack(side="left")

        cards = tk.Frame(right, bg="#090e1a")
        cards.pack(fill="x")
        self.episode_value = self._card(cards, "EPISODE", "0", 0)
        self.reward_value = self._card(cards, "REWARD", "0.000", 1)
        self.epsilon_value = self._card(cards, "EXPLORATION", "100.0%", 2)
        self.success_value = self._card(cards, "SUCCESS (100)", "0.0%", 3)
        for column in range(4):
            cards.columnconfigure(column, weight=1)

        second_cards = tk.Frame(right, bg="#090e1a")
        second_cards.pack(fill="x", pady=(9, 18))
        self.loss_value = self._card(second_cards, "LOSS", "—", 0)
        self.memory_value = self._card(second_cards, "MEMORY", "0", 1)
        self.steps_value = self._card(second_cards, "EPISODE STEPS", "0", 2)
        for column in range(3):
            second_cards.columnconfigure(column, weight=1)

        tk.Label(right, text="REWARD HISTORY", bg="#090e1a", fg="#70809f", font=("Helvetica", 9, "bold")).pack(anchor="w")
        self.chart = tk.Canvas(right, height=180, bg="#11192b", highlightthickness=0)
        self.chart.pack(fill="x", pady=(7, 16))

        options = tk.Frame(right, bg="#090e1a")
        options.pack(fill="x")
        ttk.Button(options, text="Show A*", command=self.toggle_astar).pack(side="left")
        ttk.Button(options, text="Policy Arrows", command=self.toggle_policy).pack(side="left", padx=7)
        ttk.Button(options, text="Save Brain", command=self.save_brain).pack(side="right")
        ttk.Button(options, text="Load Brain", command=self.load_brain).pack(side="right", padx=7)
        ttk.Button(options, text="Reset Brain", command=self.reset_brain).pack(side="right")

        tk.Label(
            right,
            text="Blue = start/agent   Green = goal   Purple = A* path   Arrows = neural policy confidence",
            bg="#090e1a",
            fg="#71809f",
            font=("Helvetica", 10),
        ).pack(anchor="w", pady=(18, 0))

    def _card(self, parent: tk.Widget, title: str, value: str, column: int) -> tk.Label:
        frame = tk.Frame(parent, bg="#11192b", padx=10, pady=9)
        frame.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 5, 0))
        tk.Label(frame, text=title, bg="#11192b", fg="#71809f", font=("Helvetica", 8, "bold")).pack(anchor="w")
        label = tk.Label(frame, text=value, bg="#11192b", fg="#f4f7ff", font=("Helvetica", 14, "bold"))
        label.pack(anchor="w")
        return label

    def _draw_all(self) -> None:
        self.canvas.delete("all")
        policy: dict[tuple[int, int], tuple[int, float]] = {}
        if self.show_policy:
            cells = [(x, y) for y in range(self.maze.height) for x in range(self.maze.width) if (x, y) not in self.maze.walls]
            states = torch.stack([self.maze.observation_at(cell) for cell in cells])
            masks = torch.stack([self.maze.action_mask(cell) for cell in cells])
            actions, confidence = self.agent.policy(states, masks)
            policy = {cell: (int(action), float(score)) for cell, action, score in zip(cells, actions, confidence)}

        for y in range(self.maze.height):
            for x in range(self.maze.width):
                cell = (x, y)
                x1, y1 = x * CELL, y * CELL
                fill = "#26324a" if cell in self.maze.walls else "#101a2d"
                if cell in self.astar_path:
                    fill = "#31275a"
                if cell == self.maze.goal:
                    fill = "#21654b"
                if cell == self.maze.start:
                    fill = "#174c78"
                self.canvas.create_rectangle(x1, y1, x1 + CELL, y1 + CELL, fill=fill, outline="#24304a")
                if cell in policy and cell not in (self.maze.goal, self.maze.agent):
                    action, confidence = policy[cell]
                    color = "#567096" if confidence < 0.45 else "#80a9dc"
                    self.canvas.create_text(x1 + CELL / 2, y1 + CELL / 2, text=ARROWS[action], fill=color, font=("Helvetica", 12, "bold"))
        ax, ay = self.maze.agent
        self.canvas.create_oval(ax * CELL + 8, ay * CELL + 8, (ax + 1) * CELL - 8, (ay + 1) * CELL - 8, fill="#56b9ff", outline="#d9f2ff", width=2)
        gx, gy = self.maze.goal
        self.canvas.create_text(gx * CELL + CELL / 2, gy * CELL + CELL / 2, text="★", fill="#c3ffe1", font=("Helvetica", 19, "bold"))
        self._draw_chart()
        self._update_metrics()

    def _edit_cell(self, event) -> None:
        if self.mode != "idle":
            return
        cell = (event.x // CELL, event.y // CELL)
        mode = self.paint_mode.get()
        if mode == "Wall":
            self.maze.set_wall(cell)
        elif mode == "Erase":
            self.maze.set_wall(cell, False)
        elif mode == "Start":
            self.maze.set_start(cell)
        else:
            self.maze.set_goal(cell)
        self.astar_path = []
        self.maze.reset()
        self._reset_episode_memory()
        self._draw_all()

    def start_training(self) -> None:
        if self.mode == "training":
            return
        self.mode = "training"
        self.status.configure(text="TRAINING", fg="#7dc4ff")
        self.train_button.configure(text="Training…")
        if not self.recent_positions:
            self._reset_episode_memory()
        self.after(1, self._tick)

    def watch(self) -> None:
        if self.mode == "watching":
            return
        self.mode = "watching"
        self.maze.reset()
        self.episode_reward = 0.0
        self._reset_episode_memory()
        self.status.configure(text="WATCHING AI", fg="#c89cff")
        self.after(1, self._tick)

    def pause(self) -> None:
        self.mode = "idle"
        self.status.configure(text="PAUSED", fg="#ffd479")
        self.train_button.configure(text="Train Brain")

    def _tick(self) -> None:
        if self.mode not in ("training", "watching"):
            return
        state = self.maze.observation()
        valid_actions = self.maze.valid_actions()
        if not valid_actions:
            self.status.configure(text="NO EXIT — EDIT MAZE", fg="#ff8f9c")
            self.mode = "idle"
            self.train_button.configure(text="Train Brain")
            return
        action = self._choose_action(state, valid_actions)
        next_state, reward, done = self.maze.step(action)
        self.recent_positions.append(self.maze.agent)
        self.visit_counts[self.maze.agent] += 1
        self.episode_reward += reward
        if self.mode == "training":
            self.agent.memory.add(
                state,
                action,
                reward,
                next_state,
                done,
                self.maze.action_mask(),
            )
            self.agent.learn()

        if done:
            success = self.maze.agent == self.maze.goal
            self.successes.append(int(success))
            self.reward_history.append(self.episode_reward)
            self.episode += 1
            if self.mode == "training":
                self.agent.finish_episode()
                self.maze.reset()
                self._reset_episode_memory()
                self.episode_reward = 0.0
            else:
                self.mode = "idle"
                self.status.configure(text="GOAL REACHED" if success else "WATCH FINISHED", fg="#70efae" if success else "#ffd479")

        self._draw_all()
        delay = 8 if self.mode == "training" else 120
        if self.mode in ("training", "watching"):
            self.after(delay, self._tick)

    def _choose_action(self, state: torch.Tensor, valid_actions: list[int]) -> int:
        """Use the DQN normally, but break repeated local cycles safely."""
        position = self.maze.agent
        repeated_square = self.visit_counts[position] >= 3
        two_cell_cycle = (
            len(self.recent_positions) >= 4
            and self.recent_positions[-1] == self.recent_positions[-3]
            and self.recent_positions[-2] == self.recent_positions[-4]
        )
        if repeated_square or two_cell_cycle:
            least_visits = min(
                self.visit_counts[
                    (position[0] + ACTIONS[action][0], position[1] + ACTIONS[action][1])
                ]
                for action in valid_actions
            )
            escape_actions = [
                action
                for action in valid_actions
                if self.visit_counts[
                    (position[0] + ACTIONS[action][0], position[1] + ACTIONS[action][1])
                ] == least_visits
            ]
            self.loop_escapes += 1
            self.status.configure(text="ESCAPING LOOP", fg="#ffd479")
            return self.agent.act(state, explore=False, valid_actions=escape_actions)
        self.status.configure(
            text="TRAINING" if self.mode == "training" else "WATCHING AI",
            fg="#7dc4ff" if self.mode == "training" else "#c89cff",
        )
        return self.agent.act(
            state,
            explore=self.mode == "training",
            valid_actions=valid_actions,
        )

    def _reset_episode_memory(self) -> None:
        self.visit_counts.clear()
        self.visit_counts[self.maze.agent] = 1
        self.recent_positions.clear()
        self.recent_positions.append(self.maze.agent)

    def _update_metrics(self) -> None:
        self.episode_value.configure(text=f"{self.episode:,}")
        self.reward_value.configure(text=f"{self.episode_reward:.3f}")
        self.epsilon_value.configure(text=f"{self.agent.epsilon:.1%}")
        rate = sum(self.successes) / len(self.successes) if self.successes else 0.0
        self.success_value.configure(text=f"{rate:.1%}")
        self.loss_value.configure(text=f"{self.agent.last_loss:.5f}" if self.agent.training_steps else "—")
        self.memory_value.configure(text=f"{len(self.agent.memory):,}")
        self.steps_value.configure(text=f"{self.maze.steps:,}")

    def _draw_chart(self) -> None:
        self.chart.delete("all")
        width = max(self.chart.winfo_width(), 300)
        height = 180
        self.chart.create_line(0, height / 2, width, height / 2, fill="#26334c")
        if len(self.reward_history) < 2:
            self.chart.create_text(width / 2, height / 2, text="Reward history appears after completed episodes", fill="#667795")
            return
        values = list(self.reward_history)
        low, high = min(values), max(values)
        spread = max(high - low, 0.01)
        points = []
        for index, value in enumerate(values):
            x = index * width / max(len(values) - 1, 1)
            y = height - 10 - (value - low) / spread * (height - 20)
            points.extend((x, y))
        self.chart.create_line(*points, fill="#70efae", width=2, smooth=True)

    def random_maze(self) -> None:
        self.pause()
        self.maze.randomize()
        self.astar_path = []
        self._reset_episode_memory()
        self._draw_all()

    def clear_maze(self) -> None:
        self.pause()
        self.maze.clear()
        self.astar_path = []
        self._reset_episode_memory()
        self._draw_all()

    def toggle_astar(self) -> None:
        self.astar_path = [] if self.astar_path else self.maze.shortest_path()
        if not self.astar_path and self.maze.start != self.maze.goal:
            messagebox.showinfo("No route", "The goal is blocked. Erase at least one wall.")
        self._draw_all()

    def toggle_policy(self) -> None:
        self.show_policy = not self.show_policy
        self._draw_all()

    def reset_brain(self) -> None:
        if not messagebox.askyesno("Reset brain", "Erase everything this neural network has learned?"):
            return
        self.pause()
        self.agent = DQNAgent(self.maze.width * self.maze.height * 3)
        self.episode = 0
        self.successes.clear()
        self.reward_history.clear()
        self.maze.reset()
        self._draw_all()

    def save_brain(self) -> None:
        self.agent.save(CHECKPOINT)
        self.status.configure(text="BRAIN SAVED", fg="#70efae")

    def load_brain(self) -> None:
        if not CHECKPOINT.exists():
            messagebox.showinfo("No saved brain", "Train and save a brain first.")
            return
        try:
            self.agent.load(CHECKPOINT)
        except Exception as error:
            messagebox.showerror("Load failed", str(error))
            return
        self.status.configure(text="BRAIN LOADED", fg="#70efae")
        self._draw_all()


if __name__ == "__main__":
    MazeBrainApp().mainloop()
