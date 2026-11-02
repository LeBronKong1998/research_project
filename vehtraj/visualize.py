from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import numpy as np


def draw_scene(path, kind, history, truth, prediction, radius=None):
    path = Path(path)
    fig, axis = plt.subplots(figsize=(8, 5))
    for vehicle in history[:, 1:].transpose(1, 0, 2):
        if np.any(vehicle):
            axis.plot(vehicle[:, 0], vehicle[:, 1], color="#94a3b8", alpha=0.6)
            axis.add_patch(Rectangle((vehicle[-1, 0] - 1.8, vehicle[-1, 1] - 0.9), 3.6, 1.8, color="#64748b", alpha=0.55))
    axis.plot(history[:, 0, 0], history[:, 0, 1], color="#2563eb", label="history")
    axis.add_patch(Rectangle((history[-1, 0, 0] - 1.8, history[-1, 0, 1] - 0.9), 3.6, 1.8, color="#2563eb"))
    axis.plot(truth[:, 0], truth[:, 1], color="#16a34a", label="ground truth")
    for mode, path_points in enumerate(prediction):
        axis.plot(path_points[:, 0], path_points[:, 1], label=f"mode {mode + 1}")
        if radius is not None and np.isfinite(radius):
            axis.plot(path_points[:, 0], path_points[:, 1] + radius, color="#f97316", alpha=0.2)
            axis.plot(path_points[:, 0], path_points[:, 1] - radius, color="#f97316", alpha=0.2)
    axis.set_title(kind.replace("_", " ").title() + " (trajectories only; no map)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def animate_scene(path, kind, history, truth, prediction, fps=10):
    path = Path(path)
    fig, axis = plt.subplots(figsize=(7, 5))
    points = np.concatenate((history[:, 0, :2], truth[:, :2], prediction[:, :, :2].reshape(-1, 2)), axis=0)
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    span = np.maximum(points.max(axis=0) - points.min(axis=0) + 12, [24, 20])
    axis.set_xlim(center[0] - span[0] / 2, center[0] + span[0] / 2)
    axis.set_ylim(center[1] - span[1] / 2, center[1] + span[1] / 2)
    axis.set_aspect("equal")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(kind.replace("_", " ").title() + " (trajectories only; no map)")
    history_line, = axis.plot([], [], color="#2563eb", linewidth=2, label="observed")
    truth_line, = axis.plot([], [], color="#16a34a", linewidth=2, label="actual future")
    colors = ("#dc2626", "#f97316", "#9333ea")
    mode_lines = [axis.plot([], [], color=colors[index % len(colors)], linewidth=1.6, label=f"predicted mode {index + 1}")[0] for index in range(len(prediction))]
    ego_car = Rectangle((0, 0), 3.6, 1.8, color="#2563eb")
    axis.add_patch(ego_car)
    neighbor_cars = []
    for slot in range(1, history.shape[1]):
        if np.any(history[:, slot]):
            car = Rectangle((0, 0), 3.6, 1.8, color="#94a3b8", alpha=0.7)
            axis.add_patch(car)
            neighbor_cars.append((slot, car))
    mode_cars = []
    for index in range(len(prediction)):
        car = Rectangle((0, 0), 3.6, 1.8, color=colors[index % len(colors)], alpha=0.55)
        car.set_visible(False)
        axis.add_patch(car)
        mode_cars.append(car)
    clock = axis.text(0.02, 0.96, "", transform=axis.transAxes, va="top")
    axis.legend(loc="lower right", fontsize=8)

    def update(frame):
        if frame < len(history):
            observed = history[:frame + 1, 0, :2]
            history_line.set_data(observed[:, 0], observed[:, 1])
            ego_car.set_xy(history[frame, 0, :2] - [1.8, 0.9])
            ego_car.set_color("#2563eb")
            for slot, car in neighbor_cars:
                car.set_visible(True)
                car.set_xy(history[frame, slot, :2] - [1.8, 0.9])
            clock.set_text(f"History: {(frame + 1) / fps:.1f} s")
        else:
            future_step = frame - len(history)
            actual = truth[:future_step + 1, :2]
            truth_line.set_data(actual[:, 0], actual[:, 1])
            ego_car.set_xy(truth[future_step, :2] - [1.8, 0.9])
            ego_car.set_color("#16a34a")
            for _, car in neighbor_cars:
                car.set_visible(False)
            for index, (line, car) in enumerate(zip(mode_lines, mode_cars)):
                forecast = prediction[index, :future_step + 1, :2]
                line.set_data(forecast[:, 0], forecast[:, 1])
                car.set_visible(True)
                car.set_xy(forecast[-1] - [1.8, 0.9])
            clock.set_text(f"Future: {(future_step + 1) / fps:.1f} s")

    animation = FuncAnimation(fig, update, frames=len(history) + len(truth), interval=1000 / fps, repeat=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
