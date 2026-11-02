import csv
import hashlib
import math
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


HISTORY = 10
FUTURE = 30
DT = 0.1
MAX_AGENTS = 9


def _state(x, y, vx, vy):
    return np.stack((x, y, vx, vy, np.sin(np.arctan2(vy, vx)), np.cos(np.arctan2(vy, vx))), axis=-1)


def synthetic_scenes(count_per_kind=12, seed=7):
    rng = np.random.default_rng(seed)
    time = np.arange(HISTORY + FUTURE, dtype=np.float32) * DT
    scenes = []
    for kind in ("intersection", "merge", "lane_change"):
        for index in range(count_per_kind):
            noise = rng.normal(0, 0.25, size=4)
            if kind == "intersection":
                ego_x = -22 + (5.5 + noise[0]) * time
                ego_y = np.where(time < 3, 0, (time - 3) ** 2 * (0.25 + noise[1] * 0.1))
                other_x = np.full_like(time, 2 + noise[2])
                other_y = -19 + (4.4 + noise[3]) * time
            elif kind == "merge":
                ego_x = -20 + (5.4 + noise[0]) * time
                ego_y = -3.5 + 3.5 / (1 + np.exp(-(time - 4) * 1.7))
                other_x = -9 + (4.8 + noise[1]) * time
                other_y = np.zeros_like(time)
            else:
                ego_x = -19 + (5.2 + noise[0]) * time
                ego_y = 3.5 / (1 + np.exp(-(time - 4) * 1.8))
                other_x = -10 + (4.7 + noise[1]) * time
                other_y = np.full_like(time, 3.5)
            vehicles = []
            for x, y in ((ego_x, ego_y), (other_x, other_y)):
                vehicles.append(_state(x, y, np.gradient(x, DT), np.gradient(y, DT)))
            for neighbor in range(2, 6):
                speed = 3.7 + neighbor * 0.4 + noise[neighbor % 4]
                x = -32 + neighbor * 9 + speed * time
                y = np.full_like(time, (-1 if neighbor % 2 else 1) * 3.5)
                vehicles.append(_state(x, y, np.gradient(x, DT), np.gradient(y, DT)))
            scenes.append({"id": f"synthetic_{kind}_{index:03d}", "kind": kind,
                           "tracks": {str(i): vehicle.astype(np.float32) for i, vehicle in enumerate(vehicles)}})
    return scenes


def interaction_scenes(root, history=HISTORY, future=FUTURE):
    """Read real INTERACTION drone/challenge CSVs; never synthesize missing data."""
    if root is None:
        raise ValueError("--data-dir is required for INTERACTION CSV data")
    root = Path(root)
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No INTERACTION CSV files under {root}; run python -m vehtraj download")
    scenes = []
    for file in files:
        cases = {}
        seen = set()
        timestamps = {}
        with file.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"track_id", "frame_id", "x", "y", "vx", "vy"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError(f"{file}: missing {sorted(required - set(reader.fieldnames or ())) }")
            has_case_id = "case_id" in reader.fieldnames
            for line, row in enumerate(reader, 2):
                if row.get("agent_type", "car").lower() not in ("car", "vehicle"):
                    continue
                case = row["case_id"] if has_case_id else "recording"
                track_id, frame = row["track_id"], int(row["frame_id"])
                key = (case, track_id, frame)
                if key in seen:
                    raise ValueError(f"{file}:{line}: duplicate case/track/frame {key}")
                seen.add(key)
                x, y, vx, vy = (float(row[name]) for name in ("x", "y", "vx", "vy"))
                heading = float(row["psi_rad"]) if row.get("psi_rad") not in (None, "") else math.atan2(vy, vx)
                if not np.isfinite([x, y, vx, vy, heading]).all():
                    raise ValueError(f"{file}:{line}: non-finite state")
                if row.get("timestamp_ms"):
                    stamp = float(row["timestamp_ms"])
                    if not np.isfinite(stamp):
                        raise ValueError(f"{file}:{line}: non-finite timestamp")
                    timestamp_key = (case, frame)
                    if timestamp_key in timestamps and timestamps[timestamp_key] != stamp:
                        raise ValueError(f"{file}:{line}: inconsistent frame timestamp")
                    timestamps[timestamp_key] = stamp
                cases.setdefault(case, {}).setdefault(track_id, []).append(
                    (frame, x, y, vx, vy, math.sin(heading), math.cos(heading)))
        for case, tracks in sorted(cases.items()):
            times = sorted((frame, stamp) for (cid, frame), stamp in timestamps.items() if cid == case)
            for (f0, t0), (f1, t1) in zip(times, times[1:]):
                if not np.isclose((t1 - t0) / (f1 - f0), DT * 1000):
                    raise ValueError(f"{file}: expected 10 Hz timestamps in case {case}")
            tracks = {key: np.asarray(sorted(rows), dtype=np.float64) for key, rows in tracks.items()}
            if not any(len(track) >= history + future for track in tracks.values()):
                continue
            identity = f"{file.relative_to(root).as_posix()}:{case}"
            # A continuous raw recording stays together; challenge cases stay together.
            scenes.append({"id": identity, "split_group": identity,
                           "kind": "real_interaction", "tracks": tracks})
    if not scenes:
        raise ValueError(f"No tracks have {history + future} frames; check --history/--future and release format")
    return scenes


def load_scenes(source="interaction", data_dir=None, synthetic_count=12, history=HISTORY, future=FUTURE):
    if source == "synthetic":
        return synthetic_scenes(synthetic_count)
    if source == "interaction":
        return interaction_scenes(data_dir, history, future)
    raise ValueError(source)


def split_scenes(scenes, split):
    bins = {"train": range(0, 65), "val": range(65, 75), "cal": range(75, 88), "test": range(88, 100)}
    return [scene for scene in scenes if int(hashlib.sha256(scene.get("split_group", scene["id"]).encode()).hexdigest()[:8], 16) % 100 in bins[split]]


class TrajectoryDataset(Dataset):
    def __init__(self, scenes, stride=10, max_agents=MAX_AGENTS, history=HISTORY, future=FUTURE):
        if min(stride, max_agents, history, future) <= 0:
            raise ValueError("stride, max_agents, history and future must be positive")
        self.samples = []
        self.metadata = []
        for scene in scenes:
            tracks = scene["tracks"]
            for ego_id, ego in tracks.items():
                if ego.shape[0] < history + future:
                    continue
                if ego.shape[1] == 7:
                    frames = ego[:, 0].astype(int)
                    states = ego[:, 1:]
                else:
                    frames = np.arange(len(ego))
                    states = ego
                for start in range(0, len(ego) - history - future + 1, stride):
                    window = frames[start:start + history + future]
                    if not np.all(np.diff(window) == 1):
                        continue
                    reference = states[start + history - 1, :2].copy()
                    neighbors = []
                    for track_id, track in tracks.items():
                        if track_id == ego_id:
                            continue
                        if track.shape[1] == 7:
                            indices = {int(row[0]): row[1:] for row in track}
                            neighbor_history = [indices.get(int(frame)) for frame in window[:history]]
                            if any(row is None for row in neighbor_history):
                                continue
                            neighbor_states = np.asarray(neighbor_history)
                        else:
                            if len(track) < start + history:
                                continue
                            neighbor_states = track[start:start + history]
                        distance = np.linalg.norm(neighbor_states[-1, :2] - reference)
                        neighbors.append((distance, neighbor_states))
                    neighbors.sort(key=lambda item: item[0])
                    agents = np.zeros((history, max_agents, 6), dtype=np.float32)
                    mask = np.zeros(max_agents, dtype=np.bool_)
                    agents[:, 0] = states[start:start + history]
                    mask[0] = True
                    for slot, (_, neighbor_states) in enumerate(neighbors[:max_agents - 1], 1):
                        agents[:, slot] = neighbor_states
                        mask[slot] = True
                    agents[:, mask, :2] -= reference
                    target = states[start + history:start + history + future].astype(np.float32).copy()
                    target[:, :2] -= reference
                    self.samples.append((agents, mask, target, scene["id"], scene["kind"], reference))
                    self.metadata.append({"track_id": str(ego_id), "history_frames": window[:history].tolist(),
                                          "future_frames": window[history:].tolist()})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
