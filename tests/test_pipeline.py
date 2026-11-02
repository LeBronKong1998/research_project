import csv

import numpy as np
import torch

from vehtraj.data import HISTORY, FUTURE, TrajectoryDataset, interaction_scenes, split_scenes, synthetic_scenes
from vehtraj.metrics import nonconformity, scene_calibration
from vehtraj.model import TrajectoryGAT, multimodal_loss


def test_synthetic_shapes_and_disjoint_scenes():
    scenes = synthetic_scenes(20)
    ids = [{scene["id"] for scene in split_scenes(scenes, split)} for split in ("train", "val", "cal", "test")]
    assert len(set.union(*ids)) == len(scenes)
    assert sum(map(len, ids)) == len(scenes)
    sample = TrajectoryDataset(scenes[:1])[0]
    assert sample[0].shape == (HISTORY, 9, 6)
    assert sample[2].shape == (FUTURE, 6)
    assert sample[1].sum() == 6


def test_forward_backward_and_scene_calibration():
    agents, mask, truth, *_ = TrajectoryDataset(synthetic_scenes(1))[0]
    model = TrajectoryGAT(hidden=16)
    prediction, logits = model(torch.from_numpy(agents)[None], torch.from_numpy(mask)[None])
    loss = multimodal_loss(prediction, logits, torch.from_numpy(truth)[None])
    loss.backward()
    assert prediction.shape == (1, 3, FUTURE, 5)
    assert np.isfinite(nonconformity(prediction.detach().numpy()[0], truth))
    records = [{"scene_id": f"scene_{index}", "score": float(index)} for index in range(20)]
    records.append({"scene_id": "scene_0", "score": 100.0})
    calibration = scene_calibration(records)
    assert calibration["scene_count"] == 20
    assert calibration["radius_m"] == 100.0


def test_interaction_csv_adapter(tmp_path):
    path = tmp_path / "vehicle_tracks.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case_id", "track_id", "frame_id", "agent_type", "x", "y", "vx", "vy", "psi_rad"))
        for frame in range(HISTORY + FUTURE):
            for track in (1, 2):
                writer.writerow((1, track, frame, "car", frame * 0.5 + track, track * 3.5, 5, 0, 0))
    scenes = interaction_scenes(tmp_path)
    assert len(scenes) == 1
    assert len(TrajectoryDataset(scenes)) == 2
