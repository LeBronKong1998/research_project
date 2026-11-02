import csv
import json
import numpy as np
import pytest
import torch

from vehtraj.data import TrajectoryDataset, interaction_scenes, load_scenes, split_scenes
from vehtraj.download import download_interaction
from vehtraj.metrics import trajectory_metrics, scene_calibration
from vehtraj.model import TrajectoryGAT
from vehtraj.__main__ import constant_velocity, write_json


def make_csv(tmp_path, *, frames=40, cases=2, duplicate=False, bad_time=False, nan=False, raw=False):
    path = tmp_path / "DR_USA_Intersection_GL_train.csv"
    with path.open('w', newline='') as f:
        fields = ['track_id', 'frame_id', 'timestamp_ms', 'x', 'y', 'vx', 'vy', 'agent_type']
        if not raw:
            fields.insert(0, 'case_id')
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in range(cases):
            for frame in range(frames):
                for track in (1, 2):
                    row = dict(track_id=track, frame_id=frame, timestamp_ms=frame * (200 if bad_time else 100),
                               x='nan' if nan else frame * .5 + track, y=track * 3.5, vx=5, vy=0, agent_type='car')
                    if not raw:
                        row['case_id'] = case
                    writer.writerow(row)
                    if duplicate and frame == 0 and track == 1:
                        writer.writerow(row)
    return path


def test_real_schema_horizon_and_split_groups(tmp_path):
    make_csv(tmp_path)
    scenes = interaction_scenes(tmp_path)
    assert len(scenes) == 2
    assert len(TrajectoryDataset(scenes)) == 4
    with pytest.raises(ValueError, match='No tracks'):
        interaction_scenes(tmp_path, history=30, future=50)
    ids = [{s['id'] for s in split_scenes(scenes, name)} for name in ('train', 'val', 'cal', 'test')]
    assert sum(map(len, ids)) == len(set.union(*ids)) == 2


@pytest.mark.parametrize('option,message', [('duplicate','duplicate'), ('bad_time','10 Hz'), ('nan','non-finite')])
def test_corrupt_csv_is_rejected(tmp_path, option, message):
    make_csv(tmp_path, **{option: True})
    with pytest.raises(ValueError, match=message):
        interaction_scenes(tmp_path)


def test_raw_recording_not_randomly_split_into_time_blocks(tmp_path):
    make_csv(tmp_path, cases=1, frames=160, raw=True)
    scenes = interaction_scenes(tmp_path)
    assert len(scenes) == 1
    assert sum(bool(split_scenes(scenes, s)) for s in ('train','val','cal','test')) == 1


def test_future_neighbor_changes_cannot_change_inputs(tmp_path):
    make_csv(tmp_path, cases=1)
    scenes = interaction_scenes(tmp_path)
    before = TrajectoryDataset(scenes)[0]
    scenes[0]['tracks']['2'][10:, 1:5] += 500
    after = TrajectoryDataset(scenes)[0]
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[2], after[2])


def test_gap_does_not_produce_interpolated_trajectory(tmp_path):
    make_csv(tmp_path, cases=1)
    scenes = interaction_scenes(tmp_path)
    for k, track in scenes[0]['tracks'].items():
        scenes[0]['tracks'][k] = np.delete(track, 20, axis=0)
    assert len(TrajectoryDataset(scenes)) == 0


def test_oracle_metrics_do_not_masquerade_as_top1_accuracy():
    truth = np.zeros((30,6), np.float32)
    truth[:,5] = 1
    pred = np.zeros((2,30,5), np.float32)
    pred[0,:,0] = 3
    metrics = trajectory_metrics(pred, truth, [0.9,0.1])
    assert metrics['minADE'] == 0
    assert metrics['top1ADE'] == metrics['top1FDE'] == 3
    assert metrics['top1EndpointAccuracy2m'] == 0
    assert metrics['missRate2m'] == 0
    assert metrics['top1Displacement@3s'] == 3


def test_constant_velocity_uses_observed_state_only():
    agents = np.zeros((10,9,6), np.float32)
    agents[-1,0] = [2,3,5,0,0,1]
    pred = constant_velocity(agents,30)
    np.testing.assert_allclose(pred[0,-1,:2],[17,3])


def test_masked_agents_do_not_influence_gnn():
    torch.manual_seed(1)
    model = TrajectoryGAT(hidden=16, future=30).eval()
    x = torch.randn(1,10,9,6)
    mask = torch.zeros(1,9,dtype=torch.bool)
    mask[:,0] = True
    with torch.no_grad():
        a, _ = model(x,mask)
        x[:,:,1:] += 1000
        b, _ = model(x,mask)
    torch.testing.assert_close(a,b)


def test_unbounded_calibration_is_null_in_json(tmp_path):
    cal = scene_calibration([dict(scene_id='a',score=1)])
    assert np.isinf(cal['radius_m'])
    write_json(tmp_path/'cal.json',cal)
    assert json.loads((tmp_path/'cal.json').read_text())['radius_m'] is None


def test_downloader_does_not_overwrite_unknown_data(tmp_path):
    (tmp_path/'DR_USA_Intersection_GL_train.csv').write_text('preserve me')
    with pytest.raises(ValueError, match='Existing CSV differs'):
        download_interaction(tmp_path)
    assert (tmp_path/'DR_USA_Intersection_GL_train.csv').read_text() == 'preserve me'


def test_real_data_is_default_and_never_silently_synthetic(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scenes(data_dir=tmp_path)
