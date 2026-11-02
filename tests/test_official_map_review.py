import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from vehtraj.maps import load_map, prepare_map
from vehtraj.explore import validate_review
from vehtraj.official_viewer import OfficialViewer

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / 'data/interaction/maps/DR_USA_Intersection_GL.osm'
CSV = ROOT / 'data/interaction/DR_USA_Intersection_GL_train.csv'


def test_unmodified_official_source_hashes():
    folder = ROOT / 'vehtraj/official'
    metadata = json.loads((folder/'provenance.json').read_text())
    assert metadata['commit'] == '053213d8aeb4c6f6132d272325a11ae623bd4e35'
    for name, digest in metadata['files'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest


def test_official_projection_matches_independent_author_xy_map(tmp_path):
    projected = prepare_map(MAP, tmp_path/'map.json')
    assert projected['node_count'] == 583
    assert projected['way_count'] == 163
    assert projected['projection']['utm_zone'] == 31
    assert projected['osm_xy_max_error_m'] < .00001
    assert projected == load_map(MAP)


def test_modified_map_is_not_silently_reused(tmp_path):
    source = tmp_path/MAP.name
    shutil.copyfile(MAP, source)
    shutil.copyfile(MAP.with_suffix('.map.json'), source.with_suffix('.map.json'))
    source.write_bytes(source.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='Map changed'):
        load_map(source)


@pytest.mark.parametrize('split,count', [('val',33),('test',111)])
def test_every_prediction_target_and_frame_matches_the_raw_csv(split,count):
    review = json.loads((ROOT/f'reports/{split}-review.json').read_text())
    with CSV.open(newline='') as f: rows=list(csv.DictReader(f))
    validate_review(review,rows,CSV.name)
    assert review['sample_count'] == count
    for sample in review['samples']:
        group=f"{sample['source_csv']}:{sample['case_id']}"
        bucket=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%100
        assert bucket in (range(65,75) if split=='val' else range(88,100))
        assert sample['history_frames'] == list(range(1,11))
        assert sample['future_frames'] == list(range(11,41))
        prediction=np.asarray(sample['prediction_xy']);truth=np.asarray(sample['truth_xy'])
        error=np.linalg.norm(prediction-truth[None],axis=-1)
        top=np.argmax(sample['probabilities'])
        assert np.isclose(error[top].mean(),sample['metrics']['top1ADE'],atol=1e-5)
        assert np.isclose(error[top,-1],sample['metrics']['top1FDE'],atol=1e-5)


def test_misaligned_review_is_rejected():
    review=json.loads((ROOT/'reports/val-review.json').read_text())
    with CSV.open(newline='') as f:rows=list(csv.DictReader(f))
    review['samples'][0]['truth_xy'][0][0] += 1
    with pytest.raises(ValueError,match='world position'):
        validate_review(review,rows,CSV.name)


def test_official_matplotlib_playback_and_error_vector_match_real_output():
    viewer=OfficialViewer(CSV,MAP,review_path=ROOT/'reports/val-review.json',sample_index=0)
    fig=viewer.build_figure()
    assert len(viewer.axes.lines) == 160  # 153 drawable map ways + 7 review artists; 10 traffic signs are skipped officially.
    viewer.slider.set_val(40)
    s=viewer.sample;top=np.argmax(s['probabilities']);truth=np.asarray(s['truth_xy'][-1]);pred=np.asarray(s['prediction_xy'][top][-1])
    np.testing.assert_allclose(viewer.error_line.get_xdata(),[pred[0],truth[0]])
    np.testing.assert_allclose(viewer.error_line.get_ydata(),[pred[1],truth[1]])
    viewer.step(-1)
    assert viewer.frames[viewer.index] == 39
    viewer.slider.set_val(10)
    assert len(viewer.error_line.get_xdata()) == 0
    plt.close(fig)
