"""Build an offline, source-backed raw trajectory explorer using only stdlib."""
import base64
import csv
import hashlib
import io
import json
import math
from pathlib import Path

REQUIRED_COLUMNS = {'case_id', 'track_id', 'frame_id', 'timestamp_ms', 'x', 'y', 'vx', 'vy', 'psi_rad', 'length', 'width'}


def build_explorer(csv_path, output, *, template=None, review_files=None, default_mode="raw"):
    csv_path, output = Path(csv_path), Path(output)
    content = csv_path.read_bytes()
    reader = csv.DictReader(io.StringIO(content.decode('utf-8-sig')))
    if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
        raise ValueError(f'Missing CSV columns: {sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))}')
    rows = list(reader)
    if not rows:
        raise ValueError('CSV is empty')
    seen, cases, tracks = set(), set(), set()
    for row in rows:
        key = (row['case_id'], row['track_id'], int(row['frame_id']))
        if key in seen:
            raise ValueError(f'Duplicate case/track/frame: {key}')
        seen.add(key)
        cases.add(row['case_id'])
        tracks.add(key[:2])
        for name in REQUIRED_COLUMNS - {'case_id', 'track_id'}:
            if row[name] == '' and name in {'psi_rad', 'length', 'width'}:
                continue
            if not math.isfinite(float(row[name])):
                raise ValueError(f'Non-finite {name}')
    payload = {
        'filename': csv_path.name,
        'csv_base64': base64.b64encode(content).decode('ascii'),
        'sha256': hashlib.sha256(content).hexdigest(),
        'row_count': len(rows), 'case_count': len(cases), 'track_count': len(tracks),
        'dataset': 'INTERACTION · InterHub 公开真实路口子集',
        'source_url': 'https://github.com/zxc-tju/InterHub/tree/1d8a764554017618447247e5c55eff4b4ab50de4/data/0_origin_datasets/interaction_multi',
    }
    from .maps import load_map
    scenario = csv_path.stem.removesuffix("_train").removesuffix("_val")
    map_path = csv_path.parent / "maps" / (scenario + ".osm")
    if not map_path.exists():
        raise FileNotFoundError(f"No matching official map: {map_path}")
    payload["map"] = load_map(map_path)
    payload["official_license"] = (Path(__file__).parent / "official" / "LICENSE").read_text(encoding="utf-8")
    payload["reviews"] = {}
    if review_files is None:
        report_root = csv_path.parent.parent.parent / "reports"
        review_files = sorted(report_root.glob("*-review.json"))
    for file in review_files:
        review = json.loads(Path(file).read_text(encoding="utf-8"))
        if review["dataset_hashes"].get(csv_path.name) != payload["sha256"]:
            raise ValueError("Review dataset hash differs from raw CSV")
        validate_review(review, rows, csv_path.name)
        payload["reviews"][review["split"]] = review
    payload["default_mode"] = default_mode
    template = Path(template) if template else Path(__file__).parent / 'web' / 'explorer.html'
    page = template.read_text(encoding='utf-8')
    script_path = template.with_name('explorer.js')
    if '__TRAJECTORY_SCRIPT__' in page:
        page = page.replace('__TRAJECTORY_SCRIPT__', script_path.read_text(encoding='utf-8'))
    marker = '__TRAJECTORY_PAYLOAD__'
    if page.count(marker) != 1:
        raise ValueError('Explorer template must contain exactly one payload marker')
    encoded = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page.replace(marker, encoded), encoding='utf-8')
    return {key: value for key, value in payload.items() if key not in {'csv_base64', 'map', 'reviews', 'official_license'}} | {'output': str(output)}


def validate_review(review, rows, filename):
    """Reject mismatched targets/frames instead of displaying misaligned predictions."""
    raw = {(r['case_id'], r['track_id'], int(r['frame_id'])): r for r in rows}
    if len(review['samples']) != review['sample_count'] or not review['samples']:
        raise ValueError('Review sample count is invalid')
    if review['split'] not in {'val', 'test'}:
        raise ValueError('Only held-out validation/test reviews are supported')
    for sample in review['samples']:
        if sample['source_csv'] != filename:
            raise ValueError('Review has a different source CSV')
        cid, tid = sample['case_id'], sample['track_id']
        for frames_key, xy_key in [('history_frames', 'history_xy'), ('future_frames', 'truth_xy')]:
            if len(sample[frames_key]) != len(sample[xy_key]):
                raise ValueError('Review frame/position counts differ')
            for frame, point in zip(sample[frames_key], sample[xy_key]):
                row = raw[(cid, tid, frame)]
                if math.dist([float(row['x']),float(row['y'])], point) > .001:
                    raise ValueError('Review world position differs from raw CSV')
        if len(sample['probabilities']) != len(sample['prediction_xy']):
            raise ValueError('Review modes/probabilities differ')
        for mode in sample['prediction_xy']:
            if len(mode) != len(sample['future_frames']):
                raise ValueError('Review prediction horizon differs')
            if not all(len(xy) == 2 and all(math.isfinite(v) for v in xy) for xy in mode):
                raise ValueError('Review predictions must be finite world coordinates')
