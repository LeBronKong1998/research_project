"""Build an offline, source-backed raw trajectory explorer using only stdlib."""
import base64
import csv
import hashlib
import io
import json
import math
from pathlib import Path

REQUIRED_COLUMNS = {'case_id', 'track_id', 'frame_id', 'timestamp_ms', 'x', 'y', 'vx', 'vy', 'psi_rad', 'length', 'width'}


def build_explorer(csv_path, output, *, template=None):
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
    template = Path(template) if template else Path(__file__).parent / 'web' / 'explorer.html'
    page = template.read_text(encoding='utf-8')
    marker = '__TRAJECTORY_PAYLOAD__'
    if page.count(marker) != 1:
        raise ValueError('Explorer template must contain exactly one payload marker')
    encoded = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page.replace(marker, encoded), encoding='utf-8')
    return {key: value for key, value in payload.items() if key != 'csv_base64'} | {'output': str(output)}
