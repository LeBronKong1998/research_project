import base64
import csv
import hashlib
import io
import json
import re
from pathlib import Path

import pytest

from vehtraj.download import CSV_SHA256, CSV_SIZE, FILENAME
from vehtraj.explore import build_explorer

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_original_csv_is_complete_and_unchanged():
    content = (ROOT / 'data/interaction' / FILENAME).read_bytes()
    assert len(content) == CSV_SIZE
    assert hashlib.sha256(content).hexdigest() == CSV_SHA256
    rows = list(csv.DictReader(io.StringIO(content.decode())))
    assert len(rows) == 24771
    assert len({r['case_id'] for r in rows}) == 52
    assert len({(r['case_id'], r['track_id']) for r in rows}) == 755
    missing = [r for r in rows if not r['psi_rad']]
    assert len(missing) == 356
    assert all(r['agent_type'] == 'pedestrian/bicycle' and r['length'] == r['width'] == '' for r in missing)


def test_offline_html_embeds_identical_original_csv_and_is_reproducible(tmp_path):
    output = tmp_path / 'data-explorer.html'
    summary = build_explorer(ROOT / 'data/interaction' / FILENAME, output)
    assert summary['row_count'] == 24771 and summary['case_count'] == 52
    html = output.read_text()
    match = re.search(r'<script id="trajectory-data" type="application/json">(.*?)</script>', html, re.S)
    payload = json.loads(match[1])
    assert payload['sha256'] == CSV_SHA256
    assert base64.b64decode(payload['csv_base64']) == (ROOT / 'data/interaction' / FILENAME).read_bytes()
    assert not re.search(r'<script[^>]+src=|<link[^>]+href=', html)
    assert html == (ROOT / 'docs/data-explorer.html').read_text()


def test_malformed_csv_cannot_generate_a_misleading_explorer(tmp_path):
    source = tmp_path / 'bad.csv'
    source.write_text('case_id,track_id\n1,1\n')
    with pytest.raises(ValueError, match='Missing CSV columns'):
        build_explorer(source, tmp_path / 'bad.html')
    assert not (tmp_path / 'bad.html').exists()


def test_missing_data_is_not_synthesized(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_explorer(tmp_path / 'missing.csv', tmp_path / 'missing.html')
