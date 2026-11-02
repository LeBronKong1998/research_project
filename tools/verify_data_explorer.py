"""Optional UI regression checks: requires Playwright and an installed Chromium.

python tools/verify_data_explorer.py --chromium /usr/bin/chromium
The training environment does not need browser tooling to use the generated HTML.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--chromium', default='/usr/bin/chromium')
args = parser.parse_args()
original = (ROOT / 'data/interaction/DR_USA_Intersection_GL_train.csv').read_bytes()
results = {}
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=args.chromium, headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1080})
    errors, requests = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('request', lambda request: requests.append(request.url))
    # Render the exact offline file. Some managed browsers disable file:// navigation.
    page.set_content((ROOT / 'docs/data-explorer.html').read_text(), wait_until='load')
    page.wait_for_selector('html[data-ready="true"]')
    assert page.locator('#case-select option').count() == 52
    assert page.locator('#stat-rows').inner_text() == '24,771'
    for option in page.locator('#case-select option').all():
        page.locator('#case-select').select_option(option.get_attribute('value'))
        assert page.locator('#raw-body tr').count() > 0
        assert page.locator('#track-select option').count() > 0
    results['all_cases_rendered'] = 52
    # Reproducible seeded selection, without generating any data.
    sequences = []
    for _ in range(2):
        page.locator('#case-select').select_option('1')
        page.locator('#seed').fill('123')
        page.locator('#seed').dispatch_event('change')
        sequence = []
        for _ in range(3):
            page.locator('#random-case').click()
            sequence.append(page.locator('#case-select').input_value())
        sequences.append(sequence)
    assert sequences[0] == sequences[1]
    results['seeded_random_cases'] = sequences[0]
    page.locator('#case-select').select_option('5')
    page.locator('#track-select').select_option('1')
    assert page.locator('#detail-type').inner_text() == 'pedestrian/bicycle'
    assert page.locator('#detail-heading').inner_text() == '原始值缺失'
    page.locator('#case-select').select_option('13')
    page.locator('#frame-slider').fill('20')
    assert page.locator('#detail-frame').inner_text() == '21'
    page.locator('#trajectory-canvas').focus()
    page.keyboard.press('ArrowRight')
    assert page.locator('#detail-frame').inner_text() == '22'
    page.keyboard.press('Space')
    page.wait_for_timeout(250)
    page.locator('#play').click()
    assert page.locator('#play').inner_text() == '播放'
    assert int(page.locator('#detail-frame').inner_text()) > 22
    page.locator('#table-mode').select_option('track')
    assert page.locator('#raw-body tr').count() == 40
    page.locator('#raw-body tr').first.click()
    assert page.locator('#detail-frame').inner_text() == '1'
    page.locator('#table-mode').select_option('frame')
    page.locator('#show-others').uncheck()
    page.locator('#show-others').check()
    page.locator('#show-labels').uncheck()
    page.locator('#show-labels').check()
    with page.expect_download() as event:
        page.locator('#download-all').click()
    downloaded = Path(event.value.path()).read_bytes()
    assert downloaded == original
    results['download_sha256'] = hashlib.sha256(downloaded).hexdigest()
    with page.expect_download() as event:
        page.locator('#download-case').click()
    rows = list(csv.DictReader(io.StringIO(Path(event.value.path()).read_text())))
    assert len(rows) == 352 and {r['case_id'] for r in rows} == {'13'}
    results['case_export_rows'] = len(rows)
    before = page.locator('#trajectory-canvas').evaluate('(canvas) => canvas.toDataURL()')
    page.locator('#zoom-in').click()
    assert before != page.locator('#trajectory-canvas').evaluate('(canvas) => canvas.toDataURL()')
    page.locator('#reset-view').click()
    assert before == page.locator('#trajectory-canvas').evaluate('(canvas) => canvas.toDataURL()')
    page.screenshot(path=str(ROOT / 'reports/data-explorer-preview.png'), full_page=True)
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    page.screenshot(path=str(ROOT / 'reports/data-explorer-mobile.png'), full_page=True)
    # The same interface reviews every real validation and test target on the map.
    page.set_viewport_size({'width': 1440, 'height': 1080})
    for split, expected_count in [('val',33),('test',111)]:
        review = json.loads((ROOT/f'reports/{split}-review.json').read_text())
        page.locator('#view-mode').select_option(split)
        assert page.locator('#summary-count').inner_text() == str(expected_count)
        assert page.locator('#summary-ade').inner_text() == f"{review['metrics']['top1ADE']:.3f}"
        assert page.locator('#track-select').is_disabled()
        page.locator('#prediction-choice').select_option('top1')
        for sample in review['samples']:
            if page.locator('#case-select').input_value() != sample['case_id']:
                page.locator('#case-select').select_option(sample['case_id'])
            page.locator('#sample-select').select_option(str(sample['sample_index']))
            assert page.locator('#track-select').input_value() == sample['track_id']
            page.locator('#frame-slider').fill('39')
            assert page.locator('#sample-step-error').inner_text() == f"{sample['metrics']['top1FDE']:.3f}"
            assert page.locator('#prediction-body tr').count() == 30
        results[f'{split}_targets_checked'] = len(review['samples'])
    # Show a fixed moving validation target, independent of whether its errors are good/bad.
    page.locator('#view-mode').select_option('val')
    page.locator('#case-select').select_option('16')
    page.locator('#sample-select').select_option('9')
    page.locator('#prediction-choice').select_option('all')
    page.locator('#frame-slider').fill('39')
    before_map = page.locator('#trajectory-canvas').evaluate('(canvas)=>canvas.toDataURL()')
    page.locator('#show-map').uncheck()
    assert before_map != page.locator('#trajectory-canvas').evaluate('(canvas)=>canvas.toDataURL()')
    page.locator('#show-map').check()
    assert before_map == page.locator('#trajectory-canvas').evaluate('(canvas)=>canvas.toDataURL()')
    page.screenshot(path=str(ROOT/'reports/model-validation-preview.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    page.screenshot(path=str(ROOT/'reports/model-validation-mobile.png'),full_page=True)
    assert not errors, errors
    assert not requests, requests
    results.update({'browser_errors': errors, 'external_requests': requests,
                    'controls': 'case/track selection, seeded random, slider, keyboard, play/pause, table, toggles, zoom/reset, exports',
                    'desktop_viewport': [1440, 1080], 'mobile_viewport': [390, 844]})
    browser.close()
(ROOT / 'reports/data-explorer-validation.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(results, ensure_ascii=False, indent=2))
