"""Export real model outputs with exact target/frame identity for map-based review."""
import json
import math
from pathlib import Path
import numpy as np
from .download import sha256
from .explore import build_explorer


def export_review(args, records, split, output, *, checkpoint, training=None, summary=None):
    if not records:
        raise ValueError('Cannot visualize zero evaluation targets')
    output = Path(output)
    samples = []
    for r in records:
        filename, case_id = r['scene_id'].rsplit(':', 1)
        samples.append({'sample_index': r['sample_index'], 'case_id': case_id, 'track_id': r['track_id'],
                        'source_csv': filename, 'history_frames': r['history_frames'], 'future_frames': r['future_frames'],
                        'history_xy': (r['history'][:, 0, :2] + r['reference']).tolist(),
                        'truth_xy': (r['truth'][:, :2] + r['reference']).tolist(),
                        'prediction_xy': (r['prediction'][:, :, :2] + r['reference']).tolist(),
                        'probabilities': r['probabilities'].tolist(), 'metrics': r['metrics'],
                        'baseline_metrics': r['baseline_metrics']})
    metrics = {k: float(np.mean([s['metrics'][k] for s in samples])) for k in samples[0]['metrics']}
    baseline = {k: float(np.mean([s['baseline_metrics'][k] for s in samples])) for k in samples[0]['baseline_metrics']}
    training_data = json.loads(Path(training).read_text()) if training and Path(training).exists() else None
    if training_data and training_data['data_manifest'] != args_manifest(args):
        raise ValueError('Training log and review dataset hashes differ')
    import torch
    checkpoint_meta = torch.load(checkpoint, map_location='cpu', weights_only=True)
    report = {'format_version': 1, 'selected_epoch': checkpoint_meta['epoch'], 'split': split, 'checkpoint_filename': Path(checkpoint).name,
              'checkpoint_sha256': sha256(checkpoint), 'dataset_hashes': args_manifest(args),
              'history_frames': args.history, 'future_frames': args.future, 'dt_seconds': .1,
              'sample_count': len(samples), 'metrics': metrics, 'constant_velocity': baseline,
              'summary': finite_json(summary or {}), 'training_history': training_data['epochs'] if training_data else [],
              'training_max_batches': training_data['max_batches'] if training_data else None,
              'samples': samples}
    path = output / f'{split}-review.json'
    path.write_text(json.dumps(report, separators=(',', ':'), allow_nan=False)+'\n', encoding='utf-8')
    csvs = {s['source_csv'] for s in samples}
    if len(csvs) != 1:
        raise ValueError('Map review currently requires one CSV/scenario per report; split review by scenario')
    csv_path = Path(args.data_dir) / next(iter(csvs))
    result = build_explorer(csv_path, output / f'{split}-review.html', review_files=[path], default_mode=split)
    print(f"map_review={result['output']} split={split} samples={len(samples)}", flush=True)
    return report


def args_manifest(args):
    root = Path(args.data_dir)
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*.csv'))}


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value, list): return [finite_json(v) for v in value]
    return value
