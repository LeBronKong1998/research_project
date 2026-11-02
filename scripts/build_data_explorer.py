"""Rebuild the offline raw-data viewer without installing training dependencies."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vehtraj.explore import build_explorer

parser = argparse.ArgumentParser(description='Build the standalone raw trajectory data explorer')
parser.add_argument('--csv', type=Path, default=ROOT / 'data/interaction/DR_USA_Intersection_GL_train.csv')
parser.add_argument('--output', type=Path, default=ROOT / 'docs/data-explorer.html')
parser.add_argument("--review", type=Path, action="append", default=None)
parser.add_argument("--mode", choices=("raw","val","test"), default="raw")
args = parser.parse_args()
print(json.dumps(build_explorer(args.csv, args.output, review_files=args.review, default_mode=args.mode), ensure_ascii=False, indent=2))
