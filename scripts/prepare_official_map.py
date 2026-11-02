"""Project bundled Lanelet2 OSM using the pinned official INTERACTION helper."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vehtraj.maps import prepare_map
parser = argparse.ArgumentParser()
parser.add_argument('--map', type=Path, default=ROOT / 'data/interaction/maps/DR_USA_Intersection_GL.osm')
args = parser.parse_args()
result = prepare_map(args.map)
print(json.dumps({k:v for k,v in result.items() if k != 'ways'},indent=2))
