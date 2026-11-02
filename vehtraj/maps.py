"""Lanelet2 OSM geometry using INTERACTION's unmodified official projector."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def prepare_map(source, output=None, lat_origin=0.0, lon_origin=0.0):
    from .official.map_vis_without_lanelet import LL2XYProjector
    source = Path(source)
    tree = ET.parse(source).getroot()
    projector = LL2XYProjector(lat_origin, lon_origin)
    nodes = {n.attrib['id']: list(map(float, projector.latlon2xy(float(n.attrib['lat']), float(n.attrib['lon']))))
             for n in tree.findall('node')}
    if not nodes:
        raise ValueError('OSM map has no nodes')
    ways = []
    for way in tree.findall('way'):
        tags = {t.attrib['k']: t.attrib['v'] for t in way.findall('tag')}
        if 'type' not in tags:
            raise ValueError(f"Map way {way.attrib['id']} has no type")
        points = [nodes[nd.attrib['ref']] for nd in way.findall('nd')]
        ways.append({'id': way.attrib['id'], 'type': tags['type'], 'subtype': tags.get('subtype', ''), 'points': points})
    # Source .osm_xy is used only as an independent consistency check, never to shift the map.
    comparison = source.with_suffix('.osm_xy')
    max_error = None
    if comparison.exists():
        xy = {n.attrib['id']: [float(n.attrib['x']), float(n.attrib['y'])] for n in ET.parse(comparison).getroot().findall('node')}
        if set(nodes) != set(xy):
            raise ValueError('OSM and OSM_XY node identities differ')
        max_error = max(math.dist(nodes[key], xy[key]) for key in nodes)
        if max_error > 0.001:
            raise ValueError(f'Official projection differs from source OSM_XY by {max_error} metres')
    xs, ys = zip(*nodes.values())
    result = {'scenario': source.stem, 'filename': source.name,
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'projection': {'method': 'official LL2XYProjector: WGS84 UTM minus projected origin',
                             'lat_origin': lat_origin, 'lon_origin': lon_origin, 'utm_zone': projector.zone},
              'official_commit': '053213d8aeb4c6f6132d272325a11ae623bd4e35',
              'node_count': len(nodes), 'way_count': len(ways), 'ways': ways,
              'bounds': [min(xs), min(ys), max(xs), max(ys)], 'osm_xy_max_error_m': max_error}
    output = Path(output) if output else source.with_suffix('.map.json')
    output.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
    return result


def load_map(source):
    """Read the projected cache without importing pyproj, checking source identity."""
    source = Path(source)
    cache = source.with_suffix('.map.json')
    result = json.loads(cache.read_text(encoding='utf-8'))
    if result['source_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError('Map changed; rebuild projection with scripts/prepare_official_map.py')
    return result
