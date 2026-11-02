"""Download the real INTERACTION subset distributed by the InterHub authors."""
import csv
import hashlib
import json
import urllib.request
from pathlib import Path

SOURCE_COMMIT = "1d8a764554017618447247e5c55eff4b4ab50de4"
FILENAME = "DR_USA_Intersection_GL_train.csv"
SOURCE_URL = (f"https://raw.githubusercontent.com/zxc-tju/InterHub/{SOURCE_COMMIT}/"
              f"data/0_origin_datasets/interaction_multi/train/{FILENAME}")
CSV_SHA256 = "c59094abd09f3638268b916a8f24a1b83458ace02d2d8dd4d6861082401b6970"
CSV_SIZE = 1535255


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download_interaction(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / FILENAME
    if target.exists():
        if target.stat().st_size != CSV_SIZE or sha256(target) != CSV_SHA256:
            raise ValueError(f"Existing CSV differs from pinned source; preserve/relocate it first: {target}")
    else:
        partial = target.with_suffix(".csv.part")
        # Exclusive creation prevents concurrent download clobbering.
        with partial.open("xb") as stream:
            try:
                request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "trajectory-prediction/0.2"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                stream.flush()
                if partial.stat().st_size != CSV_SIZE or sha256(partial) != CSV_SHA256:
                    raise ValueError("Downloaded CSV failed pinned SHA-256/size verification")
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
        partial.replace(target)
    download_maps(output / "maps")
    with target.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    manifest = {"dataset": "INTERACTION multi-agent, InterHub authors' real subset",
                "source_url": SOURCE_URL, "source_commit": SOURCE_COMMIT,
                "sha256": CSV_SHA256, "bytes": CSV_SIZE, "row_count": len(rows),
                "case_count": len({r["case_id"] for r in rows}),
                "license_note": "Bundled unmodified upstream CSV subset; attribution and upstream license are in data/interaction. Original dataset terms apply.",
                "scope": "52 cases from DR_USA_Intersection_GL, not the full benchmark"}
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"output": str(target), **manifest}


def download_maps(output):
    """Restore only the two pinned matching map source files; never guess another map."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected = {
        'DR_USA_Intersection_GL.osm': '96d9b096e70bab53f5b55e167dfcaaf6fdd9b68adb4ddb5c25cef38cbcf24db0',
        'DR_USA_Intersection_GL.osm_xy': '02e741afc862b1bc30a29075289240411b6698818eef78bc2bbd0d66c6f1b803',
    }
    for name, digest in expected.items():
        target = output / name
        if target.exists():
            if sha256(target) != digest:
                raise ValueError(f'Existing map differs from pinned source: {target}')
            continue
        url = f'https://raw.githubusercontent.com/zxc-tju/InterHub/{SOURCE_COMMIT}/data/0_origin_datasets/interaction_multi/maps/{name}'
        part = target.with_suffix(target.suffix + '.part')
        with part.open('xb') as stream:
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    while block := response.read(1024 * 1024): stream.write(block)
                stream.flush()
                if sha256(part) != digest: raise ValueError(f'Map checksum mismatch: {name}')
            except BaseException:
                part.unlink(missing_ok=True)
                raise
        part.replace(target)
    from .maps import load_map, prepare_map
    source = output / 'DR_USA_Intersection_GL.osm'
    if not source.with_suffix('.map.json').exists(): prepare_map(source)
    else: load_map(source)
