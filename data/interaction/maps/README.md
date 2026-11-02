# Matching Lanelet2 map

`DR_USA_Intersection_GL.osm` and `.osm_xy` are the unmodified map files distributed beside this CSV by the InterHub authors, fixed to commit `1d8a764554017618447247e5c55eff4b4ab50de4`. Exact source URLs and SHA-256 values are in `provenance.json`. Dataset authors retain their rights; see the parent data attribution and upstream notices.

The map is projected with the unmodified `LL2XYProjector` from the official INTERACTION visualization repository, commit `053213d8aeb4c6f6132d272325a11ae623bd4e35`. Following its default, `lat_origin=0`, `lon_origin=0`, WGS84 UTM zone 31. The map coordinates are intentionally local/referenced coordinates; do not interpret the small OSM latitude/longitude values as the real geographic location of this USA intersection.

`DR_USA_Intersection_GL.map.json` is a browser cache of the projected map. It is not a handmade road schematic. Rebuild with `python scripts/prepare_official_map.py` after installing dependencies. All 583 node IDs are compared against the independently supplied `.osm_xy`; maximum difference is about 0.000006875 metres. No fitted translation, scaling or rotation is used.

Both the offline HTML and Matplotlib viewer share this coordinate system and official way-type conventions. The Matplotlib viewer directly calls the unmodified official `draw_map_without_lanelet` and vehicle polygon routines; the browser renders the same projected geometry. Maps are for visualization, not model inputs in this baseline.
