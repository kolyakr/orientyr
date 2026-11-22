# Data layout

Orientyr accepts a map image and an ordered folder of frames or a video. Pass their paths directly to
`scripts/run.py`. The full benchmark datasets are external to the repository.

If you already have a benchmark dataset, use the layout below. Check it with:

```bash
python scripts/check_data.py --data data
```

The checker reports missing or truncated benchmark files by name.

## Layout

```
data/
  hackatons-data/            the organisers' task data
    global_map.png
    crops/crop_0.png … crop_49.png
  irpin/                     our showcase data (Irpin, three captures: 2017, 2022, 2025)
    maps/irpin_<year>.png + .json      the map and its georeference sidecar (M0-05)
    hires/irpin_<year>_z18.png         higher-zoom layer for 2022 and 2025 (named by the sidecar's "hires" block)
    unosat.geojson                     damage polygons, derived from source/
    source/unosat_ukraine_zenodo15088349.geojson
    flights/<preset>/<year>/           synthetic flights: preset in mild, mild_30mps, harsh, harsh_30mps
      meta.json                        camera, render preset, fps, frame count, keyframe list
      truth.jsonl                      one ground-truth record per rendered frame (meta.json "frames" of them)
      flight.mp4                       the rendered flight
      frames/frame_NNNNN.png           the keyframes meta.json lists (not every rendered frame)
    drone/<year>/               a short real-looking flight for quick checks
      truth.json                       camera + per-frame ground truth; "frames" names each frame_NNN.png
      frame_NNN.png
    preview/                           curated preview images
  video/                      phone/webcam takes over the map on a screen (M2-01); added once filmed, optional
```
