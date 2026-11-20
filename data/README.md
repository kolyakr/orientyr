# data/

Not in git (`.gitignore` excludes `data/*`, except this file). The single source is the team's Google Drive
folder: <https://drive.google.com/drive/folders/1Z5iIhFekHIFkAjH49jxUdf3k4SGw7voe>, which this layout mirrors.
Get it with `tools/data.sh pull` (needs `tools/data.sh setup` once), or by downloading the folder in the
browser and unzipping it here. Then check it unpacked correctly:

```bash
python scripts/check_data.py
```

It reports every missing or truncated file by name; see `src/dronenav/check_data.py` for exactly what it
checks. Pull before starting work that reads data (M0-06).

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
    preview/                           curated stills used in the README / pitch
  video/                      phone/webcam takes over the map on a screen (M2-01); added once filmed, optional
```
