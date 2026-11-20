# Izium: pre-war vs wartime map drift

Checks that a real, free before/after pair of Izium exists, and measures how much the change hurts frame-to-map registration.

## Data (Esri World Imagery Wayback, imagery © Maxar)

| Role | Wayback release | Capture | Sensor | GSD |
|---|---|---|---|---|
| Before | 48232 (2022-04-06) | 2020-04-13 (NE corner 2020-07-09) | GeoEye-1 | 0.46 m |
| After | 45134 (2022-12-14) | 2022-08-29, whole area | GeoEye-1 | 0.46 m |

Both are downloaded as 8192×8192 mosaics at zoom 17 (0.78 m/px, ~6.4 km square, centred 49.190 N, 37.280 E) into `data/izium/` (gitignored). The JSON sidecars hold the tile origin.
Later releases (2024–2026) still use the 2022-08-29 capture, so no post-liberation (rebuilt) imagery is available yet.

```bash
python wayback_releases.py 49.2087 37.2565 ../../data/izium/releases_center.json 17
python fetch_mosaic.py 48232 49.190 37.280 16 17 ../../data/izium/izium_before_2020.png
python fetch_mosaic.py 45134 49.190 37.280 16 17 ../../data/izium/izium_after_2022.png
python drift_check.py ../../data/izium/izium_before_2020.png ../../data/izium/izium_after_2022.png results 60
```

## Result (60 synthetic frames cut from the 2022 map, SIFT + MAGSAC, 1.4 km window prior)

| Map | Localized (<15 px, ≥12 inliers) | Median inliers |
|---|---|---|
| 2022 (control) | 60/60 | 400 |
| 2020 (drift) | 1/60 | 5 |

The two captures are co-registered to about 3 px, so the failure comes from appearance, not georeferencing.
It mixes three causes:
- destroyed buildings in the centre;
- season (April, no leaves, vs late August, full canopy);
- sun and view angle (long shadows and off-nadir lean in 2020, near-nadir in 2022).

Classic features don't survive this, which is the case for a learned or semantic matcher.
