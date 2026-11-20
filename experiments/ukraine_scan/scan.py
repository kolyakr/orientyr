"""Where in Ukraine can we get ONE clean satellite image per epoch (pre-war, war, post-war)?

For every UNOSAT damage-assessment area (AOI) we query the metadata layer of every Esri Wayback
release. Each feature there is the visible part of one Maxar capture (date + sensor) in that
release. A capture is the same photograph in every release, so the union of its polygons over
all releases is the capture's true footprint (minus Esri's cloud cut-outs).

Every footprint is rasterised onto a coarse Web-Mercator grid. Then, for each square size and each
triple of captures (pre, war, post), we look for the square that lies ENTIRELY inside all three
footprints and holds the most UNOSAT destroyed/severe points. A square that passes this test can
be cut from three single-date photographs: no seams, no mixed colours.

Epochs: pre < 2022-02-24 <= war <= 2022-12-31 < post (post starts 2023-04-01: rebuilt / long after).

Writes <out>/candidates.json (every AOI x size x triple that has a feasible square, ranked) and
prints a summary table. Metadata answers are cached in <cache>/<aoi>/<release>.json.

Usage: python scan.py <unosat_labels.geojson> <unosat_aois.geojson> <cache_dir> <out_dir>
"""
import datetime as dt
import itertools
import json
import time
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(pool_maxsize=16, max_retries=Retry(
    total=6, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=None)))
CONFIG = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"

Z = 16                    # grid in z16 Web-Mercator pixels (1.0-1.6 m/px in Ukraine), like our maps
CELL = 32                 # scan cell = 32 z16 px (~45 m): fine enough for footprint edges
MARGIN_KM = 4.0           # search beyond the AOI box, a flight can start outside the damage zone
SIZES_KM = (3.0, 5.0, 8.0)
WAR0, WAR1, POST0 = "2022-02-24", "2022-12-31", "2023-04-01"
MIN_COVER = 0.02          # ignore captures covering < 2 % of the search box (slivers)
META_LAYER = max(0, 23 - Z)


def merc(lon, lat):
    """lon/lat -> z16 Web-Mercator pixel coords."""
    n = 2 ** Z * 256
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def unmerc(x, y):
    n = 2 ** Z * 256
    return x / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))


def m_per_px(lat):
    return 156543.03392 * math.cos(math.radians(lat)) / 2 ** Z


def all_coords(c):
    if isinstance(c[0], (int, float)):
        yield c
    else:
        for x in c:
            yield from all_coords(x)


def query(cfg, rel, bbox, path):
    """Metadata features of one release inside bbox (cached). Pages through the layer's record limit."""
    if os.path.exists(path):
        return json.load(open(path))
    lon0, lat0, lon1, lat1 = bbox
    feats, off = [], 0
    while True:
        for attempt in range(6):  # the service throws sporadic 500s under load: back off and retry
            try:
                r = SESSION.get(cfg[rel]["metadataLayerUrl"] + f"/{META_LAYER}/query", params=dict(
                    geometry=f"{lon0},{lat0},{lon1},{lat1}", geometryType="esriGeometryEnvelope", inSR=4326,
                    outSR=4326, spatialRel="esriSpatialRelIntersects", outFields="SRC_DATE2,SRC_DESC,SRC_RES,SRC_ACC",
                    returnGeometry="true", resultOffset=off, f="json"), timeout=120).json()
            except (requests.RequestException, ValueError) as e:
                r = {"error": str(e)}
            if "error" not in r:
                break
            time.sleep(2 ** attempt)
        else:  # some releases' metadata services are broken for good: skip them, don't cache
            print(f"  warning: release {rel} metadata unavailable ({r['error']}), skipped", file=sys.stderr)
            return []
        feats += r.get("features", [])
        if not r.get("exceededTransferLimit"):
            break
        off = len(feats)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(feats, open(path, "w"))
    return feats


def footprints(cfg, bbox, cache):
    """{(date, sensor): {'mask': coarse bool raster, 'res', 'acc', 'releases'}} over the search box."""
    x0, y1 = merc(bbox[0], bbox[1]); x1, y0 = merc(bbox[2], bbox[3])
    W, H = int((x1 - x0) // CELL) + 1, int((y1 - y0) // CELL) + 1
    rels = sorted(cfg, key=lambda r: cfg[r]["itemTitle"])
    with ThreadPoolExecutor(8) as ex:
        answers = list(ex.map(lambda r: (r, query(cfg, r, bbox, f"{cache}/{r}.json")), rels))
    caps = {}
    for rel, feats in answers:
        for f in feats:
            a = f["attributes"]
            if a.get("SRC_DATE2") is None or not f.get("geometry"):
                continue
            key = (dt.datetime.fromtimestamp(a["SRC_DATE2"] / 1000, dt.timezone.utc).strftime("%Y-%m-%d"), a["SRC_DESC"])
            c = caps.setdefault(key, {"mask": np.zeros((H, W), np.uint8), "res": a["SRC_RES"], "acc": a["SRC_ACC"],
                                      "releases": set()})
            # one fillPoly per feature: holes (inner rings) are respected by the even-odd fill
            polys = [np.int32(np.round([((merc(lon, lat)[0] - x0) / CELL, (merc(lon, lat)[1] - y0) / CELL)
                                        for lon, lat in ring])) for ring in f["geometry"]["rings"]]
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, polys, 1)
            c["mask"] |= m
            c["releases"].add(rel)
    # erode 1 cell: polygon edges are approximate; the square must not touch them
    k = np.ones((3, 3), np.uint8)
    for c in caps.values():
        c["mask"] = cv2.erode(c["mask"], k).astype(bool)
    return caps, (x0, y0, W, H)


def epoch(date):
    return "pre" if date < WAR0 else "war" if date <= WAR1 else "post" if date >= POST0 else None


def doy_gap(d1, d2):
    a = dt.date.fromisoformat(d1).timetuple().tm_yday; b = dt.date.fromisoformat(d2).timetuple().tm_yday
    return min(abs(a - b), 365 - abs(a - b))


def scan_aoi(aoi, labels, cfg, cache):
    pts = list(all_coords(aoi["geometry"]["coordinates"]))
    lat_c = np.mean([p[1] for p in pts])
    dlat, dlon = MARGIN_KM / 111.0, MARGIN_KM / (111.0 * math.cos(math.radians(lat_c)))
    bbox = (min(p[0] for p in pts) - dlon, min(p[1] for p in pts) - dlat,
            max(p[0] for p in pts) + dlon, max(p[1] for p in pts) + dlat)
    caps, (x0, y0, W, H) = footprints(cfg, bbox, cache)

    # label rasters: destroyed+severe (damage 1-2) and all; and the assessment dates
    lab = np.zeros((H, W), np.float32); lab_all = np.zeros((H, W), np.float32)
    dates = set()
    for f in labels:
        x, y = merc(*f["geometry"]["coordinates"][:2])
        i, j = int((y - y0) // CELL), int((x - x0) // CELL)
        if 0 <= i < H and 0 <= j < W:
            lab_all[i, j] += 1
            if f["properties"]["damage"] in (1, 2):
                lab[i, j] += 1
            dates.add(f["properties"]["date"][:10])

    by_ep = {"pre": [], "war": [], "post": []}
    for key, c in caps.items():
        ep = epoch(key[0])
        if ep and c["mask"].mean() >= MIN_COVER:
            by_ep[ep].append(key)
    mpp = m_per_px(lat_c)
    out = []
    for size_km in SIZES_KM:
        s = max(1, int(round(size_km * 1000 / mpp / CELL)))  # square side in cells
        if s >= min(H, W):
            continue
        k = np.ones((s, s), np.uint8)
        # feasible top-left corners: the whole s x s square inside the footprint
        # (erosion with an anchor at (0,0) gives "square starting here is fully inside")
        feas = {}
        for key in itertools.chain(*by_ep.values()):
            e = cv2.erode(caps[key]["mask"].astype(np.uint8), k, anchor=(0, 0), borderValue=0)
            if e.any():
                feas[key] = e.astype(bool)
        cnt = cv2.boxFilter(lab, -1, (s, s), anchor=(0, 0), normalize=False, borderType=cv2.BORDER_CONSTANT)
        cnt_all = cv2.boxFilter(lab_all, -1, (s, s), anchor=(0, 0), normalize=False, borderType=cv2.BORDER_CONSTANT)
        P = [c for c in by_ep["pre"] if c in feas]; Wr = [c for c in by_ep["war"] if c in feas]
        Po = [c for c in by_ep["post"] if c in feas]
        for pre, war, post in itertools.product(P, Wr, Po):
            ok = feas[pre] & feas[war] & feas[post]
            if not ok.any():
                continue
            sc = np.where(ok, cnt, -1)
            i, j = np.unravel_index(int(np.argmax(sc)), sc.shape)
            cx, cy = x0 + (j + s / 2) * CELL, y0 + (i + s / 2) * CELL
            lon, lat = unmerc(cx, cy)
            trip = (pre, war, post)
            out.append({
                "aoi": aoi["properties"]["aoi"], "size_km": size_km, "center": [round(lat, 5), round(lon, 5)],
                "square_merc_z16": [round(cx - s * CELL / 2), round(cy - s * CELL / 2), s * CELL],
                "destroyed_severe": int(cnt[i, j]), "all_labels": int(cnt_all[i, j]),
                "feasible_area_cells": int(ok.sum()),
                "pre": {"date": pre[0], "sensor": pre[1], "res": caps[pre]["res"], "acc": caps[pre]["acc"]},
                "war": {"date": war[0], "sensor": war[1], "res": caps[war]["res"], "acc": caps[war]["acc"],
                        "is_unosat_date": war[0] in dates},
                "post": {"date": post[0], "sensor": post[1], "res": caps[post]["res"], "acc": caps[post]["acc"]},
                "max_season_gap_days": max(doy_gap(a[0], b[0]) for a, b in itertools.combinations(trip, 2)),
                "same_sensor": len({t[1] for t in trip}) == 1,
                "worst_res": max(caps[t]["res"] or 9 for t in trip),
            })
    summary = {ep: sorted([(k[0], k[1], round(float(caps[k]["mask"].mean()), 3)) for k in v], key=lambda t: t[0])
               for ep, v in by_ep.items()}
    return out, summary, sorted(dates)


def score(c):
    """Rank: damage inside first, then a clean comparison (same season, sensor, sharp, UNOSAT's own image)."""
    return (c["destroyed_severe"]
            * (1.0 if c["war"]["is_unosat_date"] else 0.7)
            * (1.0 - min(c["max_season_gap_days"], 120) / 240)     # 120+ days apart halves the score
            * (1.0 if c["same_sensor"] else 0.85)
            * (1.0 if c["worst_res"] <= 0.6 else 0.8))


def main():
    labels_p, aois_p, cache, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    cfg = SESSION.get(CONFIG, timeout=30).json()
    labels = json.load(open(labels_p))["features"]
    aois = json.load(open(aois_p))["features"]
    allc, summ = [], {}
    for aoi in aois:
        name = aoi["properties"]["aoi"]
        mine = [f for f in labels if f["properties"]["aoi"] == name]
        cities = sorted({f["properties"]["city"] or "?" for f in mine})
        cands, s, dates = scan_aoi(aoi, mine, cfg, f"{cache}/{name}")
        for c in cands:
            c["cities"] = cities; c["score"] = round(score(c), 1)
        allc += cands
        summ[name] = {"cities": cities, "unosat_dates": dates, "captures": s}
        best = max(cands, key=lambda c: c["score"], default=None)
        print(f"{name:6s} {','.join(cities)[:40]:40s} pre {len(s['pre']):2d} war {len(s['war']):2d} post {len(s['post']):2d}"
              f"  feasible triples {len(cands):4d}"
              + (f"  best {best['size_km']}km {best['destroyed_severe']} lbl {best['pre']['date']}/{best['war']['date']}/"
                 f"{best['post']['date']} gap {best['max_season_gap_days']}d" if best else ""), flush=True)
    allc.sort(key=lambda c: -c["score"])
    json.dump({"candidates": allc, "aois": summ}, open(f"{out_dir}/candidates.json", "w"), indent=1)


if __name__ == "__main__":
    main()
