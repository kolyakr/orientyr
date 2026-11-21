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

There is no separate AOI-polygon file in our data, so each area's search box comes from the extent
of its own UNOSAT labels (see `bbox_from_points`), expanded by MARGIN_KM.

Writes <out>/candidates.json (every AOI x size x triple that has a feasible square, ranked) and
results/candidates_summary.json (the best candidate per area and size, committed) next to this
script, and prints a summary table. Metadata answers are cached in <cache>/<aoi>/<release>.json.

Usage: python scan.py <unosat_labels.geojson> <cache_dir> <out_dir>
"""
import datetime as dt
import itertools
import json
import math
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request

import cv2
import numpy as np

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # no requests/certifi dependency: fall back to the OS's own CA bundle
    _SSL_CONTEXT = ssl.create_default_context()
urlopen = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_SSL_CONTEXT)).open

CONFIG = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

Z = 16                    # grid in z16 Web-Mercator pixels (1.0-1.6 m/px in Ukraine), like our maps
CELL = 32                 # scan cell = 32 z16 px (~45 m): fine enough for footprint edges
MARGIN_KM = 4.0           # search beyond the AOI box, a flight can start outside the damage zone
SIZES_KM = (3.0, 5.0, 8.0)
WAR0, WAR1, POST0 = "2022-02-24", "2022-12-31", "2023-04-01"
PRE_2019 = "2019-01-01"   # a "pre" capture older than this is a weak before/after comparison
MIN_COVER = 0.02          # ignore captures covering < 2 % of the search box (slivers)
META_LAYER = max(0, 23 - Z)

# Occupied / front-line status per UNOSAT area, looked up 2026-11-21 against ISW/Critical Threats'
# daily control-of-terrain assessments (criticalthreats.org, "Russian offensive campaign assessment",
# Nov 2026 editions) plus each area's own occupation history. "occupied" = under Russian control;
# "front_line" = Ukrainian-held but adjacent to active fighting or under regular fire; "none" = rear
# area, no current front-line activity. An area scanned that is missing here is an error (see
# status_for), never silently "none".
STATUS = {
    "UKR1": "occupied",     # Mariupol (Donetsk) -- occupied since May 2022
    "UKR2": "none",         # Bucha/Hostomel/Irpin/Moschun/Vorzel, Antonov airport (Kyiv) -- liberated Mar 2022, rear since
    "UKR3": "occupied",     # Lysychansk/Rubizhne/Sievierodonetsk (Luhansk) -- occupied since Jul 2022
    "UKR4": "occupied",     # Volnovakha (Donetsk) -- occupied since Mar 2022
    "UKR5": "occupied",     # Avdiivka (Donetsk) -- occupied since Feb 2024
    "UKR6": "none",         # Chernihiv -- siege lifted Apr 2022, no front-line activity since
    "UKR7": "front_line",   # Kharkiv -- near the Vovchansk/Kupiansk footholds, regularly shelled (CT, Aug-Nov 2026)
    "UKR8": "none",         # Borodyanka (Kyiv) -- liberated Mar 2022, rear since
    "UKR9": "none",         # Makariv (Kyiv) -- liberated Mar 2022, rear since
    "UKR10": "none",        # Mykolaiv -- front-line 2022-23, rear since the Kherson liberation (Nov 2022)
    "UKR11": "occupied",    # Shchastia (Luhansk) -- occupied since 2022
    "UKR12": "front_line",  # Sumy -- active Russian offensive operations north of Sumy City (CT, Nov 3-4 2026)
    "UKR13": "none",        # Trostianets (Sumy) -- liberated Mar 2022, south of the active border fighting
    "UKR14": "front_line",  # Kramatorsk (Donetsk) -- Ukraine's main Donbas stronghold, directly contested (CT, 2026)
    "UKR15": "none",        # Okhtyrka (Sumy) -- liberated Mar 2022, interior
    "UKR16": "front_line",  # Kherson/Antonivka -- right bank, faces the occupied left bank across the Dnipro
    "UKR17": "none",        # Kremenchuk (Poltava) -- interior, no front-line
    "UKR18": "occupied",    # Melitopol (Zaporizhzhia) -- occupied since 2022
}
STATUS_PENALTY = {"occupied": 0.2, "front_line": 0.5, "none": 1.0}


def status_for(aoi):
    """STATUS[aoi], or an error -- a scanned area missing from STATUS must be added, not assumed safe."""
    if aoi not in STATUS:
        raise ValueError(f"no status recorded for area {aoi!r}: add it to STATUS, don't assume 'none'")
    return STATUS[aoi]


def merc(lon, lat):
    """lon/lat -> z16 Web-Mercator pixel coords."""
    n = 2 ** Z * 256
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def unmerc(x, y):
    n = 2 ** Z * 256
    return x / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))


def m_per_px(lat):
    return 156543.03392 * math.cos(math.radians(lat)) / 2 ** Z


def bbox_from_points(pts, margin_km=MARGIN_KM):
    """lon/lat bounding box of `pts`, expanded by `margin_km` on every side (flat-earth approximation,
    like the rest of this module). Used instead of an AOI-polygon file, which we don't have."""
    if not pts:
        raise ValueError("no points to derive a bounding box from")
    lat_c = sum(p[1] for p in pts) / len(pts)
    dlat, dlon = margin_km / 111.0, margin_km / (111.0 * math.cos(math.radians(lat_c)))
    return (min(p[0] for p in pts) - dlon, min(p[1] for p in pts) - dlat,
            max(p[0] for p in pts) + dlon, max(p[1] for p in pts) + dlat)


def _fetch_json(url, params, timeout, attempts=6):
    """GET url?params as JSON, retrying 429/5xx and connection errors with exponential backoff."""
    qs = "?" + urllib.parse.urlencode(params) if params else ""
    for attempt in range(attempts):
        try:
            with urlopen(Request(url + qs), timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code not in RETRYABLE_STATUS or attempt == attempts - 1:
                raise
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
        time.sleep(2 ** attempt)


def query(cfg, rel, bbox, path):
    """Metadata features of one release inside bbox (cached). Pages through the layer's record limit."""
    if os.path.exists(path):
        return json.load(open(path))
    lon0, lat0, lon1, lat1 = bbox
    feats, off = [], 0
    while True:
        try:
            r = _fetch_json(cfg[rel]["metadataLayerUrl"] + f"/{META_LAYER}/query", dict(
                geometry=f"{lon0},{lat0},{lon1},{lat1}", geometryType="esriGeometryEnvelope", inSR=4326,
                outSR=4326, spatialRel="esriSpatialRelIntersects", outFields="SRC_DATE2,SRC_DESC,SRC_RES,SRC_ACC",
                returnGeometry="true", resultOffset=off, f="json"), timeout=120)
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            # some releases' metadata services are broken for good: skip them, don't cache
            print(f"  warning: release {rel} metadata unavailable ({e}), skipped", file=sys.stderr)
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


def scan_aoi(name, labels, cfg, cache):
    pts = [tuple(f["geometry"]["coordinates"][:2]) for f in labels]
    bbox = bbox_from_points(pts)
    caps, (x0, y0, W, H) = footprints(cfg, bbox, cache)
    lat_c = sum(p[1] for p in pts) / len(pts)

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
                "aoi": name, "size_km": size_km, "center": [round(lat, 5), round(lon, 5)],
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


def penalties(c):
    """Named, independent multiplicative factors behind score() -- each documents one ranking rule."""
    return {
        "pre_2019": 0.7 if c["pre"]["date"] < PRE_2019 else 1.0,
        "off_date": 1.0 if c["war"]["is_unosat_date"] else 0.7,
        "season_gap": 1.0 - min(c["max_season_gap_days"], 120) / 240,     # 120+ days apart halves the score
        "mixed_sensor": 1.0 if c["same_sensor"] else 0.85,
        "resolution": 1.0 if c["worst_res"] <= 0.6 else 0.8,
        "status": STATUS_PENALTY[status_for(c["aoi"])],
    }


def score(c):
    """Rank: damage inside first, then a clean, safe comparison (penalties())."""
    p = 1.0
    for factor in penalties(c).values():
        p *= factor
    return c["destroyed_severe"] * p


def summarize(candidates):
    """Best-scoring candidate per (aoi, size_km) -- what results/candidates_summary.json commits."""
    best = {}
    for c in candidates:
        key = (c["aoi"], c["size_km"])
        if key not in best or c["score"] > best[key]["score"]:
            best[key] = c
    return [best[k] for k in sorted(best)]


def main():
    labels_p, cache, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    cfg = _fetch_json(CONFIG, None, timeout=30)
    labels = json.load(open(labels_p))["features"]
    by_aoi = {}
    for f in labels:
        by_aoi.setdefault(f["properties"]["aoi"], []).append(f)
    allc, summ = [], {}
    for name in sorted(by_aoi, key=lambda n: int(n[3:])):
        mine = by_aoi[name]
        cities = sorted({f["properties"]["city"] or "?" for f in mine})
        cands, s, dates = scan_aoi(name, mine, cfg, f"{cache}/{name}")
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
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    json.dump(summarize(allc), open(os.path.join(results_dir, "candidates_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
