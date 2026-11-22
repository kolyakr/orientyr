"""M1-07 · Trajectory image renderer.

"Done when: the image is readable at full size and as a 1600 px preview." Readability is judged by a human on
the screenshots in the PR; what can be checked is tested here (spec update in the issue):
1. the CLI writes both images at the right sizes; a missing map or run fails clearly;
2. points, segments, footprints and markers are drawn where and how the run says;
3. labels: one per labelled point, inside the image, no overlaps; the legend covers no route point;
4. strokes are equally thick in the preview of a small and a large map.

Most tests use a hand-made run on a plain grey map, so every drawn pixel stands out and every position is known.
"""
import itertools
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronenav.fake_run import make_fake_run
from dronenav.telemetry import Record, RunInfo, RunWriter, write_run_info
from dronenav.types import Fix, TrackMode, TrackState
from dronenav.viz import trajectory_image as ti

ROOT = Path(__file__).resolve().parents[1]
GREY = 128


# --- a hand-made run -------------------------------------------------------------------------------------------

def _H(x, y, heading_deg, scale, frame):
    """Frame px -> map px, conventions of dronenav.types (heading clockwise from up, y down)."""
    w, h = frame
    a = math.radians(heading_deg)
    sR = scale * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    t = np.array([x, y]) - sR @ np.array([w / 2, h / 2])
    return np.array([[*sR[0], t[0]], [*sR[1], t[1]], [0, 0, 1.0]])


def _corners(H, frame):
    w, h = frame
    p = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ H.T
    return p[:, :2] / p[:, 2:]


def _make_run(tmp_path, map_size, pts, missed=(), rejected=(), interpolated=(), frame=(64, 64), scale=1.0,
              interpolated_with_fix=(), idxs=None, heading=None):
    """A run folder over a plain grey map. Points in ``missed`` have no fix (LOST); in ``rejected`` a rejected
    fix far away (RELOCALIZING); in ``interpolated`` an accepted-looking position marked interpolated, no fix."""
    w, h = map_size
    d = tmp_path / f"run_{w}x{h}"
    d.mkdir(parents=True, exist_ok=True)
    m = d / "map.png"
    cv2.imwrite(str(m), np.full((h, w, 3), GREY, np.uint8))
    write_run_info(d, RunInfo(map=str(m), map_size=(w, h), generator="test", frame_size=frame, fps=1.0))
    with RunWriter(d / "run.jsonl") as wr:
        for i, (x, y) in enumerate(pts):
            nxt = pts[min(i + 1, len(pts) - 1)] if i + 1 < len(pts) else pts[i]
            prv = pts[i - 1] if i > 0 else pts[i]
            dx, dy = (nxt[0] - prv[0], nxt[1] - prv[1])
            head = heading if heading is not None else math.degrees(math.atan2(dx, -dy)) % 360 if (dx or dy) else 90.0
            fi = idxs[i] if idxs is not None else i
            fix, state = None, TrackMode.TRACKING
            if i in missed:
                state = TrackMode.LOST
            elif i in rejected:
                state = TrackMode.RELOCALIZING
                fx, fy = w * 0.5, h * 0.12          # a false match away from the route
                H = _H(fx, fy, 0.0, scale, frame)
                fix = Fix(idx=fi, H=H, x=fx, y=fy, heading_deg=0.0, scale=scale, footprint=_corners(H, frame),
                          inliers=6, inlier_ratio=0.05, reproj_px=7.0, accepted=False, reject_reason="too_few_inliers")
            elif i not in interpolated or i in interpolated_with_fix:
                H = _H(x, y, head, scale, frame)
                fix = Fix(idx=fi, H=H, x=x, y=y, heading_deg=head, scale=scale, footprint=_corners(H, frame),
                          inliers=200, inlier_ratio=0.6, reproj_px=0.8, accepted=True)
            track = TrackState(idx=fi, state=state, x=x, y=y, vx=0.0, vy=0.0, cov=np.eye(2), fix=fix)
            wr.write(Record(idx=fi, t=float(i), source=f"frames/frame_{fi:05d}.png", track=track,
                            interpolated=i in interpolated or i in interpolated_with_fix))
    return d


def _line(map_size, n=10, y_frac=0.5):
    """n points on a horizontal line from 10 % to 90 % of the width."""
    w, h = map_size
    return [(w * (0.1 + 0.8 * i / (n - 1)), h * y_frac) for i in range(n)]


def _close(px, colour, tol=40):
    return np.abs(px.astype(int) - np.array(colour)).max() <= tol


def _changed(px, tol=30):
    return np.abs(px.astype(int) - GREY).max() > tol


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    """10 points on a 1600 x 1000 map: 4 and 5 missed (no fix), 6 rejected, 8 interpolated (with an accepted
    fix, so only the interpolated flag makes it missed)."""
    size = (1600, 1000)
    pts = _line(size)
    d = _make_run(tmp_path_factory.mktemp("plain"), size, pts, missed={4, 5}, rejected={6}, interpolated_with_fix={8})
    img, layout = ti.render_trajectory(d)
    return {"dir": d, "pts": pts, "img": img, "layout": layout, "missed": {4, 5, 6, 8}}


# --- 1. the CLI and the two images -----------------------------------------------------------------------------

def _cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "render_image.py"), *map(str, args)],
                          check=False, capture_output=True, text=True)


def test_cli_writes_full_size_and_a_1600_px_preview(tmp_path):
    d = _make_run(tmp_path, (2400, 1500), _line((2400, 1500)))
    res = _cli("--run", d)
    assert res.returncode == 0, res.stderr
    full = cv2.imread(str(d / "trajectory.png"))
    prev = cv2.imread(str(d / "trajectory_preview.jpg"))
    assert full.shape == (1500, 2400, 3)
    assert prev.shape == (1000, 1600, 3)


def test_a_small_map_is_not_upscaled_for_the_preview(tmp_path):
    d = _make_run(tmp_path, (1000, 700), _line((1000, 700)))
    full, prev, _ = ti.write_trajectory(d)
    assert cv2.imread(str(prev)).shape == cv2.imread(str(full)).shape == (700, 1000, 3)


def test_out_option_writes_elsewhere(tmp_path):
    d = _make_run(tmp_path, (1000, 700), _line((1000, 700)))
    res = _cli("--run", d, "--out", tmp_path / "elsewhere")
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "elsewhere" / "trajectory.png").exists()
    assert (tmp_path / "elsewhere" / "trajectory_preview.jpg").exists()
    assert not (d / "trajectory.png").exists()


def test_a_missing_map_is_a_clear_error(tmp_path):
    d = _make_run(tmp_path, (1000, 700), _line((1000, 700)))
    (d / "map.png").unlink()
    res = _cli("--run", d)
    assert res.returncode != 0 and "map.png" in res.stderr


def test_a_missing_run_is_a_clear_error(tmp_path):
    res = _cli("--run", tmp_path / "nothing_here")
    assert res.returncode != 0 and "run.json" in res.stderr


# --- 2. what is drawn, and where -------------------------------------------------------------------------------

def test_points_are_drawn_in_their_style(plain):
    img, pts = plain["img"], plain["pts"]
    for i, (x, y) in enumerate(pts):
        px = img[round(y), round(x)]
        if i == 0:
            assert _close(px, ti.START), i
        elif i == len(pts) - 1:
            assert _close(px, ti.FINISH), i
        elif i in plain["missed"]:
            assert _close(px, ti.MISSED), i
        else:
            assert _close(px, ti.FIXED), i


def _segment_coverage(img, a, b, colour):
    """Fraction of the middle half of segment a-b drawn in ``colour`` (the ends hold the point markers)."""
    hits = 0
    samples = np.linspace(0.25, 0.75, 101)
    for s in samples:
        x, y = a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])
        hits += _close(img[round(y), round(x)], colour)
    return hits / len(samples)


def test_route_segments_are_solid_between_fixes_and_dashed_around_missed_points(plain):
    img, pts, missed = plain["img"], plain["pts"], plain["missed"]
    for i in range(len(pts) - 1):
        if i in missed or i + 1 in missed:
            cov = _segment_coverage(img, pts[i], pts[i + 1], ti.MISSED)
            assert 0.2 <= cov <= 0.8, (i, cov)           # dashes: drawn and gaps
        else:
            assert _segment_coverage(img, pts[i], pts[i + 1], ti.FIXED) > 0.95, i


def test_footprints_of_accepted_fixes_only(plain):
    from dronenav.telemetry import read_run
    img, route_y = plain["img"], plain["pts"][0][1]
    for r in read_run(plain["dir"] / "run.jsonl"):
        f = r.track.fix
        if f is None:
            continue
        fp = f.footprint
        mids = [(fp[k] + fp[(k + 1) % 4]) / 2 for k in range(4)]
        mids = [m for m in mids if abs(m[1] - route_y) > 10]       # off the route line, which runs through two
        for m in mids:
            px = img[round(m[1]), round(m[0])]
            if f.accepted and not r.interpolated:
                assert _changed(px), (r.idx, m)
            else:   # a rejected fix is a false match, an interpolated position was not measured: not drawn
                assert not _changed(px), (r.idx, m)


def test_start_and_finish_are_labelled(plain):
    texts = {lab.text for lab in plain["layout"].labels}
    assert "START" in texts and "FINISH" in texts


# --- 3. labels and legend --------------------------------------------------------------------------------------

def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _inside(box, shape):
    return box[0] >= 0 and box[1] >= 0 and box[2] <= shape[1] and box[3] <= shape[0]


def test_every_point_is_numbered_by_its_frame_index(plain):
    numbered = {lab.idx: lab.text for lab in plain["layout"].labels if lab.idx is not None}
    assert numbered == {i: str(i) for i in range(len(plain["pts"]))}


def test_labels_are_inside_the_image_and_do_not_overlap(plain):
    labels, img = plain["layout"].labels, plain["img"]
    for lab in labels:
        assert _inside(lab.box, img.shape), lab
        x0, y0, x1, y1 = (int(v) for v in lab.box)
        assert _changed(img[y0:y1, x0:x1].reshape(-1, 3).max(axis=0)), lab     # something is written there
    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            assert not _overlap(labels[a].box, labels[b].box), (labels[a], labels[b])


def test_labels_do_not_cover_other_points(plain):
    for lab in plain["layout"].labels:
        for i, (x, y) in enumerate(plain["pts"]):
            if i != lab.idx:
                assert not (lab.box[0] <= x <= lab.box[2] and lab.box[1] <= y <= lab.box[3]), (lab, i)


def test_dense_runs_are_thinned_but_keep_start_and_finish(tmp_path):
    size = (1600, 1000)
    pts = [(200 + 4 * i, 500 + 80 * math.sin(i / 15)) for i in range(300)]       # 4 px apart: too dense to label all
    d = _make_run(tmp_path, size, pts, frame=(16, 16))
    _, layout = ti.render_trajectory(d)
    numbered = [lab for lab in layout.labels if lab.idx is not None]
    assert 5 < len(numbered) < 300 and layout.unlabelled == []      # thinned on purpose, none dropped
    texts = {lab.text for lab in layout.labels}
    assert {"START", "FINISH"} <= texts
    for a in range(len(layout.labels)):
        for b in range(a + 1, len(layout.labels)):
            assert not _overlap(layout.labels[a].box, layout.labels[b].box)


def test_label_every_overrides_the_thinning(tmp_path):
    size = (1600, 1000)
    d = _make_run(tmp_path, size, _line(size, n=20))
    _, layout = ti.render_trajectory(d, label_every=5)
    assert sorted(lab.idx for lab in layout.labels if lab.idx is not None) == [0, 5, 10, 15, 19]   # + the last


def test_legend_is_drawn_and_covers_no_route_point(plain):
    box, img = plain["layout"].legend, plain["img"]
    assert _inside(box, img.shape)
    for x, y in plain["pts"]:
        assert not (box[0] <= x <= box[2] and box[1] <= y <= box[3])
    x0, y0, x1, y1 = (int(v) for v in box)
    assert np.mean([_changed(px) for px in img[y0:y1, x0:x1].reshape(-1, 3)]) > 0.5


def test_legend_avoids_the_corner_the_route_passes(tmp_path):
    size = (1600, 1000)
    pts = [(80 + 30 * i, 80 + 4 * i) for i in range(12)]        # through the top-left corner
    d = _make_run(tmp_path, size, pts)
    _, layout = ti.render_trajectory(d)
    x0, y0, _, _ = layout.legend
    assert x0 > size[0] / 2 or y0 > size[1] / 2


# --- 4. sizes scale with the map -------------------------------------------------------------------------------

def _stroke_in_preview(tmp_path, size):
    d = _make_run(tmp_path, size, _line(size, n=5), frame=(16, 16), scale=size[0] / 1600)
    _, prev_path, _ = ti.write_trajectory(d)
    prev = cv2.imread(str(prev_path))
    k = prev.shape[1] / size[0]
    x = int((size[0] * 0.2) * k)                              # between points 0 and 1, away from markers
    y = int(size[1] * 0.5 * k)
    col = prev[max(y - 30, 0):y + 30, x]
    return int(sum(_close(px, ti.FIXED, tol=60) for px in col))


def test_strokes_look_the_same_in_the_preview_of_small_and_large_maps(tmp_path):
    small = _stroke_in_preview(tmp_path, (800, 500))
    large = _stroke_in_preview(tmp_path, (6400, 4000))
    assert small >= 2 and large >= 2
    assert abs(small - large) <= 1, (small, large)


# --- on the fake run, and the real task map --------------------------------------------------------------------

def test_renders_a_fake_run(tmp_path):
    rng = np.random.default_rng(0)
    m = tmp_path / "map.png"
    cv2.imwrite(str(m), cv2.GaussianBlur(rng.uniform(0, 255, (900, 1400, 3)).astype(np.uint8), (0, 0), 2))
    make_fake_run(m, tmp_path / "run")
    full, _, _ = ti.write_trajectory(tmp_path / "run")
    assert cv2.imread(str(full)).shape == (900, 1400, 3)
    _, layout = ti.render_trajectory(tmp_path / "run")
    assert {lab.idx for lab in layout.labels if lab.idx is not None} == set(range(50))


TASK_MAP = ROOT / "data" / "hackatons-data" / "global_map.png"


@pytest.mark.data
@pytest.mark.skipif(not TASK_MAP.exists(), reason="data/ not downloaded")
def test_task_map_renders_quickly(tmp_path):
    make_fake_run(TASK_MAP, tmp_path / "run")
    t0 = time.perf_counter()
    _, prev, _ = ti.write_trajectory(tmp_path / "run")
    assert time.perf_counter() - t0 < 10
    assert cv2.imread(str(prev)).shape == (810, 1600, 3)


# --- from the review -------------------------------------------------------------------------------------------

def test_a_map_that_does_not_match_run_json_is_refused(tmp_path):
    # a replaced map, or another zoom of the same area: every position would be drawn in the wrong place
    d = _make_run(tmp_path, (1600, 1000), _line((1600, 1000)))
    cv2.imwrite(str(d / "map.png"), np.full((500, 800, 3), GREY, np.uint8))
    with pytest.raises(ValueError, match=r"800 x 500.*1600 x 1000"):
        ti.render_trajectory(d)
    res = _cli("--run", d)
    assert res.returncode != 0 and "1600 x 1000" in res.stderr


@pytest.mark.parametrize("channels", [1, 2, 4])
def test_a_map_given_in_memory_may_be_grey_or_have_alpha(plain, channels):
    # run.py may hand over the map it already loaded; the drawing must stay in colour
    grey = np.full((1000, 1600), GREY, np.uint8)
    m = {1: grey, 2: grey[..., None], 4: np.dstack([grey] * 3 + [np.full_like(grey, 255)])}[channels]
    img, _ = ti.render_trajectory(plain["dir"], map_img=m)
    assert img.shape == (1000, 1600, 3)
    x, y = plain["pts"][1]
    assert _close(img[round(y), round(x)], ti.FIXED)


def test_labels_follow_the_spacing_not_the_median_step(tmp_path):
    # hovering makes the median step 0: thinning by it labelled every 2nd point and then dropped a quarter
    pts = [(300, 500)] * 40 + [(300 + 25 * k, 500) for k in range(1, 40)]
    _, layout = ti.render_trajectory(_make_run(tmp_path, (1600, 1000), pts))
    numbered = {lab.idx for lab in layout.labels if lab.idx is not None}
    assert set(range(40, 79)) <= numbered          # every moving point, 25 px apart, has room for its number
    assert len(numbered & set(range(40))) == 1    # the hover is labelled once
    assert layout.unlabelled == []


def test_ends_outside_the_map_still_get_start_and_finish(tmp_path):
    # a dead-reckoned position can leave the map; START and FINISH are always shown, at the map's edge
    d = _make_run(tmp_path, (1600, 1000), [(-200, 500), (400, 500), (1200, 500), (1800, 500)])
    img, layout = ti.render_trajectory(d)
    tags = {lab.text: lab.box for lab in layout.labels if lab.idx is None}
    assert set(tags) == {"START", "FINISH"}
    for box in tags.values():
        assert _inside(box, img.shape)
    edge = round((ti.END_R + 2) * layout.unit)                       # the marker's centre, pinned to the edge
    assert _close(img[500, edge], ti.START) and _close(img[500, 1600 - edge], ti.FINISH)


def test_the_legend_is_left_out_when_the_map_is_too_small(tmp_path):
    d = _make_run(tmp_path, (200, 150), [(40, 75), (100, 75), (160, 75)], frame=(16, 16))
    _, layout = ti.render_trajectory(d)
    assert layout.legend is None
    assert {lab.idx for lab in layout.labels if lab.idx is not None} == {0, 1, 2}


def test_the_legend_fits_a_narrow_map(tmp_path):
    d = _make_run(tmp_path, (1800, 4000), [(900, 400 + 400 * i) for i in range(9)])
    img, layout = ti.render_trajectory(d)
    assert layout.legend is not None and _inside(layout.legend, img.shape)


def _hits_route(box, pts, step=1.0):
    """Does the box touch the polyline through pts (sampled every ``step`` px)?"""
    for a, b in itertools.pairwise(pts):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) / step))
        for s in np.linspace(0, 1, n + 1):
            x, y = a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                return True
    return False


def _near_marker(box, x, y, r):
    cx, cy = min(max(x, box[0]), box[2]), min(max(y, box[1]), box[3])
    return math.hypot(cx - x, cy - y) < r


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake")
    rng = np.random.default_rng(0)
    m = d / "map.png"
    cv2.imwrite(str(m), cv2.GaussianBlur(rng.uniform(0, 255, (900, 1400, 3)).astype(np.uint8), (0, 0), 2))
    make_fake_run(m, d / "run")
    from dronenav.telemetry import read_run
    pts = [(r.track.x, r.track.y) for r in read_run(d / "run" / "run.jsonl")]
    img, layout = ti.render_trajectory(d / "run")
    return {"pts": pts, "img": img, "layout": layout}


def test_on_a_curved_route_labels_cover_no_marker_and_no_route(fake):
    # the straight plain run checks only point centres; here the route curves and markers have a radius
    r = ti.END_R * fake["layout"].unit
    for lab in fake["layout"].labels:
        assert not _hits_route(lab.box, fake["pts"]), lab
        for i, (x, y) in enumerate(fake["pts"]):
            if i != lab.idx:
                assert not _near_marker(lab.box, x, y, r), (lab, i)
    assert fake["layout"].unlabelled == []


def test_on_a_curved_route_the_legend_covers_no_marker_and_no_route(fake):
    box = fake["layout"].legend
    assert box is not None and not _hits_route(box, fake["pts"])
    assert not any(_near_marker(box, x, y, ti.END_R * fake["layout"].unit) for x, y in fake["pts"])


def test_strokes_match_at_a_fractional_scale_too(tmp_path):
    assert abs(_stroke_in_preview(tmp_path, (800, 500)) - _stroke_in_preview(tmp_path, (2600, 1625))) <= 1


def test_an_interpolated_point_gets_no_footprint(tmp_path):
    # interpolated means "not measured": it is drawn as missed, so its footprint is not drawn either
    pts = _line((1600, 1000))
    d = _make_run(tmp_path, (1600, 1000), pts, interpolated_with_fix={4})
    img, _ = ti.render_trajectory(d)
    x, y = pts[4]
    assert not _changed(img[round(y - 32), round(x)])          # the footprint's side, 32 px off the route
    x, y = pts[3]
    assert _changed(img[round(y - 32), round(x)])


@pytest.mark.parametrize("size", [(3001, 1777), (1777, 3001), (1601, 1601)])
def test_the_preview_long_side_is_exactly_the_preview_size(tmp_path, size):
    d = _make_run(tmp_path, size, _line(size, n=4), frame=(16, 16))
    _, prev, _ = ti.write_trajectory(d)
    assert max(cv2.imread(str(prev)).shape[:2]) == 1600


# --- third review: crowded routes, off-map points, the legend's corner ------------------------------------------

def test_numbers_keep_their_spacing_on_a_zigzag(tmp_path):
    # a zigzag point is far from the last numbered point but on top of the one before: spacing must hold
    # against every numbered point, or a third of the numbers collide and are dropped
    pts = [(600 + 400 * i / 119, 400 if i % 2 else 700) for i in range(120)]     # points two apart: 6.7 px
    _, layout = ti.render_trajectory(_make_run(tmp_path, (1600, 1000), pts, frame=(16, 16)))
    numbered = [pts[lab.idx] for lab in layout.labels if lab.idx is not None]
    assert len(numbered) > 15 and layout.unlabelled == []
    for a, b in itertools.combinations(numbered, 2):
        assert math.hypot(a[0] - b[0], a[1] - b[1]) >= ti.MIN_LABEL_SPACING * layout.unit


def test_points_off_the_map_are_counted_not_numbered(tmp_path):
    pts = [(-500 + 2600 * i / 39, 500 - 5 * i) for i in range(40)]
    _, layout = ti.render_trajectory(_make_run(tmp_path, (1600, 1000), pts, frame=(16, 16)))
    off = [i for i, (x, y) in enumerate(pts) if not (0 <= x <= 1600 and 0 <= y <= 1000)]
    assert layout.off_map == off and len(off) > 10
    assert not {lab.idx for lab in layout.labels} & set(off)
    assert layout.unlabelled == []


def test_dense_points_are_dots_so_the_dashed_route_stays_visible(tmp_path):
    # 3 px apart: full markers would bury the route; unnumbered points are small dots and the dashes run on
    pts = [(200 + 3 * i, 500.0) for i in range(300)]
    img, layout = ti.render_trajectory(_make_run(tmp_path, (1600, 1000), pts, missed=set(range(1, 299))))
    numbered = {lab.idx for lab in layout.labels if lab.idx is not None}
    plain = [i for i in range(1, 299) if all(abs(i - j) >= 3 for j in numbered)]   # ≥ 9 px from full markers
    assert len(plain) > 30

    def under_a_label(x, y):
        return any(b.box[0] <= x <= b.box[2] and b.box[1] <= y <= b.box[3] for b in layout.labels)

    beside = [_changed(img[round(pts[i][1] - 4), round(pts[i][0])]) for i in plain   # 4 px off the route line
              if not under_a_label(pts[i][0], pts[i][1] - 4)]
    assert len(beside) > 15
    assert sum(beside) / len(beside) < 0.1
    # along the line: gaps between dashes must show (a solid line, or one buried under markers, has none);
    # numbered markers' black outlines would also make "gaps", so this run is redrawn with only the ends numbered
    img, _ = ti.render_trajectory(tmp_path / "run_1600x1000", label_every=10_000)
    row = [_close(img[500, x], ti.MISSED) for x in range(round(pts[30][0]), round(pts[270][0]))]
    gaps = sum(1 for a, b in itertools.pairwise(row) if a and not b)
    assert gaps >= 20, gaps                                     # ≈ 720 px of dash + gap periods of 14 px


def test_the_legend_skips_corners_with_points_but_still_appears(tmp_path):
    # points sit in three corners; the fourth is only crossed by the route, and that is where the legend goes
    pts = [(60, 60), (1540, 60), (60, 940), (1650, 940), (1650, 500)]
    _, layout = ti.render_trajectory(_make_run(tmp_path, (1600, 1000), pts, frame=(16, 16)))
    assert layout.legend is not None
    x0, y0, x1, y1 = layout.legend
    r = ti.END_R * layout.unit
    markers = [(60, 60), (1540, 60), (60, 940), (1600 - (ti.END_R + 2) * layout.unit, 500)]   # FINISH at the edge
    assert not any(x0 - r <= x <= x1 + r and y0 - r <= y <= y1 + r for x, y in markers)


@pytest.mark.parametrize("kw", [{"label_every": 0}, {"label_every": -2}, {"preview_px": 0}])
def test_bad_arguments_are_refused(plain, kw):
    with pytest.raises(ValueError):
        ti.render_trajectory(plain["dir"], **kw)


def test_on_a_dense_run_footprints_follow_the_numbering(tmp_path):
    # every footprint of a dense run makes a solid band; one per numbered point keeps the map visible
    pts = [(200 + 3 * i, 500.0) for i in range(300)]
    _, layout = ti.render_trajectory(_make_run(tmp_path, (1600, 1000), pts, frame=(16, 16)))
    numbered = sorted(lab.idx for lab in layout.labels if lab.idx is not None)
    assert layout.footprints == numbered and len(numbered) < 100


def test_on_a_sparse_run_every_accepted_fix_has_its_footprint(plain):
    assert plain["layout"].footprints == [0, 1, 2, 3, 7, 9]      # not 4, 5 (no fix), 6 (rejected), 8 (interpolated)


def test_long_runs_space_numbers_by_the_width_of_their_longest_number(tmp_path):
    # 18 px fits "12", not "1187": with 4-digit frame numbers the labels stacked into a dense column
    pts = [(150 + 1.1 * i, 500 + 200 * math.sin(i / 200)) for i in range(1200)]
    _, layout = ti.render_trajectory(_make_run(tmp_path, (1600, 1000), pts, frame=(16, 16)))
    labs = [lab for lab in layout.labels if lab.idx is not None]
    widest = max(lab.box[2] - lab.box[0] for lab in labs)
    assert widest > ti.MIN_LABEL_SPACING * layout.unit                 # the case this is about
    for a, b in itertools.combinations(labs, 2):
        assert math.hypot(pts[a.idx][0] - pts[b.idx][0], pts[a.idx][1] - pts[b.idx][1]) >= widest, (a.text, b.text)



# --- fourth review: rules that agree with each other, tests that catch mutations, an API for run.py ----------

def test_the_colours_are_far_apart():
    # "a second colour" for missed points: the tests compare with these constants, so they must differ clearly
    palette = {"FIXED": ti.FIXED, "MISSED": ti.MISSED, "START": ti.START, "FINISH": ti.FINISH,
               "FOOTPRINT": ti.FOOTPRINT}
    for (a, ca), (b, cb) in itertools.combinations(palette.items(), 2):
        assert np.abs(np.array(ca) - np.array(cb)).max() >= 100, (a, b)


def test_a_typo_in_the_run_folder_creates_nothing(tmp_path):
    with pytest.raises(FileNotFoundError):
        ti.write_trajectory(tmp_path / "typo")
    assert not (tmp_path / "typo").exists()


def test_footprints_follow_the_numbering_with_label_every_too(tmp_path):
    pts = [(200 + 3 * i, 500.0) for i in range(300)]
    _, layout = ti.render_trajectory(_make_run(tmp_path, (1600, 1000), pts, frame=(16, 16)), label_every=100)
    assert layout.footprints == [0, 100, 200, 299]


def test_the_summary_adds_up(tmp_path):
    # off-map points are dead-reckoned, so they are also missed: "(K off the map)" is a note, not a category
    pts = [(-300 + 2200 * i / 19, 500) for i in range(20)]
    _, layout = ti.render_trajectory(_make_run(tmp_path, (1600, 1000), pts, missed={1, 2, 3}, frame=(16, 16)))
    assert layout.summary == f"20 frames, 17 fixed, 3 missed ({len(layout.off_map)} off the map)"


def test_the_legend_tries_the_middle_of_an_edge_when_the_corners_are_taken(tmp_path):
    pts = [(60, 60), (1540, 60), (1540, 940), (60, 940), (800, 500)]
    _, layout = ti.render_trajectory(_make_run(tmp_path, (1600, 1000), pts, frame=(16, 16)))
    assert layout.legend is not None
    x0, y0, x1, y1 = layout.legend
    r = ti.END_R * layout.unit
    assert not any(x0 - r <= x <= x1 + r and y0 - r <= y <= y1 + r for x, y in pts)


def test_the_legend_takes_the_free_corner_the_route_does_not_cross(tmp_path):
    # top-left holds a point; top-right is crossed by the route; bottom-left and bottom-right are free, and the
    # route clips bottom-right: the legend goes bottom-left
    pts = [(60, 60), (1700, 60), (1700, 950), (1300, 950), (800, 500)]
    _, layout = ti.render_trajectory(_make_run(tmp_path, (1600, 1000), pts, frame=(16, 16)))
    x0, y0, _, _ = layout.legend
    assert x0 < 800 and y0 > 500


def test_numbers_are_frame_numbers_when_frames_are_skipped(tmp_path):
    # a video sampled every 3rd frame: point N is frame N, and label_every counts frames, not records
    size = (1600, 1000)
    idxs = [100 + 3 * i for i in range(20)]
    d = _make_run(tmp_path, size, _line(size, n=20), idxs=idxs)
    _, layout = ti.render_trajectory(d)
    assert {lab.text for lab in layout.labels if lab.idx is not None} == {str(i) for i in idxs}
    _, layout = ti.render_trajectory(d, label_every=10)
    assert sorted(lab.idx for lab in layout.labels if lab.idx is not None) == [100, 130, 157]   # 157: the last


def test_an_end_near_the_edge_stays_where_it_is(tmp_path):
    # only ends off the map are pinned to the edge; one on the map is drawn at its position
    d = _make_run(tmp_path, (1600, 1000), [(2, 500), (800, 500), (1500, 500)], frame=(16, 16))
    img, _ = ti.render_trajectory(d)
    assert _close(img[500, 2], ti.START) and not _close(img[500, 12], ti.START)


def test_footprints_are_rotated(tmp_path):
    # a wide frame at 30°: its edge midpoints are drawn, the corners of its axis-aligned box are not
    frame, head = (96, 32), 30.0
    d = _make_run(tmp_path, (1600, 1000), [(400, 500), (800, 500), (1200, 500)], frame=frame, heading=head)
    img, _ = ti.render_trajectory(d)
    H = _H(800, 500, head, 1.0, frame)
    c = _corners(H, frame)
    for k in (0, 2):                                                   # TL-TR and BR-BL: the long sides
        mx, my = (c[k] + c[(k + 1) % 4]) / 2
        assert _changed(img[round(my), round(mx)]), k
    for x, y in ((c[:, 0].min(), c[:, 1].min()), (c[:, 0].max(), c[:, 1].max())):
        assert not _changed(img[round(y), round(x)])


@pytest.mark.parametrize("kind", ["grey+alpha", "uint16"])
def test_more_map_formats_in_memory(plain, kind):
    grey = np.full((1000, 1600), GREY, np.uint8)
    m = np.dstack([grey, np.full_like(grey, 255)]) if kind == "grey+alpha" else np.dstack([grey.astype(np.uint16) * 257] * 3)
    img, _ = ti.render_trajectory(plain["dir"], map_img=m)
    assert img.dtype == np.uint8 and img.shape == (1000, 1600, 3)
    assert abs(int(img[800, 100, 0]) - GREY) <= 1                      # bare map, away from route and legend


def test_render_from_memory_matches_the_folder(plain):
    # run.py has the records and the map in memory already: no second read, same picture
    from dronenav.telemetry import read_run, read_run_info
    records = read_run(plain["dir"] / "run.jsonl")
    info = read_run_info(plain["dir"])
    img, layout = ti.render_records(records, info, np.full((1000, 1600, 3), GREY, np.uint8))
    assert np.array_equal(img, plain["img"]) and layout == plain["layout"]


def test_write_takes_a_map_in_memory_and_returns_the_layout(tmp_path):
    d = _make_run(tmp_path, (1000, 700), _line((1000, 700)))
    (d / "map.png").unlink()                                           # proof that the file is not read
    full, prev, layout = ti.write_trajectory(d, map_img=np.full((700, 1000, 3), GREY, np.uint8))
    assert full.exists() and prev.exists() and layout.unlabelled == []


def test_start_and_finish_are_shown_even_where_every_spot_is_taken(tmp_path):
    # the route radiates from the start in 16 directions: no free spot for its tag, which must still show
    pts = [(800, 500)]
    for k in range(16):
        a = 2 * math.pi * k / 16
        pts += [(800 + 90 * math.cos(a), 500 + 90 * math.sin(a)), (800, 500)]
    pts.append((1400, 500))
    _, layout = ti.render_trajectory(_make_run(tmp_path, (1600, 1000), pts, frame=(16, 16)))
    assert {"START", "FINISH"} <= {lab.text for lab in layout.labels}
